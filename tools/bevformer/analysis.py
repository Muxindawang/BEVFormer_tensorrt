import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 👈 强制使用 1 号 GPU

import argparse
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
import torch
from mmcv.parallel import MMDataParallel
import mmcv
import time
import copy
import sys
sys.path.append(".")

from third_party.bev_mmdet3d.models.builder import build_model
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset

# ----------------------------
# Fake Quantization 模拟
# ----------------------------
from torch.quantization import FakeQuantize

def get_fake_quant():
    return FakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False
    )().cuda()

# ----------------------------
# Hook 工具：给指定模块输出加 fake quant
# ----------------------------
hooks = []

def add_output_fake_quant(model, module_name):
    """给 model.module 中名为 module_name 的子模块输出添加 fake quant"""
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor) and output.is_floating_point():
            return get_fake_quant()(output)
        elif isinstance(output, (list, tuple)):
            new_out = []
            for o in output:
                if isinstance(o, torch.Tensor) and o.is_floating_point():
                    new_out.append(get_fake_quant()(o))
                else:
                    new_out.append(o)
            return type(output)(new_out)
        return output

    target_module = None
    for name, mod in model.module.named_modules():
        if name == module_name:
            target_module = mod
            break
    if target_module is None:
        raise ValueError(f"Module {module_name} not found!")

    handle = target_module.register_forward_hook(hook_fn)
    hooks.append(handle)
    print(f"✅ Added fake quant to output of: {module_name}")

def quantize_module_recursively(target_module):
    """
    对 target_module 及其所有子模块递归插入 fake quant：
      - 权重（如果存在）
      - 输入（通过 pre-hook）
      - 输出（通过 post-hook）
    """
    hooks = []

    def _add_weight_quant(m):
        if hasattr(m, 'weight') and m.weight.is_floating_point():
            # 注意：这里不能直接修改 m.weight，而是用 pre-hook 在 forward 前量化
            def weight_pre_hook(module, input):
                module.weight.data = get_fake_quant()(module.weight.data)
            h = m.register_forward_pre_hook(weight_pre_hook)
            hooks.append(h)

    def _add_io_quant(m):
        # Input quant (pre-hook)
        def input_pre_hook(module, input):
            if isinstance(input, torch.Tensor) and input.is_floating_point():
                return (get_fake_quant()(input),)
            elif isinstance(input, (tuple, list)):
                new_input = []
                for x in input:
                    if isinstance(x, torch.Tensor) and x.is_floating_point():
                        new_input.append(get_fake_quant()(x))
                    else:
                        new_input.append(x)
                return tuple(new_input)
            return input
        h1 = m.register_forward_pre_hook(input_pre_hook)
        hooks.append(h1)

        # Output quant (post-hook)
        def output_hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.is_floating_point():
                return get_fake_quant()(output)
            elif isinstance(output, (list, tuple)):
                new_out = []
                for o in output:
                    if isinstance(o, torch.Tensor) and o.is_floating_point():
                        new_out.append(get_fake_quant()(o))
                    else:
                        new_out.append(o)
                return type(output)(new_out)
            return output
        h2 = m.register_forward_hook(output_hook)
        hooks.append(h2)

    # 遍历所有子模块（包括自己）
    modules_to_quant = []
    for name, submod in target_module.named_modules():
        # 跳过容器类（如 Sequential, ModuleList），只处理有计算的层
        if isinstance(submod, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear,
                               torch.nn.BatchNorm2d, torch.nn.LayerNorm,
                               torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU,
                               torch.nn.Identity)):
            modules_to_quant.append(submod)
        # 注意：Transformer 层中的 MultiheadAttention 可能需要特殊处理，但 BEVFormer 通常用自定义 Attention

    # 对每个可量化层添加 weight + I/O quant
    for mod in modules_to_quant:
        _add_weight_quant(mod)
        _add_io_quant(mod)

    print(f"✅ Fully fake-quantized {len(modules_to_quant)} submodules inside target module.")
    return hooks

def clear_hooks():
    global hooks
    for h in hooks:
        h.remove()
    hooks = []

# ----------------------------
# 原 evaluate 函数（稍作封装）
# ----------------------------
def evaluate_model(model, loader, dataset, config):
    ts = []
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    # 初始化 prev_bev（注意：config 中字段名可能不同，需确认）
    prev_bev = torch.randn(config.bev_h_ * config.bev_w_, 1, config._dim_).cuda()
    
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }

    for data in loader:
        # 正确提取 DataContainer 中的数据
        img = data["img"][0].data[0].cuda()  # [B, N, C, H, W]
        img_metas = data["img_metas"][0].data[0]  # list of dict, len=B

        # 因为 samples_per_gpu=1，所以 batch_size=1
        assert img.shape[0] == 1
        assert len(img_metas) == 1

        use_prev_bev = torch.tensor([1.0]).cuda()
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = torch.tensor([0.0]).cuda()
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        can_bus = torch.from_numpy(img_metas[0]["can_bus"]).cuda().float()
        lidar2img = (
            torch.from_numpy(np.stack(img_metas[0]["lidar2img"], axis=0))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        with torch.no_grad():
            torch.cuda.synchronize()
            t1 = time.time()
            # 调用 forward_trt（你已绑定到 model.forward）
            bev_embed, outputs_classes, outputs_coords = model(
                img, prev_bev, use_prev_bev, can_bus, lidar2img
            )
            torch.cuda.synchronize()
            t2 = time.time()

        # 后处理
        result = model.module.post_process(outputs_classes, outputs_coords, img_metas)
        results.extend(result)

        # 更新状态
        prev_bev = bev_embed
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle
        ts.append(t2 - t1)

        prog_bar.update()

    metric = dataset.evaluate(results)
    return metric.get("pts_bbox_NuScenes/mAP", 0.0)
# ----------------------------
# 主函数
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Quantization Sensitivity Analysis for BEVFormer")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    return parser.parse_args()

def main():
    args = parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint

    config = Config.fromfile(config_file)
    if hasattr(config, "plugin"):
        import importlib
        sys.path.append(".")
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    # 构建原始模型
    model = build_model(config.model, test_cfg=config.get("test_cfg", None))
    load_checkpoint(model, checkpoint_file, map_location="cpu")

    dataset = build_dataset(cfg=config.data.val)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=6,
        shuffle=False,
        dist=False
    )

    if "CLASSES" in model.state_dict().get("meta", {}):
        model.CLASSES = model.state_dict()["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    if hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    model.forward = model.forward_trt
    model.eval()
    model = MMDataParallel(model.cuda())

    # 🔍 定义要分析的模块（根据 BEVFormer-tiny 结构）
    modules_to_analyze = [
        "img_backbone",
        "img_neck",
        "pts_bbox_head.transformer.encoder",
        "pts_bbox_head.transformer.decoder",
        "pts_bbox_head.bev_embedding",
        "pts_bbox_head.query_embedding",
        "pts_bbox_head.cls_branches",
        "pts_bbox_head.reg_branches",
    ]

    # 1️⃣ 测 baseline
    print("\n🔍 Evaluating baseline (FP32)...")
    baseline_map = evaluate_model(model, loader, dataset, config)
    print(f"\n✅ Baseline mAP: {baseline_map:.4f}\n")

    sensitivity = {}

    # 2️⃣ 逐个模块 fake quant + eval
    for mod_name in modules_to_analyze:
        print(f"\n🧪 Analyzing module: {mod_name}")
        try:
            # 清除旧 hook
            clear_hooks()
            # 添加新 hook
            # add_output_fake_quant(model, mod_name)
            target_module = None
            for name, mod in model.module.named_modules():
                if name == mod_name:
                    target_module = mod
                    break
            if target_module is None:
                raise ValueError(f"Module {mod_name} not found!")

            # 递归量化整个模块内部
            local_hooks = quantize_module_recursively(target_module)
            hooks.extend(local_hooks)  # 用于后续 clear
            # 评估
            map_q = evaluate_model(model, loader, dataset, config)
            drop = baseline_map - map_q
            sensitivity[mod_name] = drop
            print(f"   → Quantized {mod_name} | mAP: {map_q:.4f} | ΔmAP: {drop:.4f}")
        except Exception as e:
            print(f"   ❌ Failed on {mod_name}: {e}")
            sensitivity[mod_name] = float('inf')
        finally:
            clear_hooks()

    # 3️⃣ 输出结果
    print("\n" + "="*60)
    print("📊 QUANTIZATION SENSITIVITY ANALYSIS RESULTS")
    print("="*60)
    print(f"Baseline mAP: {baseline_map:.4f}")
    print("-"*60)
    sorted_res = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    for name, drop in sorted_res:
        status = "🔴 HIGH" if drop > 0.02 else "🟡 MED" if drop > 0.005 else "🟢 LOW"
        print(f"{status} | ΔmAP={drop:.4f} | {name}")
    print("="*60)

if __name__ == "__main__":
    main()