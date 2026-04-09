import numpy as np

import torch
from torch.onnx import OperatorExportTypes
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import onnx
from onnxsim import simplify


import torch.nn as nn


class ImageBackboneNeck(nn.Module):
    def __init__(self, backbone, neck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, img):  # img: [B, N, C, H, W]
        B, N, C, H, W = img.shape
        img_flat = img.flatten(0, 1)  # [B*N, C, H, W]
        feats = self.backbone(img_flat)
        feats = self.neck(feats)
        # feats is a list of tensors, e.g., [x0, x1, x2, x3, x4]
        # ONNX only supports tensor outputs, so stack or return as tuple
        return tuple(feats)  # ONNX export supports tuple of tensors


class BEVFormerHeadWrapper(nn.Module):
    def __init__(self, bbox_head):
        super().__init__()
        self.bbox_head = bbox_head

    def forward(self, *feats):
        """
        feats: multi-scale image features from neck (list/tuple)
        We also need to simulate 'img_metas', but for ONNX we hardcode or remove dependency.
        In BEVFormer, img_metas are used for lidar2img, etc. — you must precompute or fix them.
        For simplicity, assume fixed camera params and batch=1.
        """
        # Reconstruct list from tuple
        feat_list = list(feats)

        # Simulate img_metas (⚠️ this is the tricky part!)
        # You MUST replace this with real/fixed meta if your head uses it!
        # Many BEVFormer implementations allow bypassing img_metas in deployment mode.
        # Check if your `bbox_head.forward()` has a `deploy=True` or similar flag.

        # Option A: If your bbox_head supports `forward_test` without img_metas:
        try:
            # Some versions use: outs = self.bbox_head.forward_test(feat_list, None, None)
            # But standard is:
            outs = self.bbox_head.forward(feat_list, img_metas=None, test=True)
        except Exception as e:
            raise RuntimeError(
                "Your pts_bbox_head requires img_metas or other context. "
                "You need to modify the head to support deployment without img_metas."
            ) from e

        # Assume output is dict like {'pred_logits', 'pred_boxes', ...}
        # ONNX only supports tensor outputs → flatten dict to ordered tensors
        output_tensors = []
        keys = []
        for k, v in outs.items():
            if isinstance(v, torch.Tensor):
                output_tensors.append(v)
                keys.append(k)
            elif isinstance(v, list):
                # e.g., list of tensors per decoder layer
                for i, t in enumerate(v):
                    output_tensors.append(t)
                    keys.append(f"{k}_{i}")
        self.output_keys = keys  # save for debugging
        return tuple(output_tensors)


@torch.no_grad()
def pytorch2onnx(
    config,
    checkpoint,
    output_file,
    opset_version=13,
    verbose=False,
    cuda=True,
    inputs_data=None,
):

    model = build_detector(config.model, test_cfg=config.get("test_cfg", None))
    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    if cuda:
        model.to("cuda")
    else:
        model.to("cpu")

    onnx_shapes = config.default_shapes
    input_shapes = config.input_shapes
    output_shapes = config.output_shapes
    dynamic_axes = config.dynamic_axes

    for key in onnx_shapes:
        # 避免变量名冲突（如已存在的 "model"、"checkpoint"）
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        locals()[key] = onnx_shapes[key]

    # 设置随机种子：确保输入数据可复现（每次导出的 ONNX 结构一致）
    torch.random.manual_seed(0)
    inputs = {}
    for key in input_shapes.keys():
        # 情况1：使用外部传入的自定义输入数据（如真实样本，用于调试）
        if inputs_data is not None and key in inputs_data:
            inputs[key] = inputs_data[key]
            if isinstance(inputs[key], np.ndarray):
                inputs[key] = torch.from_numpy(inputs[key])
            assert isinstance(inputs[key], torch.Tensor)
        # 情况2：使用随机数据生成输入（默认方式，无需真实样本）
        else:
            for i in range(len(input_shapes[key])):
                # 解析输入形状：将字符串形式的维度（如 "batch_size"）替换为实际值（如 1）
                if isinstance(input_shapes[key][i], str):
                    input_shapes[key][i] = eval(input_shapes[key][i])
            inputs[key] = torch.randn(*input_shapes[key])
        if cuda:
            inputs[key] = inputs[key].cuda()

    model.forward = model.forward_trt
    input_name = list(input_shapes.keys())
    output_name = list(output_shapes.keys())

    inputs = tuple(inputs.values())

    torch.onnx.export(
        model,
        inputs,
        output_file,
        input_names=input_name,
        output_names=output_name,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        verbose=verbose,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    )

    print(f"ONNX file has been saved in {output_file}")

    model = onnx.load(output_file)
    onnx.checker.check_model(model)

    simplified_model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    simplified_model = onnx.shape_inference.infer_shapes(model) 
    onns_sim_path = output_file.replace(".onnx", "_sim.onnx")
    onnx.save(simplified_model, onns_sim_path)
    print(f"Simplified ONNX model has been saved in {onns_sim_path}")
