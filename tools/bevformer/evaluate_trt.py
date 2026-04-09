# 导入CUDA自动初始化模块，用于自动管理CUDA上下文
import pycuda.autoinit
# 导入TensorRT库，用于加载和运行优化后的模型
import tensorrt as trt
# 导入PyCUDA的驱动模块，用于手动管理CUDA内存和流
import pycuda.driver as cuda
# 导入命令行参数解析模块，用于接收外部输入参数
import argparse
# 导入PyTorch库，用于张量操作和后处理
import torch
# 导入MMCV库，用于配置读取、进度条和数据处理
import mmcv
# 导入copy模块，用于深拷贝数据（如车辆位姿信息）
import copy
# 导入NumPy库，用于数值计算和数组操作
import numpy as np
# 从MMCV导入Config类，用于读取和解析配置文件（.py格式）
from mmcv import Config
# 从MMDeploy导入加载TensorRT插件的函数，处理模型中的自定义算子
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

import sys

# 将当前目录添加到Python搜索路径，确保能导入本地自定义模块
sys.path.append(".")
# 从det2trt的工具模块导入TensorRT相关工具函数
from det2trt.utils.tensorrt import (
    get_logger,          # 创建TensorRT日志器，控制日志输出级别
    create_engine_context,  # 加载TensorRT引擎并创建执行上下文
    allocate_buffers,    # 为输入输出分配CPU和GPU缓冲区
    do_inference,        # 执行TensorRT推理，包含数据传输和计算
)
# 从第三方BEV-MMDetection3D库导入模型构建函数
from third_party.bev_mmdet3d.models.builder import build_model
# 从第三方BEV-MMDetection3D库导入数据集和数据加载器构建函数
from third_party.bev_mmdet3d.datasets.builder import build_dataloader, build_dataset


def parse_args():
    # 创建命令行参数解析器，描述为"MMDet测试（并评估）模型"
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    # 添加第一个必选参数：测试配置文件路径（如bevformer_trt.py）
    parser.add_argument("config", help="test config file path")
    # 添加第二个必选参数：TensorRT引擎文件路径（如bevformer.trt）
    parser.add_argument("trt_model", help="checkpoint file")
    # 解析参数并返回
    args = parser.parse_args()
    return args


def main():
    # 解析命令行参数，获取配置文件和TensorRT引擎路径
    args = parse_args()
    # 加载MMDeploy的TensorRT自定义插件，确保模型中自定义算子能被识别
    load_tensorrt_plugin()

    # 提取命令行参数中的TensorRT引擎路径和配置文件路径
    trt_model = args.trt_model
    config_file = args.config
    # 创建TensorRT日志器，日志级别设为INTERNAL_ERROR（仅显示严重错误）
    TRT_LOGGER = get_logger(trt.Logger.INTERNAL_ERROR)

    # 加载TensorRT引擎文件，并创建对应的执行上下文（context用于实际推理）
    engine, context = create_engine_context(trt_model, TRT_LOGGER)

    # 创建CUDA流（stream），用于异步执行数据传输和推理，提升效率
    stream = cuda.Stream()

    # 从配置文件（.py）加载模型、数据集、推理参数等配置信息
    config = Config.fromfile(config_file)
    # 若配置中定义了"plugin"字段（自定义插件），动态导入插件模块
    # if hasattr(config, "plugin"):
    #     import importlib
    #     import sys
      
    #     # 将当前目录添加到搜索路径，确保插件模块能被找到
    #     sys.path.append(".")        # 若plugin是列表，循环导入每个插件；否则直接导入单个插件
    #     if isinstance(config.plugin, list):
    #         for plu in config.plugin:
    #             importlib.import_module(plu)
    #     else:
    #         importlib.import_module(config.plugin)

    # 从配置中读取输出张量的形状定义（如bev_embed、bboxes的维度）
    output_shapes = config.output_shapes
    # 从配置中读取输入张量的形状定义（如image、prev_bev的维度）
    input_shapes = config.input_shapes
    # 从配置中读取默认形状参数（如bev_h_、bev_w_，用于动态计算维度）
    default_shapes = config.default_shapes

    # 将default_shapes中的键值对注册为局部变量（如"bev_h_" → 变量bev_h_）
    for key in default_shapes:
        # 检查变量名是否已存在，避免冲突
        if key in locals():
            raise RuntimeError(f"Variable {key} has been defined.")
        # 将配置值赋值给局部变量
        locals()[key] = default_shapes[key]

    # 根据配置构建验证集（如NuScenes数据集）
    dataset = build_dataset(cfg=config.data.val)
    # 构建数据加载器：单卡、每批1个样本、6个工作线程、不打乱数据、非分布式
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=6, shuffle=False, dist=False
    )

    # 构建PyTorch模型（仅用于后处理，不参与前向推理），加载测试配置
    pth_model = build_model(config.model, test_cfg=config.get("test_cfg"))

    # 初始化存储推理耗时的列表
    ts = []
    # 初始化存储检测结果的列表（用于后续评估）
    bbox_results = []
    # 创建进度条，总长度为数据集样本数
    prog_bar = mmcv.ProgressBar(len(dataset))
    # 初始化前一帧的BEV特征（随机初始化，BEVFormer时序依赖需前帧特征）
    prev_bev = np.random.randn(config.bev_h_ * config.bev_w_, 1, config._dim_)
    # 初始化前一帧的场景信息（场景token、车辆位置、角度）
    prev_frame_info = {
        "scene_token": None,  # 场景标识，用于区分不同场景（场景切换时重置时序）
        "prev_pos": 0,        # 前一帧车辆位置（can_bus[:3]）
        "prev_angle": 0,      # 前一帧车辆角度（can_bus[-1]）
    }
    # 遍历数据加载器中的每个样本
    for data in loader:
        # 提取图像数据：从data字典中获取img，转为numpy数组（shape: [batch, cameras, c, h, w]）
        img = data["img"][0].data[0].numpy()
        # 提取图像元信息：包含相机参数、场景token、can_bus等
        img_metas = data["img_metas"][0].data[0]

        # 初始化是否使用前一帧BEV特征的标记（1.0表示使用，0.0表示不使用）
        use_prev_bev = np.array([1.0])
        # 若当前帧的场景token与前一帧不同（场景切换），重置时序，不使用前帧BEV
        if img_metas[0]["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = np.array([0.0])
        # 更新当前场景token到前帧信息中
        prev_frame_info["scene_token"] = img_metas[0]["scene_token"]
        # 深拷贝当前帧的车辆位置（can_bus[:3]）和角度（can_bus[-1]），用于后续更新前帧信息
        tmp_pos = copy.deepcopy(img_metas[0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0]["can_bus"][-1])
        # 若使用前帧BEV，计算当前帧与前帧的相对位姿（位置和角度差）
        if use_prev_bev[0] == 1:
            img_metas[0]["can_bus"][:3] -= prev_frame_info["prev_pos"]
            img_metas[0]["can_bus"][-1] -= prev_frame_info["prev_angle"]
        # 若不使用前帧BEV，重置can_bus中的位置和角度为0（绝对位姿归零）
        else:
            img_metas[0]["can_bus"][-1] = 0
            img_metas[0]["can_bus"][:3] = 0
        # 提取处理后的can_bus数据（车辆运动信息）
        can_bus = img_metas[0]["can_bus"]
        # 提取激光雷达到相机的投影矩阵（lidar2img），堆叠为数组（shape: [num_cameras, 4, 4]）
        lidar2img = np.stack(img_metas[0]["lidar2img"], axis=0)
        # 解析图像张量的维度：batch_size、相机数量、通道数、图像高、图像宽
        batch_size, cameras, _, img_h, img_w = img.shape

        # 处理输出形状：将配置中字符串形式的维度（如"batch_size"）替换为实际数值
        output_shapes_ = {}
        for key in output_shapes.keys():
            # 复制原始形状列表
            shape = output_shapes[key][:]
            # 遍历每个维度，若为字符串则通过eval转为变量值
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            # 存储处理后的实际输出形状
            output_shapes_[key] = shape

        # 处理输入形状：逻辑同输出形状，替换字符串维度为实际数值
        input_shapes_ = {}
        for key in input_shapes.keys():
            shape = input_shapes[key][:]
            for shape_i in range(len(shape)):
                if isinstance(shape[shape_i], str):
                    shape[shape_i] = eval(shape[shape_i])
            input_shapes_[key] = shape

        # 为输入和输出分配CPU和GPU缓冲区，并绑定到引擎的绑定点（bindings）
        inputs, outputs, bindings = allocate_buffers(
            engine, context, input_shapes=input_shapes_, output_shapes=output_shapes_
        )

        # 遍历所有输入缓冲区，将数据填充到CPU端的host内存中
        for inp in inputs:
            # 若输入名为"image"，将图像数据展平为一维数组，转为float32类型
            if inp.name == "image":
                inp.host = img.reshape(-1).astype(np.float32)
            # 若输入名为"prev_bev"，将前帧BEV特征展平为一维数组
            elif inp.name == "prev_bev":
                inp.host = prev_bev.reshape(-1).astype(np.float32)
            # 若输入名为"use_prev_bev"，将是否使用前帧BEV的标记展平
            elif inp.name == "use_prev_bev":
                inp.host = use_prev_bev.reshape(-1).astype(np.float32)
            # 若输入名为"can_bus"，将车辆运动信息展平为一维数组
            elif inp.name == "can_bus":
                inp.host = can_bus.reshape(-1).astype(np.float32)
            # 若输入名为"lidar2img"，将投影矩阵展平为一维数组
            elif inp.name == "lidar2img":
                inp.host = lidar2img.reshape(-1).astype(np.float32)
            # 若输入名未定义，抛出异常
            else:
                raise RuntimeError(f"Cannot find input name {inp.name}.")

        # 执行TensorRT推理：自动完成CPU→GPU数据传输、推理计算、GPU→CPU结果传输，返回输出和耗时
        trt_outputs, t = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        # 将推理输出的一维数组重塑为配置中定义的形状（如bev_embed: [H*W, 1, dim]）
        trt_outputs = {
            out.name: out.host.reshape(*output_shapes_[out.name]) for out in trt_outputs
        }

        # 从输出中提取当前帧的BEV特征（bev_embed），作为下一帧的prev_bev
        prev_bev = trt_outputs.pop("bev_embed")
        # 更新前帧信息中的车辆位置和角度为当前帧的原始值
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle

        # 将NumPy格式的推理输出转为PyTorch张量，适配后续PyTorch模型的后处理接口
        trt_outputs = {k: torch.from_numpy(v) for k, v in trt_outputs.items()}

        # 调用PyTorch模型的后处理方法，解析检测结果（如bbox、得分），并添加到结果列表
        bbox_results.extend(pth_model.post_process(**trt_outputs, img_metas=img_metas))
        # 记录当前帧的推理耗时
        ts.append(t)

        # 更新进度条（每批1个样本，循环1次）
        for _ in range(len(img)):
            prog_bar.update()

    # 使用数据集的evaluate方法评估检测结果，计算精度指标（如NDS、mAP）
    metric = dataset.evaluate(bbox_results)

    # 打印性能总结信息
    print("*" * 50 + " SUMMARY " + "*" * 50)
    # 遍历评估指标，打印NDS和mAP（保留3位小数）
    for key in metric.keys():
        if key == "pts_bbox_NuScenes/NDS":
            print(f"NDS: {round(metric[key], 3)}")
        elif key == "pts_bbox_NuScenes/mAP":
            print(f"mAP: {round(metric[key], 3)}")

    # 计算平均推理延迟：排除第一帧和最后一帧（避免初始化/收尾干扰），转为毫秒（ms）
    latency = round(sum(ts[1:-1]) / len(ts[1:-1]) * 1000, 2)
    # 打印平均延迟和帧率（FPS = 1000ms / 延迟）
    print(f"Latency: {latency}ms")
    print(f"FPS: {1000 / latency}")


if __name__ == "__main__":
    # 脚本入口，执行main函数
    main()
