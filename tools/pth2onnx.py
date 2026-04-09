# 导入PyTorch量化相关库，用于INT8量化模型的转换
from pytorch_quantization import nn as quant_nn
# 导入文件操作、命令行解析、配置读取相关库
import os
import argparse
from mmcv import Config

import sys
# 将当前目录添加到Python搜索路径，确保能导入本地自定义模块（如det2trt）
sys.path.append("/MOGO_VEPFS/PCPT/pcpt/project/czl/BEVFormer_tensorrt_muxindawang/")
# 从det2trt的转换模块导入PyTorch转ONNX的核心函数
from det2trt.convert import pytorch2onnx


def parse_args():
    # 创建命令行参数解析器，描述为"将PyTorch模型转换为ONNX格式"
    parser = argparse.ArgumentParser(description="Convert PyTorch to ONNX")
    # 添加必选参数：测试配置文件路径（如模型结构、数据配置的.py文件）
    parser.add_argument("config", help="test config file path")
    # 添加必选参数：PyTorch模型权重文件路径（.pth格式）
    parser.add_argument("checkpoint", help="checkpoint file")
    # 添加可选参数：是否启用INT8量化（--int8，默认不启用）
    parser.add_argument("--int8", action="store_true")
    # 添加可选参数：ONNX的opset版本（需指定整数，如11、13，影响算子兼容性）
    parser.add_argument("--opset_version", type=int)
    # 添加可选参数：是否使用CUDA进行转换（--cuda，默认使用CPU）
    parser.add_argument("--cuda", action="store_true")
    # 添加可选参数：自定义标记（用于区分不同转换配置的ONNX文件）
    parser.add_argument("--flag", default="", type=str)
    # 解析参数并返回
    args = parser.parse_args()
    return args


def main():
    # 解析命令行参数，获取配置文件、权重路径及转换选项
    args = parse_args()

    # 提取参数中的配置文件路径和权重文件路径
    config_file = args.config
    checkpoint_file = args.checkpoint

    # 从配置文件（.py）加载模型、转换参数等配置信息
    config = Config.fromfile(config_file)
    # 若配置中定义了"plugin"字段（自定义插件），动态导入插件模块（解决自定义算子依赖）
    if hasattr(config, "plugin"):
        import importlib

        # 若plugin是列表，循环导入每个插件；否则直接导入单个插件
        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    # 生成ONNX输出文件的基础名称：从权重文件名中提取（如"model.pth"→"model"）
    output = os.path.split(args.checkpoint)[1].split(".")[0]

    # 若启用INT8量化，设置pytorch_quantization的量化模式（使用Fake Quant模拟量化）
    if args.int8:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    # 若指定了自定义标记，将标记添加到输出文件名中（如"model_flag"）
    if args.flag:
        output += f"_{args.flag}"
    # 拼接完整的ONNX输出路径：配置中指定的ONNX保存目录 + 文件名 + .onnx后缀
    output_file = os.path.join(config.ONNX_PATH, output + ".onnx")

    # 调用pytorch2onnx函数执行转换：将PyTorch模型转为ONNX格式
    pytorch2onnx(
        config,                  # 模型配置信息
        checkpoint=checkpoint_file,  # PyTorch权重文件路径
        output_file=output_file,     # 输出ONNX文件路径
        verbose=False,          # 是否打印详细转换日志（False为不打印）
        opset_version=args.opset_version,  # ONNX的opset版本（如11）
        cuda=args.cuda          # 是否使用CUDA加速转换（True为使用GPU）
    )


if __name__ == "__main__":
    # 脚本入口，执行main函数
    main()
