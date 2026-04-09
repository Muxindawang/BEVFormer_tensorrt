import torch
import mmcv
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor


def init_quant_desc(calibrator, per_channel_quantization=False):
    assert calibrator in ["max", "histogram"]
    input_kw = {"calib_method": calibrator}
    weight_kw = {"calib_method": calibrator}
    if calibrator == "max":
        if not per_channel_quantization:
            input_kw["axis"] = None         # per-tensor 量化：axis=None 表示整个张量共享一个 scale
            weight_kw["axis"] = None
    # 创建量化描述符对象
    quant_desc_input = QuantDescriptor(**input_kw)
    quant_desc_weight = QuantDescriptor(**weight_kw)

    # 为所有量化层设置默认输入/权重量化描述符    # 设置输入激活的默认量化方式
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)            # 转置卷积
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    # 设置权重的默认量化方式
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)


def calibrator_qdq(
    model,
    calibrator,
    loader,
    per_channel_quantization=False,
    data_length=500,
    samples_per_gpu=16,
    **kwargs
):
    init_quant_desc(
        calibrator=calibrator, per_channel_quantization=per_channel_quantization
    )

    batches = (data_length + samples_per_gpu - 1) // samples_per_gpu
    print("batches: ", batches)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:              # 判断当前这个量化器（TensorQuantizer）是否配置了“校准器”（calibrator）
                    module.disable_quant()   # 关闭量化（前向用 FP32）
                    module.enable_calib()    # 启用校准（收集统计信息
                else:
                    module.disable()         # 非校准量化器直接禁用

            # TensorQuantizer 是 pytorch_quantization 中负责 Q/DQ 的核心类。
            # 每个 QuantConv2d / QuantLinear 内部都有两个 TensorQuantizer：
            # _input_quantizer（处理激活）
            # _weight_quantizer（处理权重）
            # 权重通常不需要校准（因为权重固定，可直接计算 amax），所以很多情况下 _weight_quantizer._calibrator = None。
            # 只有带 calibrator 的 quantizer 才需要“enable_calib”，否则直接 disable（比如某些被手动关闭的层）。
        print("Calibrating...")
        prog_bar = mmcv.ProgressBar(batches * samples_per_gpu)
        for i, data in enumerate(loader):
            if i >= batches:
                break
            model(**data, **kwargs)           # 👈 关键：触发前向，calibrator 自动记录
            for _ in range(samples_per_gpu):
                prog_bar.update()


        # model(**data) 会走完整的前向流程；
        # 每当遇到一个启用了 enable_calib() 的 TensorQuantizer，它会：
        # 把输入张量传给内部的 calibrator（如 HistogramCalibrator）；
        # 累积统计信息（如直方图、max 值等）；
        # 不需要 loss / backward，所以用 torch.no_grad()。

        # 退出校准模式，启用量化
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

        # 从校准器加载 amax（绝对值最大值）
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(method="percentile", percentile=99.99)
        # load_calib_amax() 会根据 calibrator 的统计结果，计算并设置 module._amax；
        # _amax 决定了量化 scale：
        # python
        # 编辑
        # scale = _amax / 127   # for INT8 symmetric
        model.cuda()

    return model
