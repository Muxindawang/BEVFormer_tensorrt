import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import time


def get_logger(level=trt.Logger.INTERNAL_ERROR):
    TRT_LOGGER = trt.Logger(level)
    return TRT_LOGGER


def create_engine_context(trt_model, trt_logger):
    with open(trt_model, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


class HostDeviceMem(object):
    def __init__(self, name, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.name = name
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return (
            "Name:\n"
            + str(self.name)
            + "\nHost:\n"
            + str(self.host)
            + "\nDevice:\n"
            + str(self.device)
            + "\n"
        )

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, context, input_shapes, output_shapes):
    # 初始化存储输入缓冲区的列表（每个元素包含输入的名称、CPU内存、GPU内存）
    inputs = []
    # 初始化存储输出缓冲区的列表（每个元素包含输出的名称、CPU内存、GPU内存）
    outputs = []
    # 初始化存储绑定地址的列表（记录每个输入/输出在GPU上的内存地址）
    bindings = []
    # 遍历引擎的所有绑定点（binding），binding_id为索引，binding为绑定点名称
    for binding_id, binding in enumerate(engine):
        # 判断当前绑定点是否为输入
        if engine.binding_is_input(binding):
            # 从输入形状配置中获取当前输入的维度（如[1, 6, 3, 256, 704]）
            dims = input_shapes[binding]
            # 在执行上下文中设置该输入的实际形状（适配动态形状推理）
            context.set_binding_shape(binding_id, dims)
        # 若为输出，从输出形状配置中获取当前输出的维度
        else:
            dims = output_shapes[binding]

        # 计算当前绑定点（输入/输出）的总元素数量（trt.volume计算张量元素总数）
        size = trt.volume(dims)
        # 获取当前绑定点的数据类型，并转为NumPy对应的数据类型（如trt.float32 → np.float32）
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # 断言：确保引擎的输入/输出仅支持FP32精度（若为其他精度需修改此处）
        assert dtype == np.float32, "Engine's inputs/outputs only support FP32."
        # 分配CPU页锁定内存（pagelocked_empty）：用于CPU与GPU之间高效数据传输（避免分页导致延迟）
        host_mem = cuda.pagelocked_empty(size, dtype)
        # 分配GPU内存：大小与CPU页锁定内存一致（host_mem.nbytes为内存字节数）
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # 将GPU内存地址转为整数，添加到绑定地址列表（用于后续推理时指定数据位置）
        bindings.append(int(device_mem))
        # 若为输入，将绑定点名称、CPU内存、GPU内存封装为HostDeviceMem对象，添加到inputs列表
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(binding, host_mem, device_mem))
        # 若为输出，同理添加到outputs列表
        else:
            outputs.append(HostDeviceMem(binding, host_mem, device_mem))
    # 返回输入缓冲区、输出缓冲区、绑定地址列表
    return inputs, outputs, bindings


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # 将所有输入数据从CPU页锁定内存异步传输到GPU内存（htod: host to device）
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # 等待输入数据传输完成（同步CUDA流，确保GPU已收到所有输入）
    stream.synchronize()
    # 记录推理开始时间
    t1 = time.time()
    # 异步执行TensorRT推理：使用绑定的GPU内存地址，通过指定CUDA流执行
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 等待推理计算完成（同步CUDA流，确保GPU已完成推理）
    stream.synchronize()
    # 记录推理结束时间（仅包含推理计算耗时，不含数据传输）
    t2 = time.time()
    # 将所有输出结果从GPU内存异步传输回CPU页锁定内存（dtoh: device to host）
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # 等待输出数据传输完成
    stream.synchronize()

    # 返回输出结果（CPU内存中的数据）和推理耗时（t2 - t1，单位：秒）
    return outputs, t2 - t1
