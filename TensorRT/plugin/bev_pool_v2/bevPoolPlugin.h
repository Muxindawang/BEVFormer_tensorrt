//
// Created by Derry Lin on 2023/05/11.
//

#ifndef TENSORRT_OPS_BEVPOOLPLUGIN_H
#define TENSORRT_OPS_BEVPOOLPLUGIN_H

#include "helper.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace trt_plugin {

class BEVPoolPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
  BEVPoolPlugin(int outWidth, int outHeight, bool use_h2);
  BEVPoolPlugin(const void *serialData, size_t serialLength, bool use_h2);
  ~BEVPoolPlugin() override;

  // IPluginV2 接口方法
  int32_t getNbOutputs() const noexcept override;

  nvinfer1::DimsExprs
  getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const *inputs,
                      int32_t nbInputs,
                      nvinfer1::IExprBuilder &exprBuilder) noexcept override;

  int32_t initialize() noexcept override;

  void terminate() noexcept override;

  size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const *inputs,
                          int32_t nbInputs,
                          nvinfer1::PluginTensorDesc const *outputs,
                          int32_t nbOutputs) const noexcept override;

  int32_t enqueue(nvinfer1::PluginTensorDesc const *inputDesc,
                  nvinfer1::PluginTensorDesc const *outputDesc,
                  void const *const *inputs, void *const *outputs,
                  void *workspace, cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void *buffer) const noexcept override;

  // 【关键】告诉 TRT 哪些 (DataType, Format) 组合是支持的
  bool supportsFormatCombination(int32_t pos,
                                 nvinfer1::PluginTensorDesc const *inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override;

  // 其他标准接口
  char const *getPluginType() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  void destroy() noexcept override;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  void setPluginNamespace(char const *pluginNamespace) noexcept override;

  char const *getPluginNamespace() const noexcept override;

  nvinfer1::DataType
  getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes,
                    int32_t nbInputs) const noexcept override;

  void attachToContext(cudnnContext *cudnn, cublasContext *cublas,
                       nvinfer1::IGpuAllocator *allocator) noexcept override;

  void detachFromContext() noexcept override;

  // 【重要】配置插件，根据输入/输出张量动态调整内部状态
  void configurePlugin(nvinfer1::DynamicPluginTensorDesc const *in,
                       int32_t nbInputs,
                       nvinfer1::DynamicPluginTensorDesc const *out,
                       int32_t nbOutputs) noexcept override;

private:
  // 是否使用 half2 向量化加速
  bool use_h2;
  bool use_int8{true};
  std::string mPluginNamespace;
  std::string mNamespace;

  //// 输出 BEV 特征图的宽度
  int mOutWidth;
  int mOutHeight;
};

// 插件创建器1：不使用 half2
class BEVPoolPluginCreator : public trt_plugin::BaseCreator {
public:
  BEVPoolPluginCreator();
  ~BEVPoolPluginCreator() override = default;

  char const *getPluginName() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(char const *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  deserializePlugin(char const *name, void const *serialData,
                    size_t serialLength) noexcept override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

class BEVPoolPluginCreator2 : public trt_plugin::BaseCreator {
public:
  BEVPoolPluginCreator2();
  ~BEVPoolPluginCreator2() override = default;

  char const *getPluginName() const noexcept override;

  char const *getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  createPlugin(char const *name,
               const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2DynamicExt *
  deserializePlugin(char const *name, void const *serialData,
                    size_t serialLength) noexcept override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace trt_plugin

#endif // TENSORRT_OPS_BEVPOOLPLUGIN_H
