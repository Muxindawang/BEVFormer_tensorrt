#include "bevPoolKernel.h"
#include "cuda_helper.h"
#include "cuda_int8.h"
/*
  Function: pillar pooling
  Args:
    c                : number of channels
    n_intervals      : number of unique points
    depth            : input depth, FloatTensor[b,n,d,h,w]
    feat             : input feat, FloatTensor[b,n,h,w,c]
    ranks_depth      : input index of depth, IntTensor[n_points]
    ranks_feat       : input index of feat, IntTensor[n_points]
    ranks_bev        : output index, IntTensor[n_points]
    interval_lengths : starting position for pooled point,
  IntTensor[n_intervals] interval_starts  : how many points in each pooled
  point, IntTensor[n_intervals] out              : output features,
  FloatTensor[b, z, h, w, c]
*/
/* 算法核心思想：
   对于每一个非空的 BEV 格子 (interval)，遍历其对应的所有3D点，
   将每个3D点的特征 (feat) 乘以其深度概率 (depth)，然后累加到该 BEV 格子上。
*/
template <typename T>
__global__ void bev_pool_v2_kernel(
    int c, int n_intervals, const T *__restrict__ depth,
    const T *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, T *__restrict__ out) {
  // 每个线程负责：一个 非空 BEV 格子（interval） 的 一个特征通道（channel） 的聚合计算。
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 当前处理的 interval ID (0 ～ n_intervals-1)
  int index = idx / c;
  // 当前处理的通道索引 (0 ～ c-1)
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  // 该 BEV 格子对应的第一个点在 ranks_* 数组中的位置  获取当前 BEV 格子在 ranks_* 数组中的起始位置。
  int interval_start = interval_starts[index];
  // 该格子包含的3D 点数量
  int interval_length = interval_lengths[index];
  T psum = 0;
  const T *cur_depth;
  const T *cur_feat;
  // 遍历格子内所有3d点
  for (int i = 0; i < interval_length; i++) {
    // ranks_depth[...]：获取该点在 depth 张量中的线性偏移
    // 获取到当前点的深度值， 等价于 &depth[ranks_depth[[interval_start + i]]
    cur_depth = depth + ranks_depth[interval_start + i];
    // ranks_feat[...]：获取该点在 feat 张量中 [B,N,H,W] 部分的线性索引（记为 base）
    // 因为 feat 是 NHWC 布局，每个空间位置后紧跟 c 个通道。
    // 所以该点的第 cur_c 通道地址为：feat + base * c + cur_c。
    // NHWC布局
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    // 加权特征聚合
    psum += *cur_feat * *cur_depth;
  }

  // 获取当前格子第一个点的 ranks_bev 地址 同一格子所有点的 ranks_bev 值相同，所以只需第一个。
  const int *cur_rank = ranks_bev + interval_start;
  // *cur_rank：解引用，得到该 BEV 格子在输出张量中的线性索引（如 y * W_bev + x）。
  // out + bev_idx * c + cur_c：定位到输出张量中该格子的第 cur_c 通道地址。
  T *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

// 下面是3种不同精度的特化实现


template <>
__global__ void bev_pool_v2_kernel(
    int c, int n_intervals, const __half *__restrict__ depth,
    const __half *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, __half *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  __half psum = 0;
  const __half *cur_depth;
  const __half *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum = __hfma(*cur_feat, *cur_depth, psum);
  }

  const int *cur_rank = ranks_bev + interval_start;
  __half *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}


// 向量FP16版本
__global__ void bev_pool_v2_kernel_h2(
    int c, int n_intervals, const __half *__restrict__ depth,
    const __half2 *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, __half2 *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  __half2 psum = __float2half2_rn(0);
  const __half *cur_depth;
  const __half2 *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum = __hfma2(*cur_feat, __half2half2(*cur_depth), psum);
  }

  const int *cur_rank = ranks_bev + interval_start;
  __half2 *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = psum;
}

template <typename T> __forceinline__ __device__ T sign_05(T x) {
  if (x > 0) {
    return 0.5f;
  }
  return -0.5f;
}

template <typename T> __forceinline__ __device__ int8_t T2int8(T a) {
  a = a > 127 ? 127 : a;
  a = a < -128 ? -128 : a;
  return int8_t(a + sign_05<T>(a));
}

__global__ void bev_pool_v2_kernel_int8(
    int c, int n_intervals, const int8_t *__restrict__ depth,
    const int8_4 *__restrict__ feat, const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat, const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths, int8_4 *__restrict__ out,
    const float scale_io) {
  // 总线程数 = n_intervals * c
  // 每个线程负责：一个 BEV 格子的 一个 4 通道块 的聚合
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals)
    return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  int32_4 psum = 0;
  const int8_t *cur_depth;
  const int8_4 *cur_feat;
  for (int i = 0; i < interval_length; i++) {
    cur_depth = depth + ranks_depth[interval_start + i];
    cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
    psum.x += (cur_feat->x) * (*cur_depth);
    psum.y += (cur_feat->y) * (*cur_depth);
    psum.z += (cur_feat->z) * (*cur_depth);
    psum.w += (cur_feat->w) * (*cur_depth);
  }
  int8_4 output;
  output.x = T2int8<float>(psum.x * scale_io);
  output.y = T2int8<float>(psum.y * scale_io);
  output.z = T2int8<float>(psum.z * scale_io);
  output.w = T2int8<float>(psum.w * scale_io);

  const int *cur_rank = ranks_bev + interval_start;
  int8_4 *cur_out = out + *cur_rank * c + cur_c;
  *cur_out = output;
}

// Host 函数，负责启动 kernel
template <typename T>
void bev_pool_v2(int c, int n_intervals, int num_points, const T *depth,
                 const T *feat, const int *ranks_depth, const int *ranks_feat,
                 const int *ranks_bev, const int *interval_starts,
                 const int *interval_lengths, T *out, cudaStream_t stream) {
  cudaMemset((T *)out, 0, num_points * sizeof(T));
  // 一维block，block的Dim是THREADS_PER_BLOCK，512；一共要处理的数据是 n_intervals * c 个数据
  bev_pool_v2_kernel<<<GET_BLOCKS(n_intervals * c), THREADS_PER_BLOCK, 0,
                       stream>>>(c, n_intervals, depth, feat, ranks_depth,
                                 ranks_feat, ranks_bev, interval_starts,
                                 interval_lengths, out);
  cudaCheckError();
}

void bev_pool_v2_h2(int c, int n_intervals, int num_points, const __half *depth,
                    const __half2 *feat, const int *ranks_depth,
                    const int *ranks_feat, const int *ranks_bev,
                    const int *interval_starts, const int *interval_lengths,
                    __half2 *out, cudaStream_t stream) {
  cudaMemset((__half *)out, 0, num_points * sizeof(__half));
  bev_pool_v2_kernel_h2<<<GET_BLOCKS(n_intervals * c / 2), THREADS_PER_BLOCK, 0,
                          stream>>>(c / 2, n_intervals, depth, feat,
                                    ranks_depth, ranks_feat, ranks_bev,
                                    interval_starts, interval_lengths, out);
  cudaCheckError();
}

void bev_pool_v2_int8(int c, int n_intervals, int num_points,
                      const int8_t *depth, const float &scale_d,
                      const int8_4 *feat, const float &scale_f,
                      const int *ranks_depth, const int *ranks_feat,
                      const int *ranks_bev, const int *interval_starts,
                      const int *interval_lengths, int8_4 *out,
                      const float &scale_o, cudaStream_t stream) {
  cudaMemset((int8_t *)out, 0, num_points * sizeof(int8_t));
  bev_pool_v2_kernel_int8<<<GET_BLOCKS(n_intervals * c / 4), THREADS_PER_BLOCK,
                            0, stream>>>(
      c / 4, n_intervals, depth, feat, ranks_depth, ranks_feat, ranks_bev,
      interval_starts, interval_lengths, out, scale_d * scale_f / scale_o);
  cudaCheckError();
}

template void bev_pool_v2(int c, int n_intervals, int num_points,
                          const float *depth, const float *feat,
                          const int *ranks_depth, const int *ranks_feat,
                          const int *ranks_bev, const int *interval_starts,
                          const int *interval_lengths, float *out,
                          cudaStream_t stream);

template void bev_pool_v2(int c, int n_intervals, int num_points,
                          const __half *depth, const __half *feat,
                          const int *ranks_depth, const int *ranks_feat,
                          const int *ranks_bev, const int *interval_starts,
                          const int *interval_lengths, __half *out,
                          cudaStream_t stream);
