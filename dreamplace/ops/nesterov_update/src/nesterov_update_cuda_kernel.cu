#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__device__ __forceinline__ T rounded_mul(T lhs, T rhs);

template <>
__device__ __forceinline__ float rounded_mul(float lhs, float rhs) {
  return __fmul_rn(lhs, rhs);
}

template <>
__device__ __forceinline__ double rounded_mul(double lhs, double rhs) {
  return __dmul_rn(lhs, rhs);
}

template <typename T>
__device__ __forceinline__ T rounded_sub(T lhs, T rhs);

template <>
__device__ __forceinline__ float rounded_sub(float lhs, float rhs) {
  return __fsub_rn(lhs, rhs);
}

template <>
__device__ __forceinline__ double rounded_sub(double lhs, double rhs) {
  return __dsub_rn(lhs, rhs);
}

template <typename T>
__device__ __forceinline__ T rounded_add(T lhs, T rhs);

template <>
__device__ __forceinline__ float rounded_add(float lhs, float rhs) {
  return __fadd_rn(lhs, rhs);
}

template <>
__device__ __forceinline__ double rounded_add(double lhs, double rhs) {
  return __dadd_rn(lhs, rhs);
}

template <typename T, bool ApplyBoundary>
__global__ void nesterovUpdateKernel(
    const T* __restrict__ v_k,
    const T* __restrict__ g_k,
    const T* __restrict__ u_k,
    const T* __restrict__ alpha_k,
    T coefficient,
    const T* __restrict__ node_size_x,
    const T* __restrict__ node_size_y,
    T xl,
    T yl,
    T xh,
    T yh,
    int num_movable_nodes,
    int num_filler_nodes,
    T* __restrict__ u_kp1,
    T* __restrict__ v_kp1,
    int numel) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    T next_u = rounded_mul(alpha_k[0], g_k[i]);
    next_u = rounded_sub(v_k[i], next_u);
    T extrapolation = rounded_sub(next_u, u_k[i]);
    extrapolation = rounded_mul(extrapolation, coefficient);
    T next_v = rounded_add(next_u, extrapolation);
    if (ApplyBoundary) {
      int num_nodes = numel >> 1;
      int node_id = i < num_nodes ? i : i - num_nodes;
      if (node_id < num_movable_nodes ||
          node_id >= num_nodes - num_filler_nodes) {
        if (i < num_nodes) {
          next_v = min(rounded_sub(xh, node_size_x[node_id]),
                       max(xl, next_v));
        } else {
          next_v = min(rounded_sub(yh, node_size_y[node_id]),
                       max(yl, next_v));
        }
      }
    }
    u_kp1[i] = next_u;
    v_kp1[i] = next_v;
  }
}

template <typename T>
void nesterovUpdateCudaLauncher(
    const T* v_k, const T* g_k, const T* u_k, const T* alpha_k,
    T coefficient, T* u_kp1, T* v_kp1, int numel) {
  constexpr int thread_count = 256;
  nesterovUpdateKernel<T, false>
      <<<ceilDiv(numel, thread_count), thread_count, 0, DREAMPLACE_STREAM>>>(
          v_k, g_k, u_k, alpha_k, coefficient, nullptr, nullptr, 0, 0, 0, 0,
          0, 0, u_kp1, v_kp1, numel);
}

template <typename T>
void nesterovUpdateWithBoundaryCudaLauncher(
    const T* v_k, const T* g_k, const T* u_k, const T* alpha_k,
    T coefficient, const T* node_size_x, const T* node_size_y, T xl, T yl,
    T xh, T yh, int num_movable_nodes, int num_filler_nodes, T* u_kp1,
    T* v_kp1, int numel) {
  constexpr int thread_count = 256;
  nesterovUpdateKernel<T, true>
      <<<ceilDiv(numel, thread_count), thread_count, 0, DREAMPLACE_STREAM>>>(
          v_k, g_k, u_k, alpha_k, coefficient, node_size_x, node_size_y, xl,
          yl, xh, yh, num_movable_nodes, num_filler_nodes, u_kp1, v_kp1,
          numel);
}

#define REGISTER_KERNEL_LAUNCHER(T)                                        \
  template void nesterovUpdateCudaLauncher<T>(                             \
      const T* v_k, const T* g_k, const T* u_k, const T* alpha_k,           \
      T coefficient, T* u_kp1, T* v_kp1, int numel);                       \
  template void nesterovUpdateWithBoundaryCudaLauncher<T>(                 \
      const T* v_k, const T* g_k, const T* u_k, const T* alpha_k,           \
      T coefficient, const T* node_size_x, const T* node_size_y, T xl,      \
      T yl, T xh, T yh, int num_movable_nodes, int num_filler_nodes,        \
      T* u_kp1, T* v_kp1, int numel);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
