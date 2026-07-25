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

template <typename T>
__global__ void nesterovUpdate(
    const T* __restrict__ v_k,
    const T* __restrict__ g_k,
    const T* __restrict__ u_k,
    const T* __restrict__ alpha_k,
    T coefficient,
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
    u_kp1[i] = next_u;
    v_kp1[i] = next_v;
  }
}

template <typename T>
void nesterovUpdateCudaLauncher(
    const T* v_k, const T* g_k, const T* u_k, const T* alpha_k,
    T coefficient, T* u_kp1, T* v_kp1, int numel) {
  constexpr int thread_count = 256;
  nesterovUpdate<<<ceilDiv(numel, thread_count), thread_count, 0,
                   DREAMPLACE_STREAM>>>(
      v_k, g_k, u_k, alpha_k, coefficient, u_kp1, v_kp1, numel);
}

#define REGISTER_KERNEL_LAUNCHER(T)                                      \
  template void nesterovUpdateCudaLauncher<T>(                           \
      const T* v_k, const T* g_k, const T* u_k, const T* alpha_k,         \
      T coefficient, T* u_kp1, T* v_kp1, int numel);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
