#include <stdio.h>
#include <math.h>
#include <float.h>
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
__device__ __forceinline__ T rounded_div(T lhs, T rhs);

template <>
__device__ __forceinline__ float rounded_div(float lhs, float rhs) {
    return __fdiv_rn(lhs, rhs);
}

template <>
__device__ __forceinline__ double rounded_div(double lhs, double rhs) {
    return __ddiv_rn(lhs, rhs);
}

template <typename T>
__global__ void computePWS(
        const T* net_weights, const int* flat_nodepin,
        const int* nodepin_start, const int* pin2net_map,
        int num_physical_nodes, T* node_weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_physical_nodes) {
        // ignore large degree nets
        node_weights[i] = 0;
        for (int j = nodepin_start[i]; j < nodepin_start[i + 1]; ++j) {
            node_weights[i] += net_weights[pin2net_map[flat_nodepin[j]]];
        }
    }
}

template <typename T>
__global__ void preconditionKernel(
        T* __restrict__ grad,
        const T* __restrict__ node_weights,
        const T* __restrict__ node_areas,
        const T* __restrict__ density_weight,
        T alpha,
        int num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        T area_scale = rounded_mul(alpha, density_weight[0]);
        T denominator = rounded_add(
            node_weights[i], rounded_mul(area_scale, node_areas[i]));
        denominator = DREAMPLACE_STD_NAMESPACE::max(T(1), denominator);
        grad[i] = rounded_div(grad[i], denominator);
        grad[num_nodes + i] =
            rounded_div(grad[num_nodes + i], denominator);
    }
}

template <typename T>
int computePWSCudaLauncher(const T* net_weights, const int* flat_nodepin,
                           const int* nodepin_start, const int* pin2net_map,
                           int num_physical_nodes, T* node_weights) {
    int thread_count = 512;
    int block_count = (num_physical_nodes - 1 + thread_count) / thread_count;
    computePWS<<<block_count, thread_count, 0, DREAMPLACE_STREAM>>>(
        net_weights, flat_nodepin, nodepin_start,
        pin2net_map, num_physical_nodes, node_weights);
    return 0;
}

template <typename T>
void preconditionCudaLauncher(
        T* grad, const T* node_weights, const T* node_areas,
        const T* density_weight, T alpha, int num_nodes) {
    constexpr int thread_count = 256;
    int block_count = (num_nodes - 1 + thread_count) / thread_count;
    preconditionKernel<<<block_count, thread_count, 0, DREAMPLACE_STREAM>>>(
        grad, node_weights, node_areas, density_weight, alpha, num_nodes);
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(type)             \
    template int computePWSCudaLauncher<type>(     \
        const type* net_weights,                   \
        const int* flat_nodepin,                   \
        const int* nodepin_start,                  \
        const int* pin2net_map,                    \
        int num_physical_nodes,                    \
        type* node_weights);                       \
    template void preconditionCudaLauncher<type>(  \
        type* grad, const type* node_weights,       \
        const type* node_areas,                    \
        const type* density_weight, type alpha,     \
        int num_nodes);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
