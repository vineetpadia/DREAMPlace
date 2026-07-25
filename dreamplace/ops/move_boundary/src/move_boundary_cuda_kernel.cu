#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeMoveBoundary(
        T* x_tensor,
        T* y_tensor,
        const T* node_size_x_tensor,
        const T* node_size_y_tensor,
        const T xl, const T yl, const T xh, const T yh,
        const int num_nodes,
        const int num_movable_nodes,
        const int num_filler_nodes
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_movable_nodes || (i >= num_nodes-num_filler_nodes && i < num_nodes))
    {
        x_tensor[i] = min(xh-node_size_x_tensor[i], max(xl, x_tensor[i]));
        y_tensor[i] = min(yh-node_size_y_tensor[i], max(yl, y_tensor[i]));
    }
}

template <typename T>
int computeMoveBoundaryMapCudaLauncher(
        T* x_tensor, T* y_tensor,
        const T* node_size_x_tensor, const T* node_size_y_tensor,
        const T xl, const T yl, const T xh, const T yh,
        const int num_nodes,
        const int num_movable_nodes,
        const int num_filler_nodes
        )
{
    int thread_count = 512;
    int block_count = (num_nodes - 1 + thread_count) / thread_count;
    computeMoveBoundary<<<block_count, thread_count, 0, DREAMPLACE_STREAM>>>(
            x_tensor,
            y_tensor,
            node_size_x_tensor,
            node_size_y_tensor,
            xl, yl, xh, yh,
            num_nodes,
            num_movable_nodes,
            num_filler_nodes
            );

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T) \
  template int computeMoveBoundaryMapCudaLauncher<T>(\
            T* x_tensor, T* y_tensor, \
            const T* node_size_x_tensor, const T* node_size_y_tensor, \
            const T xl, const T yl, const T xh, const T yh, \
            const int num_nodes, \
            const int num_movable_nodes, \
            const int num_filler_nodes \
            );
    
REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
