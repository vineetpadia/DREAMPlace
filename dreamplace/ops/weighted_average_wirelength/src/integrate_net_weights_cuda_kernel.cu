/**
 * @file   integrate_net_weights_cuda_kernel.cu
 * @author Yibo Lin
 * @date   Jul 2019
 */

#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "weighted_average_wirelength/src/functional_cuda.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void integrateNetWeights(
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        const T* net_weights, 
        T* grad_x_tensor, T* grad_y_tensor, 
        int num_pins
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        int net_id = pin2net_map[i]; 
        T weight = net_weights[net_id]; 
        if (net_id >= 0 && net_mask[net_id])
        {
            grad_x_tensor[i] *= weight; 
            grad_y_tensor[i] *= weight; 
        }
    }
}

template <typename T>
void integrateNetWeightsCudaLauncher(
        const int* pin2net_map, 
        const unsigned char* net_mask, 
        const T* net_weights, 
        T* grad_x_tensor, T* grad_y_tensor, 
        int num_pins
        )
{
    integrateNetWeights<<<ceilDiv(num_pins, 256), 256, 0, DREAMPLACE_STREAM>>>(pin2net_map, net_mask, net_weights, grad_x_tensor, grad_y_tensor, num_pins); 
}

template <typename T>
__global__ void scaleIntegrateNetWeightsAndMask(
        const int* pin2net_map,
        const unsigned char* net_mask,
        const T* net_weights,
        const T* grad_pos,
        const bool* pin_mask,
        T* grad_x_tensor,
        T* grad_y_tensor,
        int num_pins,
        bool has_net_weights
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
    {
        T grad_x = grad_x_tensor[i] * grad_pos[0];
        T grad_y = grad_y_tensor[i] * grad_pos[0];
        int net_id = pin2net_map[i];
        if (has_net_weights && net_id >= 0 && net_mask[net_id])
        {
            T weight = net_weights[net_id];
            grad_x *= weight;
            grad_y *= weight;
        }
        if (pin_mask[i])
        {
            grad_x = 0;
            grad_y = 0;
        }
        grad_x_tensor[i] = grad_x;
        grad_y_tensor[i] = grad_y;
    }
}

template <typename T>
void scaleIntegrateNetWeightsAndMaskCudaLauncher(
        const int* pin2net_map,
        const unsigned char* net_mask,
        const T* net_weights,
        const T* grad_pos,
        const bool* pin_mask,
        T* grad_x_tensor,
        T* grad_y_tensor,
        int num_pins,
        bool has_net_weights
        )
{
    scaleIntegrateNetWeightsAndMask<<<ceilDiv(num_pins, 256), 256, 0,
                                      DREAMPLACE_STREAM>>>(
        pin2net_map, net_mask, net_weights, grad_pos, pin_mask, grad_x_tensor,
        grad_y_tensor, num_pins, has_net_weights);
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template void integrateNetWeightsCudaLauncher<T>(\
            const int* pin2net_map, \
            const unsigned char* net_mask, \
            const T* net_weights, \
            T* grad_x_tensor, T* grad_y_tensor, \
            int num_pins\
            ); \
    template void scaleIntegrateNetWeightsAndMaskCudaLauncher<T>(\
            const int* pin2net_map, \
            const unsigned char* net_mask, \
            const T* net_weights, \
            const T* grad_pos, \
            const bool* pin_mask, \
            T* grad_x_tensor, T* grad_y_tensor, \
            int num_pins, \
            bool has_net_weights \
            );

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
