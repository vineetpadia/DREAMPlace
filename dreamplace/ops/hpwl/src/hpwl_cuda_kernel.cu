#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void computeHPWL(
        const T* x,
        const T* y,
        const int* flat_netpin,
        const int* netpin_start,
        const T* net_weights,
        const unsigned char* net_mask,
        int num_nets,
        T* partial_hpwl
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nets)
    {
        T max_x = -FLT_MAX;
        T min_x = FLT_MAX;
        T max_y = -FLT_MAX;
        T min_y = FLT_MAX;

        if (net_mask[i])
        {
            for (int j = netpin_start[i]; j < netpin_start[i+1]; ++j)
            {
                int pin_id = flat_netpin[j];
                min_x = min(min_x, x[pin_id]);
                max_x = max(max_x, x[pin_id]);
                min_y = min(min_y, y[pin_id]);
                max_y = max(max_y, y[pin_id]);
            }
            T hpwl_x = max_x - min_x;
            T hpwl_y = max_y - min_y;
            if (net_weights)
            {
                T weight = net_weights[i];
                hpwl_x *= weight;
                hpwl_y *= weight;
            }
            partial_hpwl[i] = hpwl_x;
            partial_hpwl[num_nets + i] = hpwl_y;
        }
        else
        {
            T hpwl_x = 0;
            T hpwl_y = 0;
            if (net_weights)
            {
                T weight = net_weights[i];
                hpwl_x *= weight;
                hpwl_y *= weight;
            }
            partial_hpwl[i] = hpwl_x;
            partial_hpwl[num_nets + i] = hpwl_y;
        }
    }
}

template <typename T>
int computeHPWLCudaLauncher(
        const T* x, const T* y,
        const int* flat_netpin,
        const int* netpin_start,
        const T* net_weights,
        const unsigned char* net_mask,
        int num_nets,
        T* partial_hpwl
        )
{
    const int thread_count = 512;
    const int block_count_nets = (num_nets + thread_count - 1) / thread_count;

    computeHPWL<<<block_count_nets, thread_count, 0, DREAMPLACE_STREAM>>>(
            x,
            y,
            flat_netpin,
            netpin_start,
            net_weights,
            net_mask,
            num_nets,
            partial_hpwl
            );

    //printArray(partial_hpwl, num_nets, "partial_hpwl");

    // I move out the summation to use ATen
    // significant speedup is observed
    //sumArray<<<1, 1>>>(partial_hpwl, num_nets, hpwl);

    return 0;
}

// manually instantiate the template function
#define REGISTER_KERNEL_LAUNCHER(type) \
    template int computeHPWLCudaLauncher<type>(\
        const type* x, const type* y, \
        const int* flat_netpin, \
        const int* netpin_start, \
        const type* net_weights, \
        const unsigned char* net_mask, \
        int num_nets, \
        type* partial_hpwl \
        ); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
