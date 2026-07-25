/**
 * @file   select.cuh
 * @author Jiaqi Gu, Yibo Lin
 * @date   Mar 2019
 */
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_SELECT_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_SELECT_CUH

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

#define THREADS 256 

template <typename FlagType>
__global__ void collect_kernel(
    const FlagType* d_flags, int* d_sums, int* d_results, const int length)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < length && d_flags[tid] == 1)
    {
        d_results[d_sums[tid]] = tid;
    }
}

template <typename FlagType>
struct FlagToInt
{
    __host__ __device__ __forceinline__ int operator()(FlagType flag) const
    {
        return static_cast<int>(flag);
    }
};

template <typename T, typename V>
__global__ void select_kernel_add(const T* a, const V* b, int* c) 
{
    if (blockIdx.x == 0 && threadIdx.x == 0) 
    {
        *c = (int)(*a) + (int)(*b);
    }
}

template <typename FlagType>
void select(
    const FlagType* d_flags, int* d_results, const int length, int* scratch,
    int *num_collected)
{
    size_t   temp_storage_bytes = 0;
    void     *d_temp_storage = NULL; //need this NULL pointer to get temp_storage_bytes
    int* prefix_sum = scratch;
    using FlagIterator = cub::TransformInputIterator<
        int, FlagToInt<FlagType>, const FlagType*>;
    FlagIterator flags(d_flags, FlagToInt<FlagType>());

    checkCUDA(cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, flags, prefix_sum, length));

    // Run exclusive prefix sum
    checkCUDA(cub::DeviceScan::ExclusiveSum(
        (void*)d_results, temp_storage_bytes, flags, prefix_sum, length));
    //cudaDeviceSynchronize();

    select_kernel_add<<<1, 1>>>(prefix_sum + (length-1), d_flags + (length - 1), num_collected); 

    collect_kernel<<<(length + THREADS - 1) / THREADS, THREADS>>>(
        d_flags, prefix_sum, d_results, length);
    cudaDeviceSynchronize();
}

DREAMPLACE_END_NAMESPACE

#endif
