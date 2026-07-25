/**
 * @file   auction.cuh
 * @author Jiaqi Gu, Yibo Lin
 * @date   Jan 2019
 */

#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_AUCTION_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_AUCTION_CUH

#include <cstdlib>
#include <iostream>
#include <string>

#include <stdio.h>
#include <stdlib.h>

#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

#define BIG_NEGATIVE    -9999999
#define MAX_MINIBATCH         64

inline void init_auction(
        const int num_graphs, 
        char*& stop_flags
        )
{
    allocateCUDA(stop_flags, num_graphs, char);
}

inline void destroy_auction(
        char* stop_flags
        )
{
    destroyCUDA(stop_flags); 
}

template <typename T>
__global__ void compute_orig_cost_kernel(const T* cost_matrices, const int num_nodes, T* costs)
{
    int i = blockIdx.x; // set 
    auto cost_matrix = cost_matrices + i*num_nodes*num_nodes; 
    for (int j = threadIdx.x; j < num_nodes; j += blockDim.x)
    {
        atomicAdd(costs+i, cost_matrix[j*num_nodes+j]);
    }
}

template <typename T>
__global__ void compute_solution_cost_kernel(const T* cost_matrices, const int* solutions, const int num_nodes, T* costs)
{
    int i = blockIdx.x; // set 
    auto cost_matrix = cost_matrices + i*num_nodes*num_nodes; 
    auto solution = solutions + i*num_nodes; 
    for (int j = threadIdx.x; j < num_nodes; j += blockDim.x)
    {
        atomicAdd(costs+i, cost_matrix[j*num_nodes+solution[j]]);
    }
}

template <typename T>
__global__ void print_costs_kernel(T* a, int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", n);
        for (int i = 0; i < n; ++i)
        {
            printf("%g ", (double)a[i]);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void check_costs_kernel(const T* a, const T* b, int n, T epsilon)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            if (a[i] > b[i]+epsilon)
            {
                printf("cost error %g > %g\n", a[i], b[i]);
            }
            assert(a[i] <= b[i]+epsilon);
        }
    }
}

template <typename T>
__global__ void print_solution_kernel(const T* solutions, int num_graphs, int num_nodes)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < num_graphs; ++i)
        {
            printf("[%d]\n", i);
            for (int j = 0; j < num_nodes; ++j)
            {
                printf("%d ", solutions[i*num_nodes+j]);
            }
            printf("\n");
        }
    }
}

__global__ void print_stop_flags_kernel(char* stop_flags, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("[%dx64]\n", n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 64; ++j)
            {
                printf("%d", stop_flags[i]);
            }
            printf("\n");
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(1024)
linear_assignment_auction_kernel(const int num_nodes,
                                        const T* __restrict__ data_ptr,
                                        int* solutions_ptr,
                                        char* stop_flag_ptr,
                                        const float auction_max_eps,
                                        const float auction_min_eps,
                                        const float auction_factor,
                                        const int max_iterations)
{
    const int batch_id = blockIdx.x;
    const int node_id = threadIdx.x;
    __shared__ float auction_eps;
    __shared__ int num_iteration;
    __shared__ int num_assigned;
    extern __shared__ unsigned long long shared_storage[];
    const int price_words =
        (num_nodes * sizeof(T) + sizeof(unsigned long long) - 1) /
        sizeof(unsigned long long);
    T* s_prices = reinterpret_cast<T*>(shared_storage);
    unsigned long long* winning_bids = shared_storage + price_words;
    int* person2item =
        reinterpret_cast<int*>(winning_bids + num_nodes);
    int* item2person = person2item + num_nodes;

    if(node_id == 0){
        auction_eps = auction_max_eps;
        num_iteration = 0;
    }
    s_prices[node_id] = 0;

    const T* __restrict__ data = data_ptr + batch_id * num_nodes * num_nodes;
    int* solutions = solutions_ptr + batch_id * num_nodes;
    char* stop_flag = stop_flag_ptr + batch_id;

    __syncthreads();

    while(auction_eps >= auction_min_eps && num_iteration < max_iterations)
    {
        //clear num_assigned
        if(node_id == 0){
            num_assigned = 0;
        }

        //pre-init
        for(int i = node_id; i < num_nodes; i += blockDim.x){
            person2item[i] = -1;
            item2person[i] = -1;
        }
        __syncthreads();
    
        //start iterative solving
        while(num_assigned < num_nodes && num_iteration < max_iterations)
        {
            winning_bids[node_id] = 0;

            __syncthreads();

            //phase 2: bidding
            if(person2item[node_id] == -1){
                T top1_val = BIG_NEGATIVE; 
                T top2_val = BIG_NEGATIVE; 
                int top1_col; 
                T tmp_val;

                for (int col = 0; col < num_nodes; col++)
                {
                    tmp_val = data[node_id * num_nodes + col]; 
                    if (tmp_val < 0)
                    {
                        continue;
                    }
                    tmp_val = tmp_val - s_prices[col];
                    if (tmp_val >= top1_val)
                    {
                        top2_val = top1_val;
                        top1_col = col;
                        top1_val = tmp_val;
                    }
                    else if (tmp_val > top2_val)
                    {
                        top2_val = tmp_val;
                    }
                }
                if (top2_val == BIG_NEGATIVE)
                {
                    top2_val = top1_val;
                }
                T bid = top1_val - top2_val + auction_eps;
                // Maximize the bid and, for ties, retain the lowest bidder ID
                // to match the original ordered scan exactly.
                unsigned long long bid_key =
                    (static_cast<unsigned long long>(
                         static_cast<unsigned int>(bid))
                     << 32) |
                    (0xFFFFFFFFu - static_cast<unsigned int>(node_id));
                atomicMax(winning_bids + top1_col, bid_key);
            }

            __syncthreads();

            //phase 3 : assignment
            unsigned long long winning_bid = winning_bids[node_id];
            if(winning_bid != 0) {
                T high_bid = static_cast<T>(winning_bid >> 32);
                int high_bidder = static_cast<int>(
                    0xFFFFFFFFu -
                    static_cast<unsigned int>(winning_bid));
    
                int current_person = item2person[node_id];
                if(current_person >= 0){
                    person2item[current_person] = -1;
                } else {
                    atomicAdd(&num_assigned, 1);
                }
    
                s_prices[node_id]              += high_bid;
                person2item[high_bidder] = node_id;
                item2person[node_id]           = high_bidder;
            }
            __syncthreads();
            
            //update iteration
            if(node_id == 0){
                num_iteration++;
            }
            __syncthreads();
        }
        //scale auction_eps
        if(node_id == 0){
            auction_eps *= auction_factor;
        }
        __syncthreads();
    }
    __syncthreads();
    solutions[node_id] = person2item[node_id];
    //report whether finish solving
    if(node_id == 0){
        *stop_flag = (num_assigned == num_nodes);
    }
}

template <typename T>
void linear_assignment_auction(
                const T* cost_matrics,
                int*  solutions,
                const int num_graphs,
                const int num_nodes,
                char* stop_flags,
                const float auction_max_eps,
                const float auction_min_eps,
                const float auction_factor,
                const int max_iterations)
{
    int shared_memory_size =
        ((num_nodes * sizeof(T) + sizeof(unsigned long long) - 1) /
         sizeof(unsigned long long) + num_nodes) *
        sizeof(unsigned long long) +
        num_nodes * 2 * sizeof(int);

    //launch solver
    linear_assignment_auction_kernel<T><<<num_graphs, num_nodes, shared_memory_size>>>
                                    (
                                        num_nodes,
                                        cost_matrics,
                                        solutions,
                                        stop_flags,
                                        auction_max_eps,
                                        auction_min_eps,
                                        auction_factor,
                                        max_iterations
                                    );

}

DREAMPLACE_END_NAMESPACE

#endif
