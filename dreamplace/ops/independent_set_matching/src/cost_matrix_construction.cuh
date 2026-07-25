
#ifndef _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_CUH
#define _DREAMPLACE_INDEPENDENT_SET_MATCHING_COST_MATRIX_CONSTRUCTION_CUH

#include "utility/src/utils.cuh"
#include "independent_set_matching/src/adjust_pos.h"

DREAMPLACE_BEGIN_NAMESPACE

#define MAX_NODE_DEGREE 32
constexpr int kCostMatrixRowsPerBlock = 8;
constexpr int kPostprocessRowsPerBlock = 4;

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void print_net_boxes_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int node_id = 0; node_id < db.num_movable_nodes; ++node_id) 
        {
            if (state.selected_markers[node_id])
            {
                int node2pin_id = db.flat_node2pin_start_map[node_id];
                const int node2pin_id_end = db.flat_node2pin_start_map[node_id+1];
                for (; node2pin_id < node2pin_id_end; ++node2pin_id)
                {
                    int node_pin_id = db.flat_node2pin_map[node2pin_id];
                    int net_id = db.pin2net_map[node_pin_id];
                    auto const& box = state.net_boxes[net_id];
                    printf("node %d: net %d (%g, %g, %g, %g)\n", node_id, net_id, box.xl, box.yl, box.xh, box.yh);
                }
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void compute_cost_matrix_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    int i = blockIdx.y; // set 
    const int* __restrict__ independent_set = state.independent_sets + i*state.set_size; 
    __shared__ DreamPlace::Utility::SharedBox<typename DetailedPlaceDBType::type> net_boxes[MAX_NODE_DEGREE]; 
    __shared__ typename DetailedPlaceDBType::type pin_offset_x[MAX_NODE_DEGREE];
    __shared__ typename DetailedPlaceDBType::type pin_offset_y[MAX_NODE_DEGREE];
    __shared__ unsigned char net_enabled[MAX_NODE_DEGREE];
    int pos_id = independent_set[threadIdx.x];
    typename DetailedPlaceDBType::type target_x_orig;
    typename DetailedPlaceDBType::type target_y;
    Space<typename DetailedPlaceDBType::type> target_space;
    if (pos_id < db.num_movable_nodes)
    {
        target_x_orig = db.x[pos_id];
        target_y = db.y[pos_id];
        target_space = state.spaces[pos_id];
    }
    int thread_max_cost = 0;
    int j_end = min(
        (blockIdx.x + 1) * kCostMatrixRowsPerBlock, state.set_size);
    for (int j = blockIdx.x * kCostMatrixRowsPerBlock; j < j_end; ++j)
    {
    auto cost_matrix = state.cost_matrices + i*state.cost_matrix_size + j*state.set_size;
    int node_id = independent_set[j];
    typename DetailedPlaceDBType::type node_width =
        DREAMPLACE_CUDA_NAMESPACE::numeric_limits<
            typename DetailedPlaceDBType::type>::max();
    int node2pin_id_bgn = 0;
    int node2pin_id_end = 0;
    if (node_id < db.num_movable_nodes)
    {
        node_width = db.node_size_x[node_id];

        node2pin_id_bgn = db.flat_node2pin_start_map[node_id];
        node2pin_id_end = db.flat_node2pin_start_map[node_id+1];
        node2pin_id_end = min(
            node2pin_id_bgn+MAX_NODE_DEGREE, node2pin_id_end);

        constexpr int net_threads = 4;
        int net_lane = threadIdx.x & (net_threads - 1);
        int idx = threadIdx.x / net_threads;
        int num_node_pins = node2pin_id_end - node2pin_id_bgn;
        if (idx < num_node_pins)
        {
            int node2pin_id = node2pin_id_bgn + idx;
            int node_pin_id = db.flat_node2pin_map[node2pin_id];
            int net_id = db.pin2net_map[node_pin_id];
#ifdef DEBUG
            assert(node_pin_id >= 0 && node_pin_id < db.num_pins);
            assert(net_id >= 0 && net_id < db.num_nets);
#endif
            unsigned char enabled = db.net_mask[net_id];
            if (net_lane == 0)
            {
                pin_offset_x[idx] = db.pin_offset_x[node_pin_id];
                pin_offset_y[idx] = db.pin_offset_y[node_pin_id];
                net_enabled[idx] = enabled;
            }

            DreamPlace::Utility::SharedBox<
                typename DetailedPlaceDBType::type> box;
            box.xl = db.xh;
            box.yl = db.yh;
            box.xh = db.xl;
            box.yh = db.yl;
            if (enabled)
            {
                int net2pin_id_bgn = db.flat_net2pin_start_map[net_id];
                int net2pin_id_end = db.flat_net2pin_start_map[net_id+1];
                for (int net2pin_id = net2pin_id_bgn + net_lane;
                     net2pin_id < net2pin_id_end;
                     net2pin_id += net_threads)
                {
                    int net_pin_id = db.flat_net2pin_map[net2pin_id];
                    int other_node_id = db.pin2node_map[net_pin_id];
                    if (other_node_id != node_id)
                    {
                        typename DetailedPlaceDBType::type xxl =
                            db.x[other_node_id]+db.pin_offset_x[net_pin_id];
                        typename DetailedPlaceDBType::type yyl =
                            db.y[other_node_id]+db.pin_offset_y[net_pin_id];
                        box.xl = min(box.xl, xxl);
                        box.xh = max(box.xh, xxl);
                        box.yl = min(box.yl, yyl);
                        box.yh = max(box.yh, yyl);
                    }
                }
            }

            unsigned int active_mask = __activemask();
            for (int offset = net_threads / 2; offset > 0; offset >>= 1)
            {
                auto other_xl =
                    __shfl_down_sync(active_mask, box.xl, offset, net_threads);
                auto other_yl =
                    __shfl_down_sync(active_mask, box.yl, offset, net_threads);
                auto other_xh =
                    __shfl_down_sync(active_mask, box.xh, offset, net_threads);
                auto other_yh =
                    __shfl_down_sync(active_mask, box.yh, offset, net_threads);
                if (net_lane < offset)
                {
                    box.xl = min(box.xl, other_xl);
                    box.yl = min(box.yl, other_yl);
                    box.xh = max(box.xh, other_xh);
                    box.yh = max(box.yh, other_yh);
                }
            }
            if (net_lane == 0)
            {
                net_boxes[idx] = box;
            }
        }
    }

    __syncthreads();

    for (int k = threadIdx.x; k < state.set_size; k += blockDim.x) // pos in set 
    {
        auto& cost = cost_matrix[k]; // row major 
        if (node_id < db.num_movable_nodes && pos_id < db.num_movable_nodes)
        {
#ifdef DEBUG
            assert(db.node_size_x[node_id] == db.node_size_x[pos_id]);
#endif
            typename DetailedPlaceDBType::type target_x = target_x_orig;
            int target_hpwl = 0; 
            if (adjust_pos(target_x, node_width, target_space))
            {
                // consider FENCE region 
                if (db.num_regions && !db.inside_fence(node_id, target_x, target_y))
                {
                    cost = BIG_NEGATIVE; // as a marker for post processing 
                }
                else 
                {
                    int idx = 0; 
                    for (int node2pin_id = node2pin_id_bgn; node2pin_id < node2pin_id_end; ++node2pin_id, ++idx)
                    {
#ifdef DEBUG
                        assert(node2pin_id >= 0 && node2pin_id < db.num_pins);
#endif
                        auto const& box = net_boxes[idx];
                        if (net_enabled[idx])
                        {
                            typename DetailedPlaceDBType::type xxl =
                                target_x + pin_offset_x[idx];
                            typename DetailedPlaceDBType::type yyl =
                                target_y + pin_offset_y[idx];
                            typename DetailedPlaceDBType::type bxl = min(box.xl, xxl);
                            typename DetailedPlaceDBType::type bxh = max(box.xh, xxl);
                            typename DetailedPlaceDBType::type byl = min(box.yl, yyl);
                            typename DetailedPlaceDBType::type byh = max(box.yh, yyl);
                            target_hpwl += (bxh-bxl) + (byh-byl); 
                        }
                    }
                    //target_hpwl = target_hpwl*db.row_height + (abs(target_x-node_x) + abs(target_y-node_y));
                    // row major 
#ifdef DEBUG
                    assert(state.set_size*j + k >= 0 && state.set_size*j + k < state.cost_matrix_size);
                    assert(state.large_number > target_hpwl);
#endif
                    cost = target_hpwl; 
                }
            }
            else 
            {
                cost = BIG_NEGATIVE; // as a marker for post processing 
            }
        }
        else 
        {
            //cost = state.large_number*(j != k); 
            cost = BIG_NEGATIVE; // as a marker for post processing
        }
        thread_max_cost = max(thread_max_cost, cost);
    }
    if (j + 1 < j_end)
    {
        __syncthreads();
    }
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warp_threads = min(32, (int)blockDim.x - warp*32);
    unsigned int active_mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        int other_max =
            __shfl_down_sync(active_mask, thread_max_cost, offset);
        if (lane + offset < warp_threads)
        {
            thread_max_cost = max(thread_max_cost, other_max);
        }
    }

    __shared__ int warp_max_costs[32];
    if (lane == 0)
    {
        warp_max_costs[warp] = thread_max_cost;
    }
    __syncthreads();

    if (warp == 0)
    {
        int num_warps = (blockDim.x + 31) >> 5;
        int block_max_cost =
            lane < num_warps ? warp_max_costs[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            int other_max =
                __shfl_down_sync(active_mask, block_max_cost, offset);
            if (lane + offset < num_warps)
            {
                block_max_cost = max(block_max_cost, other_max);
            }
        }
        if (lane == 0)
        {
            atomicMax(state.max_costs + i, block_max_cost);
        }
    }
}

/// @brief change from minimization problem for maximization problem with non-negative edge weights 
template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
__global__ void postprocess_cost_matrix_kernel(DetailedPlaceDBType db, IndependentSetMatchingStateType state)
{
    int i = blockIdx.y; // set 
    const int* __restrict__ independent_set = state.independent_sets + i*state.set_size; 
    auto max_cost = state.max_costs[i];
    for (int k = threadIdx.x; k < state.set_size; k += blockDim.x) // pos in set 
    {
        int pos_id = independent_set[k]; 
        int j_end = min(
            (blockIdx.x + 1) * kPostprocessRowsPerBlock, state.set_size);
        for (int j = blockIdx.x * kPostprocessRowsPerBlock; j < j_end; ++j)
        {
            int node_id = independent_set[j];
            auto& cost = state.cost_matrices[
                i*state.cost_matrix_size + j*state.set_size + k];
            if (node_id < db.num_movable_nodes && pos_id < db.num_movable_nodes)
            {
                if (cost >= 0)
                {
                    cost = max_cost - cost;
                }
                // cost < 0 is already assigned to negative
            }
            else if (j == k)
            {
                cost = max_cost; // dummy cells or positions
            }
            // j != k is already assigned to negative
        }
    }
}

template <typename T>
__global__ void print_cost_matrix_kernel(const T* cost_matrix, int set_size)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%dx%d]\n", set_size, set_size);
        for (int r = 0; r < set_size; ++r)
        {
            for (int c = 0; c < set_size; ++c)
            {
                auto cost = cost_matrix[r*set_size+c];
                if (cost == BIG_NEGATIVE)
                {
                    printf("X ");
                }
                else 
                {
                    printf("%g ", (double)cost);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void print_max_cost_kernel(IndependentSetMatchingStateType state)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", state.num_independent_sets);
        for (int i = 0; i < state.num_independent_sets; ++i)
        {
            printf("%g ", (double)state.max_costs[i]);
        }
        printf("\n");
    }
}

template <typename IndependentSetMatchingStateType>
__global__ void check_cost_matrices_kernel(IndependentSetMatchingStateType state)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < state.num_independent_sets; ++i)
        {
            for (int j = 0; j < state.cost_matrix_size; ++j)
            {
                auto cost = state.cost_matrices[i*state.cost_matrix_size+j];
                assert(cost == DREAMPLACE_CUDA_NAMESPACE::numeric_limits<typename IndependentSetMatchingStateType::cost_type>::lowest()
                        || cost >= 0);
            }
        }
    }
}

template <typename DetailedPlaceDBType, typename IndependentSetMatchingStateType>
void cost_matrix_construction(const DetailedPlaceDBType& db, IndependentSetMatchingStateType& state)
{
    dim3 compute_grid(
        ceilDiv(state.set_size, kCostMatrixRowsPerBlock),
        state.num_independent_sets, 1);
    dim3 postprocess_grid(
        ceilDiv(state.set_size, kPostprocessRowsPerBlock),
        state.num_independent_sets, 1);
    checkCUDA(cudaMemset(
        state.max_costs, 0,
        state.num_independent_sets *
            sizeof(typename IndependentSetMatchingStateType::cost_type)));
    compute_cost_matrix_kernel<<<compute_grid, state.set_size>>>(db, state);
#ifdef DEBUG
    //print_cost_matrix_kernel<<<1, 1>>>(state.cost_matrices + state.cost_matrix_size*3, state.set_size);
#endif

    constexpr int postprocess_threads = 64;
    postprocess_cost_matrix_kernel<<<postprocess_grid, postprocess_threads>>>(
        db, state);
#ifdef DEBUG
    //print_max_cost_kernel<<<1, 1>>>(state);

    //print_cost_matrix_kernel<<<1, 1>>>(state.cost_matrices + state.cost_matrix_size*3, state.set_size);
    //check_cost_matrices_kernel<<<1, 1>>>(state);
#endif
}

DREAMPLACE_END_NAMESPACE

#endif
