/**
 * @file random.cuh
 * @author daaguirre
 * @brief
 * @date 2022-08-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __CUDA_TOOLS_RANDOM_CUH_
#define __CUDA_TOOLS_RANDOM_CUH_

#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <cmath>
#include <cub/cub.cuh>
#include <vector>

#include "error.h"
#include "random.h"

namespace ct
{

namespace cg = cooperative_groups;

/**
 * @brief uniform selection from a dataset
 *
 * @tparam T
 * @tparam BS
 * @param data data to select from
 * @param n number of elements in data
 * @param n_sel number of elements to select
 * @param selected_data number of elements to select
 * @param seed random seed
 * @return __global__
 */
template <typename T, uint N_PER_THREAD, uint BS>
__global__ void random_uniform_selection_gpu(
    T *data,
    const uint n,
    T *selected_data,
    const uint n_sel,
    const uint seed)
{
    const auto block{cg::this_thread_block()};
    typedef cub::CacheModifiedInputIterator<cub::LOAD_LDG, T> InputItr;
    using BlockLoad = cub::BlockLoad<T, BS, N_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<T, BS, N_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    __shared__ union TempStorage
    {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    } temp_storage;

    uint block_offset = blockIdx.x * blockDim.x * N_PER_THREAD;
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid < n; tid += blockDim.x * gridDim.x)
    {
        T thread_data[N_PER_THREAD];

        curandState local_state;
        curand_init(static_cast<unsigned long long int>(seed + tid), 0, 0, &local_state);

#pragma unroll N_PER_THREAD
        for (uint i = 0; i < N_PER_THREAD; i++)
        {
            float x = curand_uniform(&local_state);
            auto idx = static_cast<uint>(n * x);
            thread_data[i] = data[idx];
        }

        BlockStore(temp_storage.store).Store(selected_data + block_offset, thread_data);
        block.sync();

        block_offset += blockDim.x * gridDim.x * N_PER_THREAD;
    }
}

template <typename T>
std::vector<T> random_uniform_selection(std::vector<T> data, const uint n_sel, const uint seed)
{
    int *d_data, *d_sel_data;
    size_t data_size = data.size() * sizeof(T);
    size_t sel_data_size = n_sel * sizeof(T);

    CUDA_RT_CALL(cudaMalloc(&d_data, data_size));
    CUDA_RT_CALL(cudaMalloc(&d_sel_data, sel_data_size));
    CUDA_RT_CALL(cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemset(d_sel_data, 0, sel_data_size));

    constexpr uint bs = 256;
    auto num_blocks = static_cast<uint>(std::ceil(n_sel / static_cast<float>(bs)));
    const uint n = static_cast<uint>(data.size());

    random_uniform_selection_gpu<T, 1, bs><<<num_blocks, bs>>>(d_data, n, d_sel_data, n_sel, seed);
    CUDA_GET_LAST_ERROR();
    CUDA_RT_CALL(cudaDeviceSynchronize());

    std::vector<T> sel_data(n_sel);
    CUDA_RT_CALL(cudaMemcpy(sel_data.data(), d_sel_data, sel_data_size, cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaFree(d_data));
    CUDA_RT_CALL(cudaFree(d_sel_data));

    return sel_data;
}

}  // namespace ct

#endif  // __CUDA_TOOLS_RANDOM_CUH_
