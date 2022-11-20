/**
 * @file warp.cuh
 * @author daaguirre
 * @brief
 * @date 2022-08-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __CUDA_TOOLS_WARP_CUH_
#define __CUDA_TOOLS_WARP_CUH_

namespace ct
{

/**
 * @brief implements warp aggregation to increment a counter 
 * 
 * @tparam T counter type
 * @param ctr pinter to counter variable
 * @return the value of ctr before incrementing (same value as atomicAdd) 
 */
template <typename T>
__device__ T atomic_agg_inc(T *ctr)
{
    // get active threads in warp
    auto g = cg::coalesced_threads();
    int warp_res;
    if (g.thread_rank() == 0) warp_res = atomicAdd(ctr, g.size());
    // use shuffle to broadcast warp_res to the active lanes in the warp
    return g.shfl(warp_res, 0) + g.thread_rank();
}



}  // namespace ct

#endif  // __CUDA_TOOLS_WARP_CUH_
