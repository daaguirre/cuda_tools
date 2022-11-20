/**
 * @file timer.cuh
 * @author daaguirre
 * @brief
 * @date 2022-08-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __CUDA_TOOLS_TIMER_CUH_
#define __CUDA_TOOLS_TIMER_CUH_

#include <chrono>

namespace ct
{

template <typename TRes = std::chrono::milliseconds, typename F, typename...Args>
auto timer(F&& f, Args&&... args)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = std::forward<F>(f)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<TRes>(end - start).count();
    std::cout << "Elapsed time: " << elapsed_time << "\n";
    return result;
}

}  // namespace ct

#endif  // __CUDA_TOOLS_TIMER_CUH_
