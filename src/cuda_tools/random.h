/**
 * @file random.h
 * @author daaguirre
 * @brief
 * @date 2022-08-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __CUDA_TOOLS_RANDOM_H_
#define __CUDA_TOOLS_RANDOM_H_

#include <vector>

#include "types.h"

namespace ct
{

template <typename T>
std::vector<T> random_uniform_selection(std::vector<T> data, const uint n_sel, const uint seed = 0);

}

#endif  // __CUDA_TOOLS_RANDOM_H_
