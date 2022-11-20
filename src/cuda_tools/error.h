/**
 * @file error.h
 * @author daaguirre
 * @brief
 * @date 2022-08-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __CUDA_TOOLS_ERROR_H_
#define __CUDA_TOOLS_ERROR_H_

#include <cuda_runtime.h>

#include <iostream>

#include "exception.h"

#define CUDA_RT_CALL(call) ct::cuda_rt_call((call), #call, __FILE__, __LINE__)

// checks last error
#define CUDA_GET_LAST_ERROR() CUDA_RT_CALL(cudaGetLastError())
#define CUDA_PEEK_AT_LAST_ERROR() CUDA_RT_CALL(cudaPeekAtLastError())

// forwards cudaLaunchKernel arguments and check for errors
#define CUDA_LAUNCH_KERNEL(...) CUDA_RT_CALL(cudaLaunchKernel(__VA_ARGS__))

namespace ct
{

/**
 * @brief wrapper for calling a cuda function and checks
 * the status of the function
 *
 * @param status function status
 * @param func_str function information
 * @param file file name where this function is called
 * @param line line number where this function is called
 * @throws ct::CudaRTException Thrown if cuda error found, catch this exception
 * if you don't want the program to exit
 */
void cuda_rt_call(
    cudaError_t status,
    const char* const func_str,
    const char* const file,
    const int line)
{
    if (status != cudaSuccess)
    {
        throw CudaRTException(status, func_str, file, line);
    }
}

}  // namespace ct

#endif  // __CUDA_TOOLS_ERROR_H_
