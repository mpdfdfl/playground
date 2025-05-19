/*
Common utilities for CUDA code.
*/
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>  // std::bool_constant

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#define WARP_SIZE 32U

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4_CONST(value) (reinterpret_cast<const float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])
#define FLOAT2_CONST(value) (reinterpret_cast<const float2*>(&(value))[0])
#define OFFSET(a, b, c) ((a) * (c) + (b))

constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly
// not just via macro
inline void cudaCheck_(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
template <class T>
inline void cudaFreeCheck(T** ptr, const char* file, int line)
{
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))
