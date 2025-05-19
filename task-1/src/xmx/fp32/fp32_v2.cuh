#pragma once
#include "../../cuda/cuda_common.h"
#include "playground/matmul.hpp"
#include <cstdio>

namespace playground
{

__global__ void gemm_v2(const size_t M, const size_t N, const size_t K,
                        const float32_t* const __restrict__ A,
                        const float32_t* const __restrict__ B,
                        float32_t* const __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    float32_t sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += A[OFFSET(row, k, K)] * B[OFFSET(k, col, N)];
    }
    C[OFFSET(row, col, N)] = sum;
}
}  // namespace playground
