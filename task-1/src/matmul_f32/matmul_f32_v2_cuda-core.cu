#include "../cuda/cuda_common.h"
#include "../xmx/fp32/fp32_v2.cuh"
#include "playground/matmul.hpp"
#include <cstdio>

namespace playground
{
// Implement the matmul function with DType=float16_t and Version=2
PLAYGROUND_MATMUL_DEC(float32_t, 2, M, N, K, A, B, C)
{
    // ......
    dim3 blocksize = {32, 32};
    dim3 gridsize = {CEIL_DIV(N, blocksize.x), CEIL_DIV(M, blocksize.y)};
    gemm_v2<<<gridsize, blocksize>>>(M, N, K, A, B, C);

    cudaCheck(cudaGetLastError());
}
}  // namespace playground