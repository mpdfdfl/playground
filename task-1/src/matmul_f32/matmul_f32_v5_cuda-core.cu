#include "../xmx/fp32/fp32_v5.cuh"
#include "playground/matmul.hpp"
#include <cstdio>

namespace playground
{
// Implement the matmul function with DType=float16_t and Version=2
PLAYGROUND_MATMUL_DEC(float32_t, 5, M, N, K, A, B, C)
{
    // ......
    dim3 blocksize = {16, 32};
    unsigned int gridX = static_cast<unsigned int>((N + 128 - 1) / 128);
    unsigned int gridY = static_cast<unsigned int>((M + 128 - 1) / 128);
    dim3 gridsize = {gridX, gridY};
    gemm_v5<<<gridsize, blocksize>>>(M, N, K, A, B, C);

    // 检查核函数启动是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("核函数启动错误: %s\n", cudaGetErrorString(err));
    }

    // 同步设备并检查执行期间的错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("核函数执行错误: %s\n", cudaGetErrorString(err));
    }
}
}  // namespace playground