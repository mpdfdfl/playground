#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include "../xmx/fp16/fp16_v17.cuh"
#include "playground/matmul.hpp"
#include "playground/static.hpp"
#include "playground/system.hpp"
#include <cstdio>
namespace playground
{
PLAYGROUND_MATMUL_DEC(float16_t, 17, M, N, K, A, B, C)
{

    dim3 dimBlock(32, 2, 4);
    dim3 dimGrid(N / 128, M / 256);
    size_t smem_size = (128 * 32 + 256 * 32) * sizeof(float16_t) * 4;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error before kernel: %s\n", cudaGetErrorString(err));
    }
    cudaFuncSetAttribute(gemm_fp16_v17,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
    gemm_fp16_v17<<<dimGrid, dimBlock, smem_size>>>(A, B, C, M, N, K);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel: %s\n", cudaGetErrorString(err));
    }
}
}  // namespace playground
