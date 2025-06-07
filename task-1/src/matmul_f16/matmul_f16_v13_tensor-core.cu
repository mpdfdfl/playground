// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <library_types.h>

// #include "../xmx/fp16/fp16_v13.cuh"
// #include "playground/matmul.hpp"
// #include "playground/static.hpp"
// #include "playground/system.hpp"
// #include <cstdio>
// namespace playground
// {
// PLAYGROUND_MATMUL_DEC(float16_t, 13, M, N, K, A, B, C)
// {

//     dim3 dimBlock(32, 4, 2);
//     dim3 dimGrid(N / 256, M / 128);
//     size_t smem_size =
//         (128 * (32 + 20) + (256 + 20) * 32) * sizeof(float16_t) * 3;
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error before kernel: %s\n", cudaGetErrorString(err));
//     }
//     cudaFuncSetAttribute(
//         gemm_fp16_v13, cudaFuncAttributeMaxDynamicSharedMemorySize,
//         smem_size);
//     gemm_fp16_v13<<<dimGrid, dimBlock, smem_size>>>(A, B, C, M, N, K);

//     err = cudaDeviceSynchronize();
//     if (err != cudaSuccess) {
//         printf("CUDA Error after kernel: %s\n", cudaGetErrorString(err));
//     }
// }
// }  // namespace playground