// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <library_types.h>

// #include "../xmx/fp16/fp16_v3.cuh"
// #include "playground/matmul.hpp"
// #include "playground/static.hpp"
// #include "playground/system.hpp"
// #include <cstdio>
// namespace playground
// {
// PLAYGROUND_MATMUL_SIG(float16_t, 3, M, N, K, A, B, C)
// {

//     dim3 dimBlock(32, 2, 2);
//     dim3 dimGrid(N / 128, M / 128);
//     size_t smem_size = 128 * 32 * 2 * sizeof(float16_t);
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error before kernel: %s\n", cudaGetErrorString(err));
//     }
//     gemm_fp16_v3<<<dimGrid, dimBlock, smem_size>>>(A, B, C, M, N, K);

//     err = cudaDeviceSynchronize();
//     if (err != cudaSuccess) {
//         printf("CUDA Error after kernel: %s\n", cudaGetErrorString(err));
//     }
// }
// }  // namespace playground
