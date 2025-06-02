#pragma once
#include "../../cuda/cuda_common.h"
#include "playground/matmul.hpp"
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>

namespace playground
{
const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__device__ void loadSmemA(float16_t* smem, const float16_t* A, int M, int K,
                          int ko)
{
    // 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 32 f16 for thread

    for (int i = 0; i < 32; i++) {
        int load_a_smem_m = i * 4 + tz * 2 + ty;
        int load_a_smem_k = tx;

        int store_a_gmem_m = by * 128 + load_a_smem_m;
        int store_a_gmem_k = ko * 32 + load_a_smem_k;

        smem[OFFSET(load_a_smem_m, load_a_smem_k, KI)] =
            A[OFFSET(store_a_gmem_m, store_a_gmem_k, K)];
    }
}

__device__ void loadSmemB(float16_t* smem, const float16_t* B, int K, int N,
                          int ko)
{
    // load 128 * 32

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 32 f16 for thread

    for (int i = 0; i < 32; i++) {
        int load_b_smem_k = tx;
        int load_b_smem_n = i * 4 + tz * 2 + ty;

        int store_b_gmem_k = ko * 32 + load_b_smem_k;
        int store_b_gmem_n = bx * 128 + load_b_smem_n;

        smem[OFFSET(load_b_smem_n, load_b_smem_k, KI)] =
            B[OFFSET(store_b_gmem_k, store_b_gmem_n, N)];
    }
}

__device__ void loadFragA(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::row_major>* frag,
    float16_t* smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row * 32 + col, 32);
    }
}

__device__ void loadFragB(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::col_major>* frag,
    float16_t* smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i) {
        int row = ty * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row * 32 + col, 32);
    }
}

__device__ void storeAccum(
    float16_t* ptr,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK,
                           float16_t>* frag,
    int M, int N)
{
    // store 64x64
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int row = by * 128 + tz * 64 + i * 16;
            int col = bx * 128 + ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(ptr + row * N + col,
                                            frag[i * 4 + j], N,
                                            nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void gemm_fp16_v2(const float16_t* A, const float16_t* B,
                             float16_t* const C, int M, int N, int K)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    float16_t* SA = reinterpret_cast<float16_t*>(shared_storage);
    float16_t* SB = reinterpret_cast<float16_t*>(shared_storage +
                                                 MI * KI * sizeof(float16_t));

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::row_major>
        FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::col_major>
        FragB[NII / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK,
                           float16_t>
        Accum[MII / wmmaM * NII / wmmaN];

    for (int mii = 0; mii < MII / wmmaM; mii += 1) {
        for (int nii = 0; nii < NII / wmmaN; nii += 1) {
            nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
        }
    }

    // blockTiling
    for (int ko = 0; ko < K / KI; ko++) {
        loadSmemA(SA, A, M, K, ko);
        loadSmemB(SB, B, K, N, ko);
        __syncthreads();
        // warpTiling
        for (int ki = 0; ki < KI / KII; ki++) {
            loadFragA(FragA, SA, ki);
            loadFragB(FragB, SB, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
                for (int nii = 0; nii < NII / wmmaN; nii += 1) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii],
                                           FragA[mii], FragB[nii],
                                           Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }
    }
    storeAccum(C, Accum, M, N);
    // for (int ko = 0; ko < K / KI; ko += 1) {
    //     loadSmemA(SA, A, M, K, ko);
    //     loadSmemB(SB, B, N, K, ko);
    //     __syncthreads();
    //     for (int ki = 0; ki < KI / KII; ki += 1) {
    //         // 64x64x16 mma for each warp
    //         loadFragA(FragA, SA, ki);
    //         loadFragB(FragB, SB, ki);
    //         for (int mii = 0; mii < MII / wmmaM; mii += 1) {
    //             for (int nii = 0; nii < NII / wmmaN; nii += 1) {
    //                 // 16x16x16 for each wmma
    //                 nvcuda::wmma::mma_sync(
    //                     Accum[mii * (NII / wmmaN) + nii], FragA[mii],
    //                     FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
    //             }
    //         }
    //     }
    // }
}
}  // namespace playground
