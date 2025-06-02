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
    int tid = tx + ty * 32 + tz * 2 * 32;
    // int s_a_base_addr = __cvta_generic_to_shared(&smem[0]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int load_a_smem_m = ((tid >> 2) << 2) + i;
        int load_a_smem_k = (tid & 3) << 3;

        int load_a_gmem_m = by * 128 + load_a_smem_m;
        int load_a_gmem_k = ko * 32 + load_a_smem_k;

        load_a_smem_k = load_a_smem_k ^ (((load_a_smem_m & 3) << 3));

        // int load_a_smem_addr =
        //     s_a_base_addr +
        //     OFFSET(load_a_smem_m, load_a_smem_k, 32 + PAD) *
        //     sizeof(float16_t);

        void* ptr = (void*) (smem + OFFSET(load_a_smem_m, load_a_smem_k, 32));
        uint32_t smem_ptr;
        asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
            "cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(smem_ptr), "l"(&A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]));
    }
}

__device__ void loadSmemB(float16_t* smem, const float16_t* B, int K, int N,
                          int ko)
{
    // load 32 * 128

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 32 f16 for thread
    int tid = tx + ty * 32 + tz * 2 * 32;
    // int s_b_base_addr = __cvta_generic_to_shared(&smem[0]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int load_b_smem_k = ((tid >> 4) << 2) + i;
        int load_b_smem_n = (tid & 15) << 3;

        int load_b_gmem_k = ko * 32 + load_b_smem_k;
        int load_b_gmem_n = bx * 128 + load_b_smem_n;

        load_b_smem_n = load_b_smem_n ^ (((load_b_smem_k & 3) << 3));

        // int load_b_smem_addr =
        //     s_b_base_addr +
        //     OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD) *
        //     sizeof(float16_t);
        void* ptr = (void*) (smem + OFFSET(load_b_smem_k, load_b_smem_n, 128));
        uint32_t smem_ptr;
        asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
            "cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(smem_ptr), "l"(&B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]));
        // asm volatile("cp.async.commit_group;\n" ::);
    }
}

__device__ void loadFragA(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::row_major>* frag,
    float16_t* smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        col = col ^ (((row & 3) << 3));

        nvcuda::wmma::load_matrix_sync(frag[i], smem + row * 32 + col, 32);
    }
}

__device__ void loadFragB(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::row_major>* frag,
    float16_t* smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = ki * KII;
        int col = ty * 64 + i * 16;
        col = col ^ (((row & 3) << 3));

        nvcuda::wmma::load_matrix_sync(frag[i], smem + row * 128 + col, 128);
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
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
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

__global__ void gemm_fp16_v9(const float16_t* A, const float16_t* B,
                             float16_t* const C, int M, int N, int K)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    float16_t* SA = reinterpret_cast<float16_t*>(shared_storage);
    float16_t* SB = reinterpret_cast<float16_t*>(
        shared_storage + 2 * MI * KI * sizeof(float16_t));

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::row_major>
        FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::row_major>
        FragB[NII / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK,
                           float16_t>
        Accum[MII / wmmaM * NII / wmmaN];
#pragma unroll
    for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
        for (int nii = 0; nii < NII / wmmaN; nii += 1) {
            nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
        }
    }

    {
        loadSmemA(SA, A, M, K, 0);
        loadSmemB(SB, B, K, N, 0);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    // blockTiling
    for (int ko = 1; ko < K / KI; ko++) {

        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        __syncthreads();
        int smem_sel = (ko - 1) & 1;
        int smem_sel_next = ko & 1;
        // warpTiling
        int offset_SA = smem_sel * MI * KI;
        int offset_SB = smem_sel * KI * NI;
#pragma unroll
        for (int ki = 0; ki < KI / KII; ki++) {
            loadFragA(FragA, SA + offset_SA, ki);
            loadFragB(FragB, SB + offset_SB, ki);
#pragma unroll
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
                for (int nii = 0; nii < NII / wmmaN; nii += 1) {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii],
                                           FragA[mii], FragB[nii],
                                           Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }
        offset_SA = smem_sel_next * MI * KI;
        offset_SB = smem_sel_next * KI * NI;
        loadSmemA(SA + offset_SA, A, M, K, ko);
        loadSmemB(SB + offset_SB, B, K, N, ko);
        asm volatile("cp.async.commit_group;\n" ::);

        // asm volatile("cp.async.commit_group;\n" ::);
        // asm volatile("cp.async.wait_all;\n" ::);
        // __syncthreads();
    }
    {
        int offset_SA = 1 * MI * KI;
        int offset_SB = 1 * KI * NI;
#pragma unroll
        for (int ki = 0; ki < KI / KII; ki++) {
            loadFragA(FragA, SA + offset_SA, ki);
            loadFragB(FragB, SB + offset_SB, ki);
#pragma unroll
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
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
}
}  // namespace playground
