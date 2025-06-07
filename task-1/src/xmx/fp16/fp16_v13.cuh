#pragma once
#include "../../cuda/cuda_common.h"
#include "playground/matmul.hpp"
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>

namespace playground
{
const int MI = 128;
const int NI = 256;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;
const int PAD = 8;
__device__ void loadSmemA(float16_t* smem, const float16_t* A, int M, int K,
                          int ko)
{
    // 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 32 f16 for thread
    int tid = tx + ty * 32 + tz * 4 * 32;
    // int s_a_base_addr = __cvta_generic_to_shared(&smem[0]);
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int load_a_smem_m = (tid >> 2) + i * 64;
        int load_a_smem_k = (tid & 3) << 3;

        int load_a_gmem_m = by * 128 + load_a_smem_m;
        int load_a_gmem_k = ko * 32 + load_a_smem_k;

        void* ptr =
            (void*) (smem + OFFSET(load_a_smem_m, load_a_smem_k, 32 + PAD));
        uint32_t smem_ptr;
        asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
            "cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(smem_ptr), "l"(&A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]));
        // asm volatile("cp.async.commit_group;\n" ::);
        // asm volatile("cp.async.wait_all;\n" ::);
    }
}

__device__ void loadSmemB(float16_t* smem, const float16_t* B, int K, int N,
                          int ko)
{
    // load 32 * 256

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // 32 f16 for thread
    int tid = tx + ty * 32 + tz * 4 * 32;
    // int s_b_base_addr = __cvta_generic_to_shared(&smem[0]);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int load_b_smem_k = (tid >> 5) + i * 8;
        int load_b_smem_n = (tid & 32) << 3;

        int load_b_gmem_k = ko * 32 + load_b_smem_k;
        int load_b_gmem_n = bx * 256 + load_b_smem_n;

        // int load_b_smem_addr =
        //     s_b_base_addr +
        //     OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD) *
        //     sizeof(float16_t);

        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)];
        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)+1] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)+1];
        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)+2] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)+2];
        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)+3] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)+3];

        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)+4] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)+4];
        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)+5] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)+5];
        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)+6] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)+6];
        // smem[OFFSET(load_b_smem_k, load_b_smem_n, 128 + PAD)+7] =
        // B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)+7];

        void* ptr =
            (void*) (smem + OFFSET(load_b_smem_k, load_b_smem_n, 256 + PAD));
        uint32_t smem_ptr;
        asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
            "cvt.u32.u64 %0, smem_ptr; }\n"
            : "=r"(smem_ptr)
            : "l"(ptr));
        asm("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(smem_ptr), "l"(&B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]));
        // asm volatile("cp.async.commit_group;\n" ::);
        // asm volatile("cp.async.wait_all;\n" ::);
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
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row * (32 + PAD) + col,
                                       32 + PAD);
    }
}

__device__ void loadFragB(
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK,
                           float16_t, nvcuda::wmma::row_major>* frag,
    float16_t* smem, int ki)
{
    // load 16*64
    int ty = threadIdx.y;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = ki * KII;
        int col = ty * 64 + i * 16;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row * (256 + PAD) + col,
                                       256 + PAD);
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
            int col = bx * 256 + ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(ptr + row * N + col,
                                            frag[i * 4 + j], N,
                                            nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void gemm_fp16_v13(const float16_t* A, const float16_t* B,
                              float16_t* const C, int M, int N, int K)
{
    extern __shared__ uint8_t shared_storage[];
    float16_t* SA = reinterpret_cast<float16_t*>(shared_storage);
    float16_t* SB = reinterpret_cast<float16_t*>(
        shared_storage + 3 * MI * (KI + PAD) * sizeof(float16_t));

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

    int offset_SA = 0;
    int offset_SB = 0;

    {
        loadSmemA(SA + offset_SA, A, M, K, 0);
        loadSmemB(SB + offset_SB, B, K, N, 0);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    offset_SA += MI * (KI + PAD);
    offset_SB += KI * (NI + PAD);

    {
        loadSmemA(SA + offset_SA, A, M, K, 1);
        loadSmemB(SB + offset_SB, B, K, N, 1);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int ko = 2; ko < K / KI; ko++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));

        int smem_sel = (ko - 2) % 3;
        int smem_sel_next = ko % 3;
        offset_SA = smem_sel * MI * (KI + PAD);
        offset_SB = smem_sel * KI * (NI + PAD);
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
        offset_SA = smem_sel_next * MI * (KI + PAD);
        offset_SB = smem_sel_next * KI * (NI + PAD);
        loadSmemA(SA + offset_SA, A, M, K, ko);
        loadSmemB(SB + offset_SB, B, K, N, ko);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        int smem_sel = (K / KI - 2) % 3;
        int offset_SA = smem_sel * MI * (KI + PAD);
        int offset_SB = smem_sel * KI * (NI + PAD);
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
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        int smem_sel = (K / KI - 1) % 3;
        int offset_SA = smem_sel * MI * (KI + PAD);
        int offset_SB = smem_sel * KI * (NI + PAD);
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