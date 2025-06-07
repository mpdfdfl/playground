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

// 16 * 16
__device__ void mmaSync(unsigned int* fragA, unsigned int* fragB, float* accum)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]),
                   "=f"(accum[5])
                 : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
                   "r"(fragB[0]), "r"(fragB[1]), "f"(accum[0]), "f"(accum[1]),
                   "f"(accum[4]), "f"(accum[5]));

    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,  %1,  %2,  %3},"
                 "{%4,  %5,  %6,  %7},"
                 "{%8,  %9},"
                 "{%10, %11, %12, %13};\n"
                 : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]),
                   "=f"(accum[7])
                 : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
                   "r"(fragB[2]), "r"(fragB[3]), "f"(accum[2]), "f"(accum[3]),
                   "f"(accum[6]), "f"(accum[7]));
}

__device__ void storeAccum(float16_t* ptr, float32_t* frag, int M, int N)
{
    // store 64x64
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tx = threadIdx.x;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int row = by * 128 + tz * 64 + i * 16 + tx / 4;
            int col = bx * 128 + ty * 64 + j * 16 + (tx % 4) * 2;
            float32_t* frag_base = frag + i * 4 * 8 + j * 8;

            ptr[OFFSET(row, col, N)] = float16_t(frag_base[0]);
            ptr[OFFSET(row, col + 1, N)] = float16_t(frag_base[1]);

            ptr[OFFSET(row + 8, col, N)] = float16_t(frag_base[3]);
            ptr[OFFSET(row + 8, col + 1, N)] = float16_t(frag_base[4]);

            ptr[OFFSET(row, col + 8, N)] = float16_t(frag_base[5]);
            ptr[OFFSET(row, col + 1 + 8, N)] = float16_t(frag_base[6]);

            ptr[OFFSET(row + 8, col + 8, N)] = float16_t(frag_base[7]);
            ptr[OFFSET(row + 8, col + 1 + 8, N)] = float16_t(frag_base[8]);
        }
    }
}

__global__ void gemm_fp16_v15(const float16_t* A, const float16_t* B,
                              float16_t* const C, int M, int N, int K)
{
    extern __shared__ uint8_t shared_storage[];
    float16_t* SA = reinterpret_cast<float16_t*>(shared_storage);
    float16_t* SB = reinterpret_cast<float16_t*>(
        shared_storage + 4 * MI * KI * sizeof(float16_t));

    unsigned int FragA[4 * 4];
    unsigned int FragB[4 * 2 * 2];
    float Accum[4 * 4 * 8] = {0.0};

    int offset_SA = 0;
    int offset_SB = 0;

    {
        // loadSmemA(SA + offset_SA, A, M, K, 0);
        {  // 128 * 32
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
                int load_a_gmem_k = 0 * 32 + load_a_smem_k;

                load_a_smem_k = load_a_smem_k ^ (((load_a_smem_m & 3) << 3));

                // int load_a_smem_addr =
                //     s_a_base_addr +
                //     OFFSET(load_a_smem_m, load_a_smem_k, 32 + PAD) *
                //     sizeof(float16_t);

                void* ptr = (void*) (SA + offset_SA +
                                     OFFSET(load_a_smem_m, load_a_smem_k, 32));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]));
            }
        }

        // loadSmemB(SB + offset_SB, B, K, N, 0);
        {
            // load 128 * 32

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

                int load_b_gmem_k = 0 * 32 + load_b_smem_k;
                int load_b_gmem_n = bx * 128 + load_b_smem_n;

                load_b_smem_n = load_b_smem_n ^ (((load_b_smem_k & 3) << 3));
                void* ptr =
                    (void*) (SB + offset_SB +
                             OFFSET(load_b_smem_k, load_b_smem_n, 128));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    offset_SA += MI * KI;
    offset_SB += KI * NI;

    {
        // loadSmemA(SA + offset_SA, A, M, K, 1);
        // loadSmemB(SB + offset_SB, B, K, N, 1);

        // loadSmemA(SA + offset_SA, A, M, K, 1);
        {  // 128 * 32
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
                int load_a_gmem_k = 1 * 32 + load_a_smem_k;

                load_a_smem_k = load_a_smem_k ^ (((load_a_smem_m & 3) << 3));

                // int load_a_smem_addr =
                //     s_a_base_addr +
                //     OFFSET(load_a_smem_m, load_a_smem_k, 32 + PAD) *
                //     sizeof(float16_t);

                void* ptr = (void*) (SA + offset_SA +
                                     OFFSET(load_a_smem_m, load_a_smem_k, 32));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]));
            }
        }

        // loadSmemB(SB + offset_SB, B, K, N, 1);
        {
            // load 128 * 32

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

                int load_b_gmem_k = 1 * 32 + load_b_smem_k;
                int load_b_gmem_n = bx * 128 + load_b_smem_n;

                load_b_smem_n = load_b_smem_n ^ (((load_b_smem_k & 3) << 3));
                void* ptr =
                    (void*) (SB + offset_SB +
                             OFFSET(load_b_smem_k, load_b_smem_n, 128));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    offset_SA += MI * KI;
    offset_SB += KI * NI;

    {
        // loadSmemA(SA + offset_SA, A, M, K, 2);
        // loadSmemB(SB + offset_SB, B, K, N, 2);

        // loadSmemA(SA + offset_SA, A, M, K, 0);
        {  // 128 * 32
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
                int load_a_gmem_k = 2 * 32 + load_a_smem_k;

                load_a_smem_k = load_a_smem_k ^ (((load_a_smem_m & 3) << 3));

                // int load_a_smem_addr =
                //     s_a_base_addr +
                //     OFFSET(load_a_smem_m, load_a_smem_k, 32 + PAD) *
                //     sizeof(float16_t);

                void* ptr = (void*) (SA + offset_SA +
                                     OFFSET(load_a_smem_m, load_a_smem_k, 32));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]));
            }
        }

        // loadSmemB(SB + offset_SB, B, K, N, 0);
        {
            // load 128 * 32

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

                int load_b_gmem_k = 2 * 32 + load_b_smem_k;
                int load_b_gmem_n = bx * 128 + load_b_smem_n;

                load_b_smem_n = load_b_smem_n ^ (((load_b_smem_k & 3) << 3));
                void* ptr =
                    (void*) (SB + offset_SB +
                             OFFSET(load_b_smem_k, load_b_smem_n, 128));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ko = 3; ko < K / KI; ko++) {

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();

        int smem_sel = (ko - 3) % 4;
        int smem_sel_next = ko % 4;
        offset_SA = smem_sel * MI * (KI);
        offset_SB = smem_sel * KI * (NI);

        for (int ki = 0; ki < KI / KII; ki++) {
            // loadFragA(FragA, SA + offset_SA, ki);
            {
                // load 64x16 fp16 per warps
                int tx = threadIdx.x;
                int tz = threadIdx.z;

                for (int i = 0; i < 4; i++) {
                    int row = tz * 64 + tx % 16;
                    int col = ki * KII + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SA + offset_SA + row * 32 + col);

                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { "
                                 "%0, %1, %2, %3 }, [ "
                                 "%4 ];\n"
                                 : "=r"(FragA[i * 4]), "=r"(FragA[i * 4 + 1]),
                                   "=r"(FragA[i * 4 + 2]),
                                   "=r"(FragA[i * 4 + 3])
                                 : "r"(smem_base));
                }
            }

            // loadFragB(FragB, SB + offset_SB, ki);
            //  load 64x16
            {
                int ty = threadIdx.y;
                int tx = threadIdx.x;

                for (int i = 0; i < 4; i++) {

                    int row = ki * 16 + tx % 16;
                    int col = ty * 64 + i * 16 + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SB + offset_SB + row * 128 + col);

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                        "{ %0, %1, %2, %3 "
                        "}, "
                        "[ %4 "
                        "];\n"
                        : "=r"(FragB[i * 4]), "=r"(FragB[i * 4 + 1]),
                          "=r"(FragB[i * 4 + 2]), "=r"(FragB[i * 4 + 3])
                        : "r"(smem_base));
                }
            }
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
                for (int nii = 0; nii < NII / wmmaN; nii += 1) {
                    mmaSync(FragA + mii * 4, FragB + nii * 4,
                            Accum + mii * 4 * 8 + nii * 8);
                }
            }
        }
        offset_SA = smem_sel_next * MI * KI;
        offset_SB = smem_sel_next * KI * NI;
        // loadSmemA(SA + offset_SA, A, M, K, ko);
        // loadSmemB(SB + offset_SB, B, K, N, ko);

        // loadSmemA(SA + offset_SA, A, M, K, 0);
        {  // 128 * 32
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

                void* ptr = (void*) (SA + offset_SA +
                                     OFFSET(load_a_smem_m, load_a_smem_k, 32));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]));
            }
        }

        // loadSmemB(SB + offset_SB, B, K, N, 0);
        {
            // load 128 * 32

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
                void* ptr =
                    (void*) (SB + offset_SB +
                             OFFSET(load_b_smem_k, load_b_smem_n, 128));
                uint32_t smem_ptr;
                asm volatile(
                    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
                    "cvt.u32.u64 %0, smem_ptr; }\n"
                    : "=r"(smem_ptr)
                    : "l"(ptr));
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :
                    : "r"(smem_ptr),
                      "l"(&B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        int smem_sel = (K / KI - 3) % 4;
        int offset_SA = smem_sel * MI * KI;
        int offset_SB = smem_sel * KI * NI;
#pragma unroll
        for (int ki = 0; ki < KI / KII; ki++) {
            // loadFragA(FragA, SA + offset_SA, ki);
            {
                // load 64x16 fp16 per warps
                int tx = threadIdx.x;
                int tz = threadIdx.z;

                for (int i = 0; i < 4; i++) {
                    int row = tz * 64 + tx % 16;
                    int col = ki * KII + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SA + offset_SA + row * 32 + col);

                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { "
                                 "%0, %1, %2, %3 }, [ "
                                 "%4 ];\n"
                                 : "=r"(FragA[i * 4]), "=r"(FragA[i * 4 + 1]),
                                   "=r"(FragA[i * 4 + 2]),
                                   "=r"(FragA[i * 4 + 3])
                                 : "r"(smem_base));
                }
            }

            // loadFragB(FragB, SB + offset_SB, ki);
            //  load 64x16
            {
                int ty = threadIdx.y;
                int tx = threadIdx.x;

                for (int i = 0; i < 4; i++) {

                    int row = ki * 16 + tx % 16;
                    int col = ty * 64 + i * 16 + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SB + offset_SB + row * 128 + col);

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                        "{ %0, %1, %2, %3 "
                        "}, "
                        "[ %4 "
                        "];\n"
                        : "=r"(FragB[i * 4]), "=r"(FragB[i * 4 + 1]),
                          "=r"(FragB[i * 4 + 2]), "=r"(FragB[i * 4 + 3])
                        : "r"(smem_base));
                }
            }
#pragma unroll
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
                for (int nii = 0; nii < NII / wmmaN; nii += 1) {
                    mmaSync(FragA + mii * 4, FragB + nii * 4,
                            Accum + mii * 4 * 8 + nii * 8);
                }
            }
        }
    }
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        __syncthreads();
        int smem_sel = (K / KI - 2) % 4;
        int offset_SA = smem_sel * MI * KI;
        int offset_SB = smem_sel * KI * NI;
#pragma unroll
        for (int ki = 0; ki < KI / KII; ki++) {
            // loadFragA(FragA, SA + offset_SA, ki);
            {
                // load 64x16 fp16 per warps
                int tx = threadIdx.x;
                int tz = threadIdx.z;

                for (int i = 0; i < 4; i++) {
                    int row = tz * 64 + tx % 16;
                    int col = ki * KII + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SA + offset_SA + row * 32 + col);

                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { "
                                 "%0, %1, %2, %3 }, [ "
                                 "%4 ];\n"
                                 : "=r"(FragA[i * 4]), "=r"(FragA[i * 4 + 1]),
                                   "=r"(FragA[i * 4 + 2]),
                                   "=r"(FragA[i * 4 + 3])
                                 : "r"(smem_base));
                }
            }

            // loadFragB(FragB, SB + offset_SB, ki);
            //  load 64x16
            {
                int ty = threadIdx.y;
                int tx = threadIdx.x;

                for (int i = 0; i < 4; i++) {

                    int row = ki * 16 + tx % 16;
                    int col = ty * 64 + i * 16 + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SB + offset_SB + row * 128 + col);

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                        "{ %0, %1, %2, %3 "
                        "}, "
                        "[ %4 "
                        "];\n"
                        : "=r"(FragB[i * 4]), "=r"(FragB[i * 4 + 1]),
                          "=r"(FragB[i * 4 + 2]), "=r"(FragB[i * 4 + 3])
                        : "r"(smem_base));
                }
            }
#pragma unroll
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
                for (int nii = 0; nii < NII / wmmaN; nii += 1) {
                    mmaSync(FragA + mii * 4, FragB + nii * 4,
                            Accum + mii * 4 * 8 + nii * 8);
                }
            }
        }
    }
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
        __syncthreads();
        int smem_sel = (K / KI - 1) % 4;
        int offset_SA = smem_sel * MI * KI;
        int offset_SB = smem_sel * KI * NI;
#pragma unroll
        for (int ki = 0; ki < KI / KII; ki++) {
            // loadFragA(FragA, SA + offset_SA, ki);
            {
                // load 64x16 fp16 per warps
                int tx = threadIdx.x;
                int tz = threadIdx.z;

                for (int i = 0; i < 4; i++) {
                    int row = tz * 64 + tx % 16;
                    int col = ki * KII + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SA + offset_SA + row * 32 + col);

                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { "
                                 "%0, %1, %2, %3 }, [ "
                                 "%4 ];\n"
                                 : "=r"(FragA[i * 4]), "=r"(FragA[i * 4 + 1]),
                                   "=r"(FragA[i * 4 + 2]),
                                   "=r"(FragA[i * 4 + 3])
                                 : "r"(smem_base));
                }
            }

            // loadFragB(FragB, SB + offset_SB, ki);
            //  load 64x16
            {
                int ty = threadIdx.y;
                int tx = threadIdx.x;

                for (int i = 0; i < 4; i++) {

                    int row = ki * 16 + tx % 16;
                    int col = ty * 64 + i * 16 + (tx / 16) * 8;
                    col = col ^ (((row & 3) << 3));

                    uint32_t smem_base = __cvta_generic_to_shared(
                        SB + offset_SB + row * 128 + col);

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                        "{ %0, %1, %2, %3 "
                        "}, "
                        "[ %4 "
                        "];\n"
                        : "=r"(FragB[i * 4]), "=r"(FragB[i * 4 + 1]),
                          "=r"(FragB[i * 4 + 2]), "=r"(FragB[i * 4 + 3])
                        : "r"(smem_base));
                }
            }
#pragma unroll
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
#pragma unroll
                for (int nii = 0; nii < NII / wmmaN; nii += 1) {
                    mmaSync(FragA + mii * 4, FragB + nii * 4,
                            Accum + mii * 4 * 8 + nii * 8);
                }
            }
        }
    }
    // storeAccum(C, Accum, M, N);
    {
        // store 64x64
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int ty = threadIdx.y;
        int tz = threadIdx.z;
        int tx = threadIdx.x;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int row = by * 128 + tz * 64 + i * 16 + tx / 4;
                int col = bx * 128 + ty * 64 + j * 16 + (tx % 4) * 2;
                float32_t* frag_base = Accum + i * 4 * 8 + j * 8;

                C[OFFSET(row, col, N)] = float16_t(frag_base[0]);
                C[OFFSET(row, col + 1, N)] = float16_t(frag_base[1]);

                C[OFFSET(row + 8, col, N)] = float16_t(frag_base[3]);
                C[OFFSET(row + 8, col + 1, N)] = float16_t(frag_base[4]);

                C[OFFSET(row, col + 8, N)] = float16_t(frag_base[5]);
                C[OFFSET(row, col + 1 + 8, N)] = float16_t(frag_base[6]);

                C[OFFSET(row + 8, col + 8, N)] = float16_t(frag_base[7]);
                C[OFFSET(row + 8, col + 1 + 8, N)] = float16_t(frag_base[8]);
            }
        }
    }
}
}  // namespace playground