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

__device__ void loadFragA(unsigned int* frag, float16_t* smem, int ki)
{
    // load 64x16 fp16 per warps
    int tx = threadIdx.x;
    int tz = threadIdx.z;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int row = tz * 64 + i * 16 + j * 8 + tx / 4;
                int col = ki * KII + k * 8 + (tx % 4) * 2;

                col = col ^ ((row & 3) << 3);

                unsigned int* ptr =
                    reinterpret_cast<unsigned int*>(smem + row * 32 + col);
                frag[i * 4 + j * 2 + k] = ptr[0];
            }
        }
    }
}

__device__ void loadFragB(unsigned int* frag, float16_t* smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int row = ki * 32 + tx % 4 * 2;
                int col = ty * 64 + 16 * i + j * 8 + tx / 4;

                int col1 = col ^ ((row & 3) << 3);
                int col2 = col ^ (((row + 1) & 3) << 3);
                // 从 smem 读取 (row, col) 和 (row + 1, col) 的 fp16 值
                float16_t fp16_1 = smem[OFFSET(row, col1, 128)];
                float16_t fp16_2 = smem[OFFSET(row + 1, col2, 128)];

                // 将 fp16 转换为 uint16_t 以获取位模式
                uint16_t bits_1 = *reinterpret_cast<uint16_t*>(&fp16_1);
                uint16_t bits_2 = *reinterpret_cast<uint16_t*>(&fp16_2);

                // 合并为 unsigned int：bits_1 占高 16 位，bits_2 占低 16 位
                frag[i * 4 + j * 2 + k] =
                    (static_cast<unsigned int>(bits_1) << 16) |
                    static_cast<unsigned int>(bits_2);
            }
        }
    }
}

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

__global__ void gemm_fp16_v8(const float16_t* A, const float16_t* B,
                             float16_t* const C, int M, int N, int K)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    // float16_t* SA = reinterpret_cast<float16_t*>(shared_storage);
    // float16_t* SB = reinterpret_cast<float16_t*>(
    //     shared_storage + 2 * MI * KI * sizeof(float16_t));

    float16_t* SA1 = reinterpret_cast<float16_t*>(shared_storage);
    float16_t* SA2 = SA1 + MI * KI;
    float16_t* SA3 = SA2 + MI * KI;
    float16_t* SA4 = SA3 + MI * KI;

    float16_t* SB1 = SA4 + MI * KI;
    float16_t* SB2 = SB1 + NI * KI;
    float16_t* SB3 = SB2 + NI * KI;
    float16_t* SB4 = SB3 + NI * KI;

    float16_t* SA[] = {SA1, SA2, SA3, SA4};
    float16_t* SB[] = {SB1, SB2, SB3, SB4};

    unsigned int FragA[4 * 4];
    unsigned int FragB[4 * 2 * 2];
    float Accum[4 * 4 * 8] = {0.0};

    for (int i = 0; i < 3; ++i) {
        loadSmemA(SA[i], A, M, K, i);
        loadSmemB(SB[i], B, N, K, i);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    // asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    // __syncthreads();
    // blockTiling
    for (int ko = 0; ko < K / KI; ko++) {

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();

        // loadFragA(FragA, SA[(ko) % 4], 0);
        // loadFragB(FragB, SB[(ko) % 4], 0);

        // // warpTiling
        // int offset_SA = smem_sel * MI * KI;
        // int offset_SB = smem_sel * KI * NI;
        for (int ki = 0; ki < KI / KII; ki++) {
            loadFragA(FragA, SA[ko % 4], ki);
            loadFragB(FragB, SB[ko % 4], ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1) {
                for (int nii = 0; nii < NII / wmmaN; nii += 1) {
                    mmaSync(FragA + mii * 4, FragB + nii * 4,
                            Accum + mii * 4 * 8 + nii * 8);
                }
            }
        }
        if (ko + 3 < K / KI) {
            loadSmemA(SA[(ko + 3) % 4], A, M, K, ko + 3);
            loadSmemB(SB[(ko + 3) % 4], B, N, K, ko + 3);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        // loadSmemA(SA + offset_SA, A, M, K, ko);
        // loadSmemB(SB + offset_SB, B, K, N, ko);
        // asm volatile("cp.async.commit_group;\n" ::);

        // asm volatile("cp.async.commit_group;\n" ::);
        // asm volatile("cp.async.wait_all;\n" ::);
        // __syncthreads();
    }
    // {
    //     int offset_SA = 1 * MI * KI;
    //     int offset_SB = 1 * KI * NI;
    //     for (int ki = 0; ki < KI / KII; ki++) {
    //         loadFragA(FragA, SA + offset_SA, ki);
    //         loadFragB(FragB, SB + offset_SB, ki);
    //         for (int mii = 0; mii < MII / wmmaM; mii += 1) {
    //             for (int nii = 0; nii < NII / wmmaN; nii += 1) {
    //                 // 16x16x16 for each wmma
    //                 mmaSync(FragA + mii * 4, FragB + nii * 4,
    //                         Accum + mii * 4 * 8 + nii * 8);
    //             }
    //         }
    //     }
    // }
    storeAccum(C, Accum, M, N);
}
}  // namespace playground
