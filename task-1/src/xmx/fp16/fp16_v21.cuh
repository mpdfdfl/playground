#pragma once
#include "../../cuda/cuda_common.h"
#include "playground/matmul.hpp"
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>

namespace playground
{
#define INT4(value) (reinterpret_cast<int4*>((value))[0])

const int MI = 256;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int mmaM = 16;
const int mmaN = 8;
const int mmaK = 16;

// blockDIM 32 2 4
__global__ void gemm_fp16_v21(const float16_t* A, const float16_t* B,
                              float16_t* const C, int M, int N, int K)
{
    extern __shared__ uint8_t shared_storage[];

    float16_t* SA = reinterpret_cast<float16_t*>(shared_storage);
    float16_t* SB = reinterpret_cast<float16_t*>(
        shared_storage + 4 * MI * KI * sizeof(float16_t));
    float16_t* SC = reinterpret_cast<float16_t*>(shared_storage);

    uint32_t FragA[4][4];
    uint32_t FragB[8][2];
    uint32_t Accum[4][8][2] = {0};

    size_t bid = blockIdx.x + gridDim.x * blockIdx.y;
    // block swizzing
    // size_t bx = (bid / 32) * 2 + (blockIdx.x % 2);
    // size_t by = ((bid / 32) % 2) ? (15 - ((bid % 32) / 2)) : ((bid % 32) /
    // 2);

    size_t bx =
        (blockIdx.y % 2 == 0) ? (blockDim.x - 1 - blockIdx.x) : blockIdx.x;
    size_t by = blockIdx.y;

    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t tz = threadIdx.z;
    size_t tid = tx + ty * 32 + tz * 2 * 32;

    // 4 buffer

    // 0
    //  loadA
    //  256 * 32
    size_t offset_SA = 0;
    size_t offset_SB = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;

        int load_a_gmem_m = by * 256 + row;
        int load_a_gmem_k = 0 * 32 + col;

        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        void* ptr = (void*) (SA + offset_SA + OFFSET(row, col, 32));
        uint32_t smem_ptr = __cvta_generic_to_shared(ptr);
        // asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1;
        // "
        //              "cvt.u32.u64 %0, smem_ptr; }\n"
        //              : "=r"(smem_ptr)
        //              : "l"(ptr));
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                     :
                     : "r"(smem_ptr),
                       "l"(A + OFFSET(load_a_gmem_m, load_a_gmem_k, K)));
    }
    // loadB
    // 32*128
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;

        int load_b_gmem_k = 0 * 32 + row;
        int load_b_gmem_n = bx * 128 + col;

        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        void* ptr = (void*) (SB + offset_SB + OFFSET(row, col, 128));
        uint32_t smem_ptr = __cvta_generic_to_shared(ptr);

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                     :
                     : "r"(smem_ptr),
                       "l"(B + OFFSET(load_b_gmem_k, load_b_gmem_n, N)));
    }
    asm volatile("cp.async.commit_group;\n" ::);

    // 1
    //  loadA
    //  256 * 32
    offset_SA += MI * KI;
    offset_SB += KI * NI;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;

        int load_a_gmem_m = by * 256 + row;
        int load_a_gmem_k = 1 * 32 + col;

        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        void* ptr = (void*) (SA + offset_SA + OFFSET(row, col, 32));
        uint32_t smem_ptr = __cvta_generic_to_shared(ptr);

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                     :
                     : "r"(smem_ptr),
                       "l"(A + OFFSET(load_a_gmem_m, load_a_gmem_k, K)));
    }
    // loadB
    // 32*128
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;

        int load_b_gmem_k = 1 * 32 + row;
        int load_b_gmem_n = bx * 128 + col;

        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        void* ptr = (void*) (SB + offset_SB + OFFSET(row, col, 128));
        uint32_t smem_ptr = __cvta_generic_to_shared(ptr);

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                     :
                     : "r"(smem_ptr),
                       "l"(B + OFFSET(load_b_gmem_k, load_b_gmem_n, N)));
    }
    asm volatile("cp.async.commit_group;\n" ::);

    // 2
    //  loadA
    //  256 * 32
    offset_SA += MI * KI;
    offset_SB += KI * NI;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;

        int load_a_gmem_m = by * 256 + row;
        int load_a_gmem_k = 2 * 32 + col;

        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);

        void* ptr = (void*) (SA + offset_SA + OFFSET(row, col, 32));
        uint32_t smem_ptr = __cvta_generic_to_shared(ptr);

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                     :
                     : "r"(smem_ptr),
                       "l"(A + OFFSET(load_a_gmem_m, load_a_gmem_k, K)));
    }
    // loadB
    // 32*128
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;

        int load_b_gmem_k = 2 * 32 + row;
        int load_b_gmem_n = bx * 128 + col;

        col = col ^ ((row & ((1 << 4) - 1)) << 3);

        void* ptr = (void*) (SB + offset_SB + OFFSET(row, col, 128));
        uint32_t smem_ptr = __cvta_generic_to_shared(ptr);

        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                     :
                     : "r"(smem_ptr),
                       "l"(B + OFFSET(load_b_gmem_k, load_b_gmem_n, N)));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    // calculate
#pragma unroll
    for (int ko = 3; ko < K / KI; ko++) {

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();
        int smem_sel = (ko - 3) % 4;
        int smem_sel_next = ko % 4;
        offset_SA = smem_sel * MI * (KI);
        offset_SB = smem_sel * KI * (NI);

        for (int ki = 0; ki < KI / KII; ki++) {
// load fragA
#pragma unrool
            for (int i = 0; i < 4; i++) {
                int row = tz * 64 + i * 16 + tx % 16;
                int col = ki * KII + (tx / 16) * 8;
                col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);

                uint32_t smem_base =
                    __cvta_generic_to_shared(SA + offset_SA + row * 32 + col);

                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, "
                             "%1, %2, %3 }, [ "
                             "%4 ];\n"
                             : "=r"(FragA[i][0]), "=r"(FragA[i][1]),
                               "=r"(FragA[i][2]), "=r"(FragA[i][3])
                             : "r"(smem_base));
            }
// load fragB
#pragma unrool
            for (int i = 0; i < 8; i++) {

                int row = ki * 16 + tx % 16;
                int col = ty * 64 + i * 8;
                col = col ^ ((row & ((1 << 4) - 1)) << 3);
                uint32_t smem_base =
                    __cvta_generic_to_shared(SB + offset_SB + row * 128 + col);

                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                             "{ %0, %1"
                             "}, "
                             "[ %2 "
                             "];\n"
                             : "=r"(FragB[i][0]), "=r"(FragB[i][1])
                             : "r"(smem_base));
            }
#pragma unroll
            for (int mii = 0; mii < MII / mmaM; mii += 1) {
#pragma unroll
                for (int nii = 0; nii < NII / mmaN; nii += 1) {
                    int _nii = (mii & 1) ? (7 - nii) : nii;
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f16.f16."
                        "f16.f16 "
                        "{%0,  %1},"
                        "{%2,  %3,  %4,  %5},"
                        "{%6,  %7},"
                        "{%8, %9};\n"
                        : "=r"(Accum[mii][_nii][0]), "=r"(Accum[mii][_nii][1])
                        : "r"(FragA[mii][0]), "r"(FragA[mii][1]),
                          "r"(FragA[mii][2]), "r"(FragA[mii][3]),
                          "r"(FragB[_nii][0]), "r"(FragB[_nii][1]),
                          "r"(Accum[mii][_nii][0]), "r"(Accum[mii][_nii][1]));
                }
            }
        }

        offset_SA = smem_sel_next * MI * KI;
        offset_SB = smem_sel_next * KI * NI;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = ((tid >> 2) << 2) + i;
            int col = (tid & 3) << 3;

            int load_a_gmem_m = by * 256 + row;
            int load_a_gmem_k = ko * 32 + col;

            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            void* ptr = (void*) (SA + offset_SA + OFFSET(row, col, 32));
            uint32_t smem_ptr = __cvta_generic_to_shared(ptr);

            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                         :
                         : "r"(smem_ptr),
                           "l"(A + OFFSET(load_a_gmem_m, load_a_gmem_k, K)));
        }
        // loadB
        // 32*128
#pragma unroll
        for (int i = 0; i < 2; i++) {
            int row = ((tid >> 4) << 1) + i;
            int col = (tid & 15) << 3;

            int load_b_gmem_k = ko * 32 + row;
            int load_b_gmem_n = bx * 128 + col;

            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            void* ptr = (void*) (SB + offset_SB + OFFSET(row, col, 128));
            uint32_t smem_ptr = __cvta_generic_to_shared(ptr);

            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                         :
                         : "r"(smem_ptr),
                           "l"(B + OFFSET(load_b_gmem_k, load_b_gmem_n, N)));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    int smem_sel = (K / KI - 3) % 4;

    offset_SA = smem_sel * MI * KI;
    offset_SB = smem_sel * KI * NI;
#pragma unroll
    for (int ki = 0; ki < KI / KII; ki++) {
// load fragA
#pragma unrool
        for (int i = 0; i < 4; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = ki * KII + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);

            uint32_t smem_base =
                __cvta_generic_to_shared(SA + offset_SA + row * 32 + col);

            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, "
                         "%1, %2, %3 }, [ "
                         "%4 ];\n"
                         : "=r"(FragA[i][0]), "=r"(FragA[i][1]),
                           "=r"(FragA[i][2]), "=r"(FragA[i][3])
                         : "r"(smem_base));
        }
        // load fragB
#pragma unrool
        for (int i = 0; i < 8; i++) {

            int row = ki * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(SB + offset_SB + row * 128 + col);

            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                         "{ %0, %1"
                         "}, "
                         "[ %2 "
                         "];\n"
                         : "=r"(FragB[i][0]), "=r"(FragB[i][1])
                         : "r"(smem_base));
        }
#pragma unroll
        for (int mii = 0; mii < MII / mmaM; mii += 1) {
#pragma unroll
            for (int nii = 0; nii < NII / mmaN; nii += 1) {
                int _nii = (mii & 1) ? (7 - nii) : nii;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,  %1},"
                    "{%2,  %3,  %4,  %5},"
                    "{%6,  %7},"
                    "{%8, %9};\n"
                    : "=r"(Accum[mii][_nii][0]), "=r"(Accum[mii][_nii][1])
                    : "r"(FragA[mii][0]), "r"(FragA[mii][1]),
                      "r"(FragA[mii][2]), "r"(FragA[mii][3]),
                      "r"(FragB[_nii][0]), "r"(FragB[_nii][1]),
                      "r"(Accum[mii][_nii][0]), "r"(Accum[mii][_nii][1]));
            }
        }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();
    smem_sel = (K / KI - 2) % 4;
    offset_SA = smem_sel * MI * KI;
    offset_SB = smem_sel * KI * NI;
    for (int ki = 0; ki < KI / KII; ki++) {
        // load fragA
#pragma unrool
        for (int i = 0; i < 4; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = ki * KII + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);

            uint32_t smem_base =
                __cvta_generic_to_shared(SA + offset_SA + row * 32 + col);

            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, "
                         "%1, %2, %3 }, [ "
                         "%4 ];\n"
                         : "=r"(FragA[i][0]), "=r"(FragA[i][1]),
                           "=r"(FragA[i][2]), "=r"(FragA[i][3])
                         : "r"(smem_base));
        }
        // load fragB
#pragma unrool
        for (int i = 0; i < 8; i++) {

            int row = ki * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(SB + offset_SB + row * 128 + col);

            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                         "{ %0, %1"
                         "}, "
                         "[ %2 "
                         "];\n"
                         : "=r"(FragB[i][0]), "=r"(FragB[i][1])
                         : "r"(smem_base));
        }
#pragma unroll
        for (int mii = 0; mii < MII / mmaM; mii += 1) {
#pragma unroll
            for (int nii = 0; nii < NII / mmaN; nii += 1) {
                int _nii = (mii & 1) ? (7 - nii) : nii;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,  %1},"
                    "{%2,  %3,  %4,  %5},"
                    "{%6,  %7},"
                    "{%8, %9};\n"
                    : "=r"(Accum[mii][_nii][0]), "=r"(Accum[mii][_nii][1])
                    : "r"(FragA[mii][0]), "r"(FragA[mii][1]),
                      "r"(FragA[mii][2]), "r"(FragA[mii][3]),
                      "r"(FragB[_nii][0]), "r"(FragB[_nii][1]),
                      "r"(Accum[mii][_nii][0]), "r"(Accum[mii][_nii][1]));
            }
        }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
    smem_sel = (K / KI - 1) % 4;
    offset_SA = smem_sel * MI * KI;
    offset_SB = smem_sel * KI * NI;
#pragma unroll
    for (int ki = 0; ki < KI / KII; ki++) {
        // load fragA
#pragma unrool
        for (int i = 0; i < 4; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = ki * KII + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);

            uint32_t smem_base =
                __cvta_generic_to_shared(SA + offset_SA + row * 32 + col);

            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, "
                         "%1, %2, %3 }, [ "
                         "%4 ];\n"
                         : "=r"(FragA[i][0]), "=r"(FragA[i][1]),
                           "=r"(FragA[i][2]), "=r"(FragA[i][3])
                         : "r"(smem_base));
        }
        // load fragB
#pragma unrool
        for (int i = 0; i < 8; i++) {

            int row = ki * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(SB + offset_SB + row * 128 + col);

            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                         "{ %0, %1"
                         "}, "
                         "[ %2 "
                         "];\n"
                         : "=r"(FragB[i][0]), "=r"(FragB[i][1])
                         : "r"(smem_base));
        }
#pragma unroll
        for (int mii = 0; mii < MII / mmaM; mii += 1) {
#pragma unroll
            for (int nii = 0; nii < NII / mmaN; nii += 1) {
                int _nii = (mii & 1) ? (7 - nii) : nii;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,  %1},"
                    "{%2,  %3,  %4,  %5},"
                    "{%6,  %7},"
                    "{%8, %9};\n"
                    : "=r"(Accum[mii][_nii][0]), "=r"(Accum[mii][_nii][1])
                    : "r"(FragA[mii][0]), "r"(FragA[mii][1]),
                      "r"(FragA[mii][2]), "r"(FragA[mii][3]),
                      "r"(FragB[_nii][0]), "r"(FragB[_nii][1]),
                      "r"(Accum[mii][_nii][0]), "r"(Accum[mii][_nii][1]));
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (size_t i = 0; i < 4; ++i) {
#pragma unroll
        for (size_t j = 0; j < 8; ++j) {
            int row = tz * 64 + i * 16 + tx / 4;
            int col = ty * 64 + j * 8 + (tx % 4) * 2;
            int col1 = col ^ ((row & ((1 << 4) - 1)) << 3);
            int col2 = col ^ (((row + 8) & ((1 << 4) - 1)) << 3);

            (reinterpret_cast<uint32_t*>(SC + OFFSET(row, col1, NI)))[0] =
                Accum[i][j][0];

            (reinterpret_cast<uint32_t*>(SC + OFFSET(row + 8, col2, NI)))[0] =
                Accum[i][j][1];
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int row = i * 16 + tid / 16;
        int col = (tid % 16) * 8;
        int col2 = col ^ ((row & ((1 << 4) - 1)) << 3);
        INT4(C + OFFSET(by * 256 + row, bx * 128 + col, N)) =
            INT4(SC + OFFSET(row, col2, NI));
    }

    // for (int i = 0; i < 4; ++i) {
    //     for (int j = 0; j < 4; ++j) {

    //         int row = by * 256 + tz * 64 + i * 16 + tx / 4;
    //         int col = bx * 128 + ty * 64 + j * 16 + (tx % 4) * 2;

    //         (reinterpret_cast<uint32_t*>(&C[OFFSET(row, col, N)]))[0] =
    //             Accum[i][j * 2][0];

    //         (reinterpret_cast<uint32_t*>(&C[OFFSET(row + 8, col, N)]))[0] =
    //             Accum[i][j * 2][1];

    //         (reinterpret_cast<uint32_t*>(&C[OFFSET(row, col + 8, N)]))[0] =
    //             Accum[i][j * 2 + 1][0];

    //         (reinterpret_cast<uint32_t*>(&C[OFFSET(row + 8, col + 8,
    //         N)]))[0] =
    //             Accum[i][j * 2 + 1][1];
    //     }
    // }
}
}  // namespace playground