#pragma once
#include "../../cuda/cuda_common.h"
#include "playground/matmul.hpp"
#include <cstdio>

namespace playground
{
__global__ void gemm_v3(const size_t M, const size_t N, const size_t K,
                        const float32_t* const __restrict__ A,
                        const float32_t* const __restrict__ B,
                        float32_t* const __restrict__ C)
{
    const int32_t BM = 128;
    const int32_t BN = 128;
    const int32_t BK = 8;
    const int32_t TM = 16;
    const int32_t TN = 4;

    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;
    const int32_t tx = threadIdx.x;
    const int32_t ty = threadIdx.y;
    const int32_t tid = ty * blockDim.x + tx;

    __shared__ float32_t s_a[BM][BK];
    __shared__ float32_t s_b[BK][BN];

    float32_t r_c[TM][TN] = {0.0f};

    int32_t load_a_smem_m = tid >> 1;
    int32_t load_a_smem_k = (tid & 1) << 2;
    int32_t load_b_smem_k = tid >> 5;
    int32_t load_b_smem_n = (tid & 31) << 2;

    int32_t load_a_gmem_m = by * BM + load_a_smem_m;
    int32_t load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int32_t bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int32_t load_a_gmem_k = bk * BK + load_a_smem_k;
        int32_t load_b_gmem_k = bk * BK + load_b_smem_k;
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) =
            FLOAT4_CONST(A[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) =
            FLOAT4_CONST(B[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]);
        __syncthreads();

        for (int32_t k = 0; k < BK; k++) {
            for (int32_t m = 0; m < TM; m++) {
                int row = ty * TM + m;
                for (int32_t n = 0; n < TN; n++) {
                    int col = tx * TN + n;
                    r_c[m][n] += s_a[row][k] * s_b[k][col];
                }
            }
        }
        __syncthreads();
    }
    for (int32_t i = 0; i < TM; i++) {
        for (int32_t j = 0; j < TN; j += 4) {
            int32_t row = by * BM + ty * TM + i;
            int32_t col = bx * BN + tx * TN + j;
            FLOAT4(C[OFFSET(row, col, N)]) = FLOAT4(r_c[i][j]);
        }
    }
}
}  // namespace playground
