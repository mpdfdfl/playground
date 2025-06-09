#pragma once
#include <mma.h>

#define WARP_SIZE 32
#ifndef OFFSET
    #define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif
#define LDST32BITS(pointer)                                                   \
    (*reinterpret_cast<half2*>(std::addressof(pointer)))
#define LDST64BITS(pointer)                                                   \
    (*reinterpret_cast<float2*>(std::addressof(pointer)))
#define LDST128BITS(pointer)                                                  \
    (*reinterpret_cast<float4*>(std::addressof(pointer)))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                          \
    asm volatile(                                                             \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),    \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                          \
    asm volatile(                                                             \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),    \
        "l"(src), "n"(bytes))

// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP()                                          \
    asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n)                                           \
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes)                                        \
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::128B "      \
                 "[%0], [%1], %2;\n" ::"r"(dst),                              \
                 "l"(src), "n"(bytes))

// ldmatrix
#define LDMATRIX_X1(R, addr)                                                  \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"     \
                 : "=r"(R)                                                    \
                 : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                     \
    asm volatile(                                                             \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"  \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                              \
        : "r"(addr))
#define LDMATRIX_X1_T(R, addr)                                                \
    asm volatile(                                                             \
        "ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"        \
        : "=r"(R)                                                             \
        : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                           \
    asm volatile(                                                             \
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
        : "=r"(R0), "=r"(R1)                                                  \
        : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                   \
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, "   \
                 "%2, %3}, [%4];\n"                                           \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                     \
                 : "r"(addr))

// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R)                                                  \
    asm volatile(                                                             \
        "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), \
        "r"(R))
#define STMATRIX_X2(addr, R0, R1)                                             \
    asm volatile(                                                             \
        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(   \
            addr),                                                            \
        "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3)                                     \
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, "   \
                 "%3, %4};\n" ::"r"(addr),                                    \
                 "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R)                                                \
    asm volatile(                                                             \
        "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"( \
            addr),                                                            \
        "r"(R))
#define STMATRIX_X2_T(addr, R0, R1)                                           \
    asm volatile(                                                             \
        "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" :: \
            "r"(addr),                                                        \
        "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3)                                   \
    asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, " \
                 "%2, %3, %4};\n" ::"r"(addr),                                \
                 "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
// 矩阵乘累加宏（MMA，Tensor Core 计算）
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)           \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, "    \
                 "%1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"               \
                 : "=r"(RD0), "=r"(RD1)                                       \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0),          \
                   "r"(RB1), "r"(RC0), "r"(RC1))

__device__ __host__ inline int div_ceil(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}