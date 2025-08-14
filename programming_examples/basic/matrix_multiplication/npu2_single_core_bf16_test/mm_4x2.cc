//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in, typename T_out, int rowA, int colA, int colB>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      c[row * colB + col] += running_sum;
    }
  }
  event1();
}

/* Similar to the kernel above, but we expand matrix A (in 'm' dimension, or
 * rowA) 4 times, while matrix B is expanded 2 times (in 'n' dimension, or
 * ColB). This is very helpful in attaining high kernel efficiency for some
 * precisions.
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true>
static inline void matmul_vectorized_4x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  // AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (unsigned z = 0; z < rowA; z += 4) {
    T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
    T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
    T_out *__restrict pC3 = pC + ((z + 2) * colB + 0) * MMUL::size_C;
    T_out *__restrict pC4 = pC + ((z + 3) * colB + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;

        const T_in *__restrict pB1;
        const T_in *__restrict pB2;
        if constexpr (b_row_maj) {
          pB1 = pB + (j)*MMUL::size_B;
          pB2 = pB + (j + 1) * MMUL::size_B;
        } else {
          pB1 = pB + (j * colA) * MMUL::size_B;
          pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
        }

        aie::vector<T_in, MMUL::size_A> A01;
        aie::vector<T_in, MMUL::size_A> A11;
        aie::vector<T_in, MMUL::size_A> A21;
        aie::vector<T_in, MMUL::size_A> A31;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C20 =
            aie::load_v<MMUL::size_C>(pC3);
        aie::vector<T_out, MMUL::size_C> acc_C21 =
            aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C30 =
            aie::load_v<MMUL::size_C>(pC4);
        aie::vector<T_out, MMUL::size_C> acc_C31 =
            aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C20(acc_C20);
        MMUL C21(acc_C21);
        MMUL C30(acc_C30);
        MMUL C31(acc_C31);

        for (unsigned i = 0; i < colA; i += 1)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A01 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A11 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            A21 = aie::load_v<MMUL::size_A>(pA3);
            pA3 += MMUL::size_A;
            A31 = aie::load_v<MMUL::size_A>(pA4);
            pA4 += MMUL::size_A;
            if constexpr (b_row_maj) {
              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += (MMUL::size_B * colB);
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += (MMUL::size_B * colB);
            } else {
              B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
              pB1 += MMUL::size_B;
              B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
              pB2 += MMUL::size_B;
            }

            C00.mac(A01, B0);
            C01.mac(A01, B1);
            C10.mac(A11, B0);
            C11.mac(A11, B1);
            C20.mac(A21, B0);
            C21.mac(A21, B1);
            C30.mac(A31, B0);
            C31.mac(A31, B1);
          }

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC3, C20.template to_vector<T_out>());
        pC3 += MMUL::size_C;
        aie::store_v(pC3, C21.template to_vector<T_out>());
        pC3 += MMUL::size_C;
        aie::store_v(pC4, C30.template to_vector<T_out>());
        pC4 += MMUL::size_C;
        aie::store_v(pC4, C31.template to_vector<T_out>());
        pC4 += MMUL::size_C;
      }
  }

  event1();
}

#ifdef B_COL_MAJ
constexpr bool is_b_row_maj = false;
#else
constexpr bool is_b_row_maj = true;
#endif

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t, is_b_row_maj>(pA, pB, pC);
}

// Note that this shape is only possible for bf16 when using bfp16 emulation
// during matmuls.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_4x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t, is_b_row_maj>(pA, pB, pC);
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

// The emulation of bf16 changes the available shapes for matrix multiplication
#ifdef bf16_bf16_ONLY
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 8, 8, 8)
#else
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 8)
#endif
#endif

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void matmul_##mlir_type_in##_##mlir_type_out(ctype_in *a_in, ctype_in *b_in, \
                                               ctype_out *c_out) {             \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);                               \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(a_in, b_in,        \
                                                            c_out);            \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out) {                                \
    zero_vectorized<ctype_out, DIM_M, DIM_N>(c_out);                           \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t)                                            \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_M, DIM_N>(c_out);                               \
  }

combos(matmul_vectorized_c_func) combos(matmul_scalar_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"