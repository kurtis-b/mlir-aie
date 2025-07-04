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

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 2x2 expansion in output matrix C (see C00, C01, C10, C11
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
          const T_in *__restrict pB1;
          const T_in *__restrict pB2;
          if constexpr (b_row_maj) {
            pB1 = pB + (j)*MMUL::size_B;
            pB2 = pB + (j + 1) * MMUL::size_B;
          } else {
            pB1 = pB + (j * colA) * MMUL::size_B;
            pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
          }
          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0;
          aie::vector<T_in, MMUL::size_B> B1;
          if constexpr (b_row_maj) {
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;
          } else {
            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
            pB1 += MMUL::size_B;
            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
            pB2 += MMUL::size_B;
          }

          // Load partial results from C buffer for accumulation in-place. The
          // zero.cc function handles the zeroing of data when a new
          // accumulation is needed (after the 'K' reduction dimension)
          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          for (unsigned i = 1; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              if constexpr (b_row_maj) {
                B0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B * colB;
                B1 = aie::load_v<MMUL::size_B>(pB2);
                pB2 += MMUL::size_B * colB;
              } else {
                B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                pB1 += MMUL::size_B;
                B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                pB2 += MMUL::size_B;
              }

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          // TODO make shift right here to keep most significat bits
          // when lowering the output
          // example below shows how to shift right 10 bits
          // #define SHIFT 10
          // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));
          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
        }
    }

  event1();
}

#ifdef B_COL_MAJ
constexpr bool is_b_row_maj = false;
#else
constexpr bool is_b_row_maj = true;
#endif

// The following kernel definitions use mmul shapes that have been found to be
// optimal for AIE2P in combination with the 2x2 mmul expanded kernel.
//
// All available matrix multiplication shapes in the AIE-API can be found here:
// https://xilinx.github.io/aie_api/group__group__mmul.html
//
// They are all defined based on the shape of the mmul, the input data format
// and the output data format.
//
// Additionally, they check for the correct
// divisibility of the tile dimensions. Note that while both the 'm' and 'n'
// dimensions of the mmul are expanded, the 'k' dimension is not.

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x8_i16_i16(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<int16, int16, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x8_i16_i32(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int32 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<int16, int32, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj>(pA, pB, pC);
}

//TODO: Unlike the others, this shape has not yet been tested for optimality
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;

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

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s),
                                    (n / t), r, s, t, is_b_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_f32(const bfloat16 *__restrict pA,
                                 const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s), (n / t),
                                    r, s, t, is_b_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_f32(const bfloat16 *__restrict pA,
                                 const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s), (n / t),
                                    r, s, t, is_b_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i8(const int8 *__restrict pA,
                                                 const int8 *__restrict pB,
                                                 int8 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<int8, int8, (m / r), (k / s), (n / t), r, s,
                                    t, is_b_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i16(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int16 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<int8, int16, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i32(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int32 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<int8, int32, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj>(pA, pB, pC);
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

#ifdef bf16_bf16_ONLY
#if defined(DIM_M) && defined(DIM_K) && defined(DIM_N)
#if B_COL_MAJ
#if (DIM_M == 64) && (DIM_K == 64) && (DIM_N == 64)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 64, 64, 64, 0)
#endif
#if (DIM_M == 16) && (DIM_K == 32) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 32, 256, 0)
#endif
#if (DIM_M == 16) && (DIM_K == 256) && (DIM_N == 16)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 256, 16, 0)
#endif
#if (DIM_M == 16) && (DIM_K == 16) && (DIM_N == 128)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 16, 128, 0)
#endif
// Added workloads
#if (DIM_M == 32) && (DIM_K == 128) && (DIM_N == 64)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 128, 64, 0)
#endif
#if (DIM_M == 32) && (DIM_K == 32) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 32, 256, 0)
#endif
#if (DIM_M == 32) && (DIM_K == 256) && (DIM_N == 16)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 256, 16, 0)
#endif
#if (DIM_M == 32) && (DIM_K == 16) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 16, 256, 0)
#endif
#if (DIM_M == 16) && (DIM_K == 16) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 16, 256, 0)
#endif
#else
#if (DIM_M == 64) && (DIM_K == 64) && (DIM_N == 64)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 64, 64, 64, 1)
#endif
#if (DIM_M == 16) && (DIM_K == 32) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 32, 256, 1)
#endif
#if (DIM_M == 16) && (DIM_K == 256) && (DIM_N == 16)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 256, 16, 1)
#endif
#if (DIM_M == 16) && (DIM_K == 16) && (DIM_N == 128)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 16, 128, 1)
#endif
// Added workloads
#if (DIM_M == 32) && (DIM_K == 128) && (DIM_N == 64)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 128, 64, 1)
#endif
#if (DIM_M == 32) && (DIM_K == 32) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 32, 256, 1)
#endif
#if (DIM_M == 32) && (DIM_K == 256) && (DIM_N == 16)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 256, 16, 1)
#endif
#if (DIM_M == 32) && (DIM_K == 16) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 32, 16, 256, 1)
#endif
#if (DIM_M == 16) && (DIM_K == 16) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4, 16, 16, 256, 1)
#endif
#endif
#endif
#endif

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t, m, k, n, rowmaj)      \
  void matmul_##mlir_type_in##_##mlir_type_out##_##m##_##k##_##n##_##rowmaj(   \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<m, k, \
                                                                         n>(   \
        a_in, b_in, c_out);                                                    \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,     \
                             r, s, t, m, k, n, rowmaj)                             \
  void                                                                             \
      matmul_scalar_##mlir_type_in##_##mlir_type_out##_##m##_##k##_##n##_##rowmaj( \
          ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar<ctype_in, ctype_out, m, k, n>(a_in, b_in, c_out);                \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t, m, k, n, rowmaj)        \
  void zero_##mlir_type_out##_##m##_##k##_##n##_##rowmaj(ctype_out *c_out) {   \
    zero_vectorized<ctype_out, m, n>(c_out);                                   \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t, m, k, n, rowmaj)                           \
  void zero_scalar_##mlir_type_out##_##m##_##k##_##n##_##rowmaj(               \
      ctype_out *c_out) {                                                      \
    zero_scalar<ctype_out, m, n>(c_out);                                       \
  }

// combos(matmul_vectorized_c_func) combos(matmul_scalar_c_func)
// combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

combos(matmul_vectorized_c_func) combos(zero_vectorized_c_func)

} // extern "C"
