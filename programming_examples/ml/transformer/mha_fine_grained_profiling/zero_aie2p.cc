//===- zero.cc --------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023-2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_CC
#define ZERO_CC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T, int M, int N>
void zero_scalar(T *__restrict c) {
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8); // 512 bit store units for AIE2P
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  event0();
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
  event1();
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifdef bf16_bf16_ONLY
#if defined(DIM_M) && defined(DIM_N)
#if (DIM_M == 64) && (DIM_N == 64)
#define combos(X) X(bfloat16, bf16, 64, 64)
#endif
#if (DIM_M == 16) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, 16, 256)
#endif
#if (DIM_M == 16) && (DIM_N == 16)
#define combos(X) X(bfloat16, bf16, 16, 16)
#endif
#if (DIM_M == 16) && (DIM_N == 128)
#define combos(X) X(bfloat16, bf16, 16, 128)
#endif
#if (DIM_M == 32) && (DIM_N == 64)
#define combos(X) X(bfloat16, bf16, 32, 64)
#endif
#if (DIM_M == 32) && (DIM_N == 256)
#define combos(X) X(bfloat16, bf16, 32, 256)
#endif
#if (DIM_M == 32) && (DIM_N == 16)
#define combos(X) X(bfloat16, bf16, 32, 16)
#endif
#if (DIM_M == 16) && (DIM_N == 1)
#define combos(X) X(bfloat16, bf16, 16, 1)
#endif
#if (DIM_M == 32) && (DIM_N == 1)
#define combos(X) X(bfloat16, bf16, 32, 1)
#endif
#endif
#endif

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#define zero_vectorized_c_func(ctype_out, mlir_type_out, m, n)                 \
  void zero_##mlir_type_out##_##m##_##n(ctype_out *c_out) {                    \
    zero_vectorized<ctype_out, m, n>(c_out);                                   \
  }

combos(zero_vectorized_c_func)

} // extern "C"

#endif