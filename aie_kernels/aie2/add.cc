//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int M, const int N>
void eltwise_add(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      c[i * N + j] = (T_out)(a[i * N + j] + b[i * N + j]);
    }
  }
}

template <typename T_in, typename T_out, const int M, const int N>
void eltwise_vadd(T_in *a, T_in *b, T_out *c) {
  constexpr int vec_factor = 16;
  event0();
  const int F = N / vec_factor;
  for (int i = 0; i < M; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      T_in *__restrict pA1 = a + i * N;
      T_in *__restrict pB1 = b + i * N;
      T_out *__restrict pC1 = c + i * N;
      for (int j = 0; j < F; j++)
        chess_prepare_for_pipelining chess_loop_range(16, ) {
          aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
          pA1 += vec_factor;
          aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
          pB1 += vec_factor;
          aie::vector<T_out, vec_factor> cout =
              aie::vector_cast<T_out>(aie::add(A0, B0));
          aie::store_v(pC1, cout);
          pC1 += vec_factor;
        }
    }
  event1();
}

extern "C" {

#ifndef DIM_M
#define DIM_M 8
#endif

#ifndef DIM_N
#define DIM_N 512
#endif

// void eltwise_add_i16_scalar(int16 *a_in, int16 *b_in, int16 *c_out) {
//   eltwise_add<int16, int16, DIM_M, DIM_N>(a_in, b_in, c_out);
// }

void eltwise_add_i16_vector(int16 *a_in, int16 *b_in, int16 *c_out) {
  eltwise_vadd<int16, int16, DIM_M, DIM_N>(a_in, b_in, c_out);
}

// void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16
// *c_out)
// {
//   eltwise_add<bfloat16, bfloat16, DIM_M, DIM_N>(a_in, b_in, c_out);
// }

// void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16
// *c_out)
// {
//   eltwise_vadd<bfloat16, bfloat16, DIM_M, DIM_N>(a_in, b_in, c_out);
// }

} // extern "C"
