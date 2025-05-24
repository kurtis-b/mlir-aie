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

template <unsigned m, unsigned n>
void relu_i16(const int16 *__restrict pA, int16 *__restrict pC) {
  const int v_factor = 32;
  aie::vector<int16, v_factor> zeroes = aie::zeros<int16, v_factor>();

  event0();
  int16 *__restrict pC1 = pC;
  for (size_t i = 0; i < m; i++)
    chess_prepare_for_pipelining {
      const int16 *__restrict pA1 = pA + i * n;
      for (size_t j = 0; j < n; j += v_factor) {
        aie::vector<int16, v_factor> a_vec = aie::load_v<v_factor>(pA1);
        pA1 += v_factor;
        aie::vector<int16, v_factor> c_vec = aie::max(a_vec, zeroes);
        aie::store_v(pC1, c_vec);
        pC1 += v_factor;
      }
    }
  event1();
}

template <unsigned m, unsigned n>
void relu_bf16(bfloat16 *__restrict pA, bfloat16 *__restrict pC) {
  const int v_factor = 32;
  aie::vector<bfloat16, v_factor> zeroes = aie::zeros<bfloat16, v_factor>();

  event0();
  bfloat16 *__restrict pC1 = pC;
  for (size_t i = 0; i < m; i++)
    chess_prepare_for_pipelining {
      const bfloat16 *__restrict pA1 = pA + i * n;
      for (size_t j = 0; j < n; j += v_factor) {
        aie::vector<bfloat16, v_factor> a_vec = aie::load_v<v_factor>(pA1);
        pA1 += v_factor;
        aie::vector<bfloat16, v_factor> c_vec = aie::max(a_vec, zeroes);
        aie::store_v(pC1, c_vec);
        pC1 += v_factor;
      }
    }
  event1();
}

extern "C" {

void i16_relu(int16 *a_in, int16 *c_out) {
  relu_i16<DIM_M, DIM_N>(a_in, c_out);
}

void bf16_relu(bfloat16 *a_in, bfloat16 *c_out) {
  relu_bf16<DIM_M, DIM_N>(a_in, c_out);
}

} // extern "C"
