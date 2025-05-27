//===- passThrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#ifndef PASSTHROUGH_CC
#define PASSTHROUGH_CC

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

template <typename T, int N>
void passThrough_aie(T *restrict in, T *restrict out, const int32 height,
                     const int32 width) {
  event0();

  v64uint8 *restrict outPtr = (v64uint8 *)out;
  v64uint8 *restrict inPtr = (v64uint8 *)in;

  for (int j = 0; j < (height * width); j += N) // Nx samples per loop
    chess_prepare_for_pipelining chess_loop_range(6, ) { *outPtr++ = *inPtr++; }

  event1();
}

void passThroughLine(int8 *in, int8 *out, int32 lineWidth) {
  passThrough_aie<int8, 64>(in, out, 1, lineWidth);
}

void passThroughTile(int8 *in, int8 *out, int32 tileHeight, int32 tileWidth) {
  passThrough_aie<int8, 64>(in, out, tileHeight, tileWidth);
}

void passThroughLine(int16 *in, int16 *out, int32 lineWidth) {
  passThrough_aie<int16, 32>(in, out, 1, lineWidth);
}

void passThroughTile(int16 *in, int16 *out, int32 tileHeight, int32 tileWidth) {
  passThrough_aie<int16, 32>(in, out, tileHeight, tileWidth);
}

void passThroughLine(int32 *in, int32 *out, int32 lineWidth) {
  passThrough_aie<int32, 16>(in, out, 1, lineWidth);
}

void passThroughTile(int32 *in, int32 *out, int32 tileHeight, int32 tileWidth) {
  passThrough_aie<int32, 16>(in, out, tileHeight, tileWidth);
}

#endif // PASSTHROUGH_CC