//===- matrix_multiplication.h ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the matrix multiplication
// host code, such as verifying and printing matrices.

#ifndef FULL_DESIGN_H
#define FULL_DESIGN_H

#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <ostream>
#include <stdfloat>

#include "test_utils.h"

namespace full_design_common {

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel_ffn1,k1", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())(
      "kernel_ffn2,k2", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())(
      "kernel_addnorm,k3", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())(
      "kernel_mha,k4", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())(
      "verbosity,v",
      "the verbosity of the output",
      cxxopts::value<int>()->default_value("0"))(
      "instr_ffn1,i1",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "instr_ffn2,i2",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "instr_addnorm,i3",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "instr_mha,i4",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "verify", "whether to verify the AIE computed output",
      cxxopts::value<int>()->default_value("1"))(
      "iters", "number of iterations",
      cxxopts::value<int>()->default_value("1"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("0"))(
      "trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))(
      "trace_file", "where to store trace output",
      cxxopts::value<std::string>()->default_value("trace.txt"))(
      "b_col_maj", "Is B matrix in colum-major format?",
      cxxopts::value<int>()->default_value("0"));
}

void parse_options(int argc, const char *argv[], cxxopts::Options &options,
                   cxxopts::ParseResult &result) {
  try {
    result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << "\n";
      std::exit(1);
    }

    // Check required options
    if (!result.count("xclbin") || !result.count("kernel_ffn1") || 
        !result.count("kernel_ffn2") || !result.count("kernel_addnorm") ||
        !result.count("kernel_mha") || !result.count("instr_ffn1") || 
        !result.count("instr_ffn2") || !result.count("instr_addnorm") || 
        !result.count("instr_mha")) {
      std::cerr << "Error: Required options missing\n\n";
      std::cerr << "Usage:\n" << options.help() << "\n";
      std::exit(1);
    }

  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << "\n";
    std::exit(1);
  }
}

template <typename T>
static inline T get_random();

template <>
std::bfloat16_t get_random<std::bfloat16_t>() {
  // Random numbers should NOT be uniformly between 0 and 1, because that
  // would make the matrix product AB always close to 1.
  return std::bfloat16_t((float)rand() / (float)(RAND_MAX));
}
} // namespace full_design_common

#endif
