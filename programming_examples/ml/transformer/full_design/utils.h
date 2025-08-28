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
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace full_design_common {
  // Thread-safe queue that can take in 1 or 2 inputs (as std::vector<T> or std::pair<std::vector<T1>, std::vector<T2>>)
  template <typename T1, typename T2 = void>
  class ThreadSafeQueue {
  public:
    using SenderId = std::thread::id;
    using ValueType = std::pair<std::vector<T1>, std::vector<T2>>;

    struct Item {
      ValueType value;
      SenderId sender;
      Item(ValueType v, SenderId s) : value(std::move(v)), sender(s) {}
    };

    ThreadSafeQueue() = default;

    // Push for two inputs
    void push(std::vector<T1> v1, std::vector<T2> v2, SenderId sender_id) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.emplace(ValueType(std::move(v1), std::move(v2)), sender_id);
      }
      cv_.notify_one();
    }

    // Pop both value and sender id
    bool pop(std::vector<T1>& out1, std::vector<T2>& out2, SenderId& sender) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !queue_.empty() || done_; });
      if (queue_.empty())
        return false;
      out1 = std::move(queue_.front().value.first);
      out2 = std::move(queue_.front().value.second);
      sender = queue_.front().sender;
      queue_.pop();
      return true;
    }

    // Overload for backward compatibility (ignores sender)
    bool pop(std::vector<T1>& out1, std::vector<T2>& out2) {
      SenderId dummy;
      return pop(out1, out2, dummy);
    }

    void setDone() {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
      }
      cv_.notify_all();
    }

    bool empty() {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }

    SenderId getCurrentThreadId() const {
      return std::this_thread::get_id();
    }
  private:
    std::queue<Item> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
  };

  // Specialization for single input
  template <typename T1>
  class ThreadSafeQueue<T1, void> {
  public:
    using SenderId = std::thread::id;
    using ValueType = std::vector<T1>;

    struct Item {
      ValueType value;
      SenderId sender;
      Item(ValueType v, SenderId s) : value(std::move(v)), sender(s) {}
    };

    ThreadSafeQueue() = default;

    void push(std::vector<T1> value, SenderId sender_id) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.emplace(std::move(value), sender_id);
      }
      cv_.notify_one();
    }

    // Pop both value and sender id
    bool pop(std::vector<T1>& result, SenderId& sender) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !queue_.empty() || done_; });
      if (queue_.empty())
        return false;
      result = std::move(queue_.front().value);
      sender = queue_.front().sender;
      queue_.pop();
      return true;
    }

    // Overload for backward compatibility (ignores sender)
    bool pop(std::vector<T1>& result) {
      SenderId dummy;
      return pop(result, dummy);
    }

    void setDone() {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true;
      }
      cv_.notify_all();
    }

    bool empty() {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }

    SenderId getCurrentThreadId() const {
      return std::this_thread::get_id();
    }
  private:
    std::queue<Item> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
  };

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
