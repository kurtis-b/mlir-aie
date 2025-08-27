//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "utils.h"
#include "../ffn-1/utils.h"
#include "../ffn-2/utils.h"
#include "../mha/utils.h"
#include "../add_and_norm/utils.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 256
#endif
#ifndef EMB_DIM
#define EMB_DIM 768
#endif
#ifndef FFN_DIM
#define FFN_DIM 3072
#endif
#ifndef ACT_VOLUME
#define ACT_VOLUME (SEQ_LEN * EMB_DIM)
#endif
#ifndef FFN1_WEIGHT_VOLUME
#define FFN1_WEIGHT_VOLUME (EMB_DIM * FFN_DIM)
#endif
#ifndef FFN1_OUT_VOLUME
#define FFN1_OUT_VOLUME (SEQ_LEN * FFN_DIM)
#endif
#ifndef FFN2_WEIGHT_VOLUME
#define FFN2_WEIGHT_VOLUME (EMB_DIM * FFN_DIM)
#endif
#ifndef FFN2_OUT_VOLUME
#define FFN2_OUT_VOLUME (SEQ_LEN * EMB_DIM)
#endif
#ifndef MHA_WEIGHT_VOLUME
#define MHA_WEIGHT_VOLUME (SEQ_LEN * EMB_DIM)
#endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;
#endif

int ffn1(xrt::device &device, xrt::xclbin &xclbin, xrt::hw_context &context,
         std::vector<A_DATATYPE> &actVec, std::vector<C_DATATYPE> &outVec,
         const std::string& instr_file, const std::string& kernel_name) {
  size_t ACT_SIZE = (ACT_VOLUME * sizeof(A_DATATYPE));
  size_t WEIGHT_SIZE = (FFN1_WEIGHT_VOLUME * sizeof(B_DATATYPE));
  size_t OUT_SIZE = (FFN1_OUT_VOLUME * sizeof(C_DATATYPE));

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(instr_file);
  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  int verbosity = 0;
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [kernel_name, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(kernel_name, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();
  // get a kernel handle
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, ACT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, WEIGHT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  // Workaround so we declare a really small trace buffer when one is not used
  auto bo_trace = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  memcpy(bufA, actVec.data(), (actVec.size() * sizeof(A_DATATYPE)));

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> weightVec(FFN1_WEIGHT_VOLUME);
  for (int i = 0; i < FFN1_WEIGHT_VOLUME; i++) {
    weightVec[i] = ffn_1_common::get_random<B_DATATYPE>();
  }
  memcpy(bufB, weightVec.data(), (weightVec.size() * sizeof(B_DATATYPE)));

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  outVec = std::vector<C_DATATYPE>(FFN1_OUT_VOLUME);
  memset(bufOut, 0, OUT_SIZE);
  char *bufTrace = bo_trace.map<char *>();

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out, bo_tmp1, bo_trace);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(outVec.data(), bufOut, (outVec.size() * sizeof(C_DATATYPE)));

  return 0;
}

int ffn2(xrt::device &device, xrt::xclbin &xclbin, xrt::hw_context &context,
         std::vector<A_DATATYPE> &actVec, std::vector<C_DATATYPE> &outVec,
         const std::string& instr_file, const std::string& kernel_name) {
  size_t ACT_SIZE = (FFN1_OUT_VOLUME * sizeof(A_DATATYPE));
  size_t WEIGHT_SIZE = (FFN2_WEIGHT_VOLUME * sizeof(B_DATATYPE));
  size_t OUT_SIZE = (FFN2_OUT_VOLUME * sizeof(C_DATATYPE));

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(instr_file);
  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  int verbosity = 0;
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [kernel_name, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(kernel_name, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();
  // get a kernel handle
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, ACT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, WEIGHT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  // Workaround so we declare a really small trace buffer when one is not used
  auto bo_trace = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  memcpy(bufA, actVec.data(), (actVec.size() * sizeof(A_DATATYPE)));

  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> weightVec(FFN2_WEIGHT_VOLUME);
  for (int i = 0; i < FFN2_WEIGHT_VOLUME; i++) {
    weightVec[i] = ffn_1_common::get_random<B_DATATYPE>();
  }
  memcpy(bufB, weightVec.data(), (weightVec.size() * sizeof(B_DATATYPE)));

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  outVec = std::vector<C_DATATYPE>(FFN2_OUT_VOLUME);
  memset(bufOut, 0, OUT_SIZE);
  char *bufTrace = bo_trace.map<char *>();

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out, bo_tmp1, bo_trace);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(outVec.data(), bufOut, (outVec.size() * sizeof(C_DATATYPE)));

  return 0;
}

int addandnorm(xrt::device &device, xrt::xclbin &xclbin, xrt::hw_context &context,
         std::vector<A_DATATYPE> &actVec, std::vector<B_DATATYPE> &skipVec,
         std::vector<C_DATATYPE> &outVec, const std::string& instr_file,
         const std::string& kernel_name) {
  size_t ACT_SIZE = (ACT_VOLUME * sizeof(A_DATATYPE));
  size_t SKIP_SIZE = (ACT_VOLUME * sizeof(B_DATATYPE));
  size_t OUT_SIZE = (ACT_VOLUME * sizeof(C_DATATYPE));

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(instr_file);
  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  int verbosity = 0;
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [kernel_name, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(kernel_name, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();
  // get a kernel handle
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, ACT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, ACT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out = xrt::bo(device, ACT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  // Workaround so we declare a really small trace buffer when one is not used
  auto bo_trace = xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  memcpy(bufA, actVec.data(), (actVec.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  memcpy(bufB, skipVec.data(), (skipVec.size() * sizeof(B_DATATYPE)));

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  outVec = std::vector<C_DATATYPE>(ACT_VOLUME);
  memset(bufOut, 0, OUT_SIZE);
  char *bufTrace = bo_trace.map<char *>();

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out, bo_tmp1, bo_trace);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  memcpy(outVec.data(), bufOut, (outVec.size() * sizeof(C_DATATYPE)));

  return 0;
}

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  full_design_common::add_default_options(options);
  full_design_common::parse_options(argc, argv, options, vm);

  // Start the XRT test code, get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

  // Load the xclbin
  xrt::xclbin xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  std::vector<A_DATATYPE> act1Vec = std::vector<A_DATATYPE>(SEQ_LEN * EMB_DIM);
  for (int i = 0; i < ACT_VOLUME; i++) {
    act1Vec[i] = full_design_common::get_random<A_DATATYPE>();
  }
  std::vector<A_DATATYPE> act2Vec = std::vector<A_DATATYPE>(SEQ_LEN * EMB_DIM);
  std::vector<C_DATATYPE> ffn1Vec = std::vector<C_DATATYPE>(SEQ_LEN * FFN_DIM);

  int fail = ffn1(device, xclbin, context, act1Vec, ffn1Vec,
                  vm["instr_ffn1"].as<std::string>(), vm["kernel_ffn1"].as<std::string>());
  std::cout << "\nFFN1 Output:\n";
  for (int i = 0; i < 10; i++) {
    std::cout << ffn1Vec[i] << ",";
  }
  fail |= ffn2(device, xclbin, context, ffn1Vec, act2Vec,
                  vm["instr_ffn2"].as<std::string>(), vm["kernel_ffn2"].as<std::string>());
  std::cout << "\nFFN2 Output:\n";
  for (int i = 0; i < 10; i++) {
    std::cout << act2Vec[i] << ",";
  }
  // First input should be output from FFN-2 and second input should be skip connection input
  fail |= addandnorm(device, xclbin, context, act2Vec, act1Vec, act1Vec,
                  vm["instr_addnorm"].as<std::string>(), vm["kernel_addnorm"].as<std::string>());
  std::cout << "\nAdd & Norm Output:\n";
  for (int i = 0; i < 10; i++) {
    std::cout << act1Vec[i] << ",";
  }
  std::cout << "\n";
  return fail;
}
