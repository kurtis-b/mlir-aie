<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Feed-Forward Network (FFN)

A Feed-Forward Network (FFN) is a fundamental building block in transformer architectures and other deep learning models. It consists of two linear transformations with an activation function applied in between. This design implements the FFN on the NPU, where the core computation involves General Matrix-Vector Multiplication (GEMV) kernels.

## Key Characteristics of FFN:

- **Layer Composition**: The FFN typically consists of two dense layers with a non-linear activation function (e.g., ReLU) applied after the first layer.
- **Parallelism**: The GEMV operations are distributed across multiple cores to maximize throughput and leverage the parallel processing capabilities of the NPU.
- **Efficiency**: By optimizing memory access patterns and computation, the design ensures high performance for matrix-vector operations, which are critical in transformer models.

This implementation uses `bfloat16` precision for GEMV operations, performed in parallel on multiple cores. The design is optimized for high throughput and low latency, making it suitable for real-time inference tasks.

## Source Files Overview

1. `npu_design.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and inst.txt for the NPU in Ryzen™ AI).

2. `kernel_ffn.cc`: A C++ implementation of the GEMV operation for AIE cores. This implementation leverages low-level intrinsics to perform matrix-vector multiplication efficiently. The source can be found [here](../../../aie_kernels/aie2/ffn.cc).

3. `host.cpp`: This C++ code is a testbench for the FFN design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the FFN design on the NPU. After execution, the script verifies the results and optionally outputs trace data.

## Usage

### C++ Testbench

To compile the design:

```shell
make
```

To run the design:

```shell
make run
```
