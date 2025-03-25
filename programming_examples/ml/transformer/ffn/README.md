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

A Feed-Forward Network (FFN) is a fundamental building block in deep learning models. It consists of one or more linear transformations with an activation function applied in between. This design implements the transformer FFN on the NPU, where the core computation involves General Matrix-Vector Multiplication (GEMV) kernels.

## Key Characteristics of FFN:

- **Layer Composition**: The FFN typically consists of two dense layers with a non-linear activation function (e.g., ReLU) applied after the first layer.
- **Parallelism**: The GEMV operations are distributed across multiple cores in one column to maximize throughput and leverage the parallel processing capabilities of the NPU (TODO).
- **Efficiency**: By optimizing memory access patterns and computation, the design ensures high performance for matrix-vector operations, which are critical in transformer models.

This implementation uses `bfloat16` precision for GEMV operations, performed in parallel on multiple cores. The design is optimized for high throughput and low latency, making it suitable for real-time inference tasks.

## Notes on Design Inspiration

The design of this Feed-Forward Network (FFN) implementation draws from two pull requests from GitHub:

- [An example for runtime increasing loop trip count with RTP](https://github.com/Xilinx/mlir-aie/pull/2122): This PR implemented RTPs with SCF to use dynamic loops in the core's execution, and was able to do it with objectFIFOs. However, there was no host-side implementation showing how the dynamic looping could be set on the host-side.
- [add example with run-time-parametrized matmul](https://github.com/Xilinx/mlir-aie/pull/1772): This PR implemented RTPs to synchronously block the core's from executing until the host sets one of the RTPs to a "ready" signal. From there, other RTP's are used to generate the loops dynamically for the core's execution. This PR was written at the buffer descriptor level, and includes a host-side implementation. On the host side, the instruction buffer is re-written with instructions compiled for the corresponding workload. By only needing to re-write the instruction buffer, this removes the need for the host to set up the NPU more than once, i.e. only one xclbin needs to be laoded, even though that xclbin was compiled for a specific workload.

The implementation here is a combination of the two PRs. SCF while loops are used with RTPs to do dynamic looping in the core tiles, and for the host side, the instruction buffer is re-written with the instructions corresponding to the workload to be run.

## Source Files Overview

1. `npu_design.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (i.e., XCLBIN and insts.txt for the NPU in Ryzen™ AI).

2. `kernel_ffn.cc`: A C++ implementation of the GEMV operation for AIE cores. This implementation leverages low-level intrinsics to perform matrix-vector multiplication efficiently. The source can be found [here](./kernel_ffn.cc).

3. `host.cpp`: This C++ code is a testbench for the FFN design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the FFN design on the NPU. After execution, the script verifies the results and optionally outputs trace data (TODO).

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
