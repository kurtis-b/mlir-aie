<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>General Matrix Multiplication (GEMM) </ins>

## Introduction

General Matrix Multiplication (GEMM) is a fundamental operation in various machine learning and scientific computing tasks. This README provides instructions for running GEMM on 16 AI Engine (AIE) cores with 8-bit precision.

## Source Files Overview

```
.
+-- mlp_design.py        # A Python script that defines the AIE array structural design using MLIR-AIE operations using a lower-level version of IRON.
+-- Makefile             # Contains instructions for building and compiling software projects.
+-- README.md            # This file.
+-- test.py              # Python code testbench for the design example.
```

## Compilation

To compile the design:

```shell
make
```

To run the design:

```shell
make run_py
```

## Performance Comparison

The AIE design's performance depends on the kernel utilization. For example, a kernel size of 64x128x64 performs worse than a kernel size of 64x184x64. To achieve performance comparable to a quantized MLP run on the CPU with a batch size of 512, higher kernel utilization is important.
