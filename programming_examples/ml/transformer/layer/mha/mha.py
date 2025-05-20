#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--seq_len", type=int, default=256)
    argparser.add_argument("--emb_dim", type=int, default=768)
    argparser.add_argument("--num_heads", type=int, default=12)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        maybe_taps = my_mha(
            args.seq_len,
            args.emb_dim,
            args.num_heads,
            args.b_col_maj,
            args.trace_size,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)

    if args.generate_taps:
        return maybe_taps


def ceildiv(a, b):
    return (a + b - 1) // b


def my_mha(
    seq_len: int,
    emb_dim: int,
    num_heads: int,
    b_col_maj: int,
    trace_size: int,
):
    head_dim = emb_dim // num_heads

    dtype_in = dtype_map["i8"]
    dtype_out = dtype_map["i32"]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    assert (
        emb_dim % num_heads == 0
    ), """Embedding dimension must be divisible by number of heads"""

    dev = AIEDevice.npu1_4col
    @device(dev)
    def device_body():
        # AIE Core Function declarations
        zero = external_func(f"zero_{dtype_out_str}", inputs=[out_l1_ty])
        matmul_vectorized_func_name = (
            f"matmul_{dtype_in_str}_{dtype_out_str}"
            if not b_col_maj
            else f"matmul_{dtype_in_str}_{dtype_out_str}_b_col_maj"
        )
        matmul = external_func(
            matmul_vectorized_func_name,
            inputs=[in_l1_ty, weight_l1_ty, out_l1_ty],
        )

        
        # Set up compute tiles
        @core(core_tiles[row][col], f"mha_mm_{m}x{k}x{n}.o") # TODO: MM kernel dimensions depend on tile
        def core_body():
            for _ in range_(0xFFFFFFFF):
                loop_out_tiles = (range_(n_out_tiles_per_core) if n_out_tiles_per_core > 1 else range(1))
                for _ in loop_out_tiles: 
                    print("TODO")

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[((M * K) + (2 * K * N),), np.dtype[dtype_in]], # Order: X, W_Q, W_K
            np.ndarray[(2 * K * N,), np.dtype[dtype_in]], # Order: W_V, W_O 
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            print("TODO")


if __name__ == "__main__":
    main()
