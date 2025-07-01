#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 4),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-H", type=int, default=12)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        maybe_taps = my_mha(
            args.dev,
            args.M,
            args.K,
            args.N,
            args.H,
            args.n_aie_cols,
            args.dtype_in,
            args.dtype_out,
            args.b_col_maj,
            args.emulate_bf16_mmul_with_bfp16,
            args.trace_size,
            args.generate_taps,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)

    if args.generate_taps:
        return maybe_taps


def ceildiv(a, b):
    return (a + b - 1) // b


def my_mha(
    dev,
    M,
    K,
    N,
    H,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    generate_taps=False,
):
    if dev == "npu":
        n_aie_rows_projs = 1
        n_aie_cols_projs = 1
    else:
        n_aie_rows_projs = 2
        n_aie_cols_projs = 1
    head_dim = N // H

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    q_matmul_dims = (32, 128, 64)
    kv_matmul_dims = (32, 128, 64)
    o1_matmul_dims = [(16, 32), (256, 32)]
    o2_softmax_dims = (16, 256)
    o3_matmul_dims = (o2_softmax_dims[0], o2_softmax_dims[1], 16)
    o4_matmul_dims = (o3_matmul_dims[0], o3_matmul_dims[2], 256)
    matmul_dims = [q_matmul_dims, kv_matmul_dims, o3_matmul_dims]
    
    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    # npu is a 4 row x 4 col array
    if dev == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    # npu2 is a 4 row x 8 col array
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    for dims in matmul_dims:
        assert (
            M % (dims[0] * n_aie_rows_projs) == 0
        ), """A must be tileable into (m * n_aie_rows_projs, dims[1])-sized blocks"""
    assert (
        M % (o1_matmul_dims[0][0]) == 0
    ), """A must be tileable into (m, dims[1])-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    for dims in matmul_dims:
        assert K % dims[1] == 0
    assert K % o1_matmul_dims[0][1] == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    for dims in matmul_dims:
        assert (
            N % (dims[2] * n_aie_cols_projs) == 0
        ), """B must be tileable into (k, dims[2] * n_aie_cols_projs)-sized blocks"""
    assert (
        N % (o1_matmul_dims[1][0]) == 0
    ), """B must be tileable into (k, dims[2])-sized blocks"""

    assert (N % H == 0), """Embedding dimension must be divisible by number of heads"""
    assert (head_dim % o1_matmul_dims[0][1] == 0), """Head dimension must be divisible by left matrix col dim"""
    assert (head_dim % o1_matmul_dims[1][1] == 0), """Head dimension must be divisible by right matrix col dim"""
    assert (head_dim % o3_matmul_dims[2] == 0), """Head dimension must be divisible by right matrix col dim"""
    assert (M / o2_softmax_dims[1] == 1), """Softmax tile col size must be the whole sequence length"""
    assert (o1_matmul_dims[0][0] == o2_softmax_dims[0]), """Softmax tile row size must be the same as the attn score"""
    assert (o1_matmul_dims[1][0] == o2_softmax_dims[1]), """Softmax tile col size must be the same as the attn score"""

    for dims in matmul_dims:
        assert (dims[0] * dims[1] * dims[2] > 0), f"Matmul dimensions {dims} must be non-zero"
        assert (dims[0] % r == 0), f"Matmul dimension {dims[0]} must be divisible by {r}"
        assert (dims[1] % s == 0), f"Matmul dimension {dims[1]} must be divisible by {s}"
        assert (dims[2] % t == 0), f"Matmul dimension {dims[2]} must be divisible by {t}"
    assert (o1_matmul_dims[0][0] * o1_matmul_dims[0][1] * o1_matmul_dims[1][0] > 0), f"Matmul dimensions {o1_matmul_dims} must be non-zero"
    assert (o1_matmul_dims[0][0] % r == 0), f"Matmul dimension {o1_matmul_dims[0][0]} must be divisible by {r}"
    assert (o1_matmul_dims[0][1] % s == 0), f"Matmul dimension {o1_matmul_dims[0][1]} must be divisible by {s}"
    assert (o1_matmul_dims[1][1] % s == 0), f"Matmul dimension {o1_matmul_dims[1][1]} must be divisible by {s}"
    assert (o1_matmul_dims[1][0] % t == 0), f"Matmul dimension {o1_matmul_dims[1][0]} must be divisible by {t}"

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    if dev == "npu":
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu1_4col
    else:
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        X_l2_ty = np.ndarray[(q_matmul_dims[0] * q_matmul_dims[1],), np.dtype[dtype_in]]
        X_l1_ty = np.ndarray[(q_matmul_dims[0] * q_matmul_dims[1],), np.dtype[dtype_in]]

        Wq_l2_ty = np.ndarray[(q_matmul_dims[1] * q_matmul_dims[2] * n_aie_rows_projs,), np.dtype[dtype_in]]
        q_l2_ty = np.ndarray[(q_matmul_dims[0] * q_matmul_dims[2] * n_aie_rows_projs,), np.dtype[dtype_out]]
        Wq_l1_ty = np.ndarray[(q_matmul_dims[1] * q_matmul_dims[2],), np.dtype[dtype_in]]
        q_l1_ty = np.ndarray[(q_matmul_dims[0] * q_matmul_dims[2],), np.dtype[dtype_out]]

        Wkv_l2_ty = np.ndarray[(kv_matmul_dims[1] * kv_matmul_dims[2] * n_aie_rows_projs,), np.dtype[dtype_in]]
        kv_l2_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[2] * n_aie_rows_projs,), np.dtype[dtype_out]]
        Wkv_l1_ty = np.ndarray[(kv_matmul_dims[1] * kv_matmul_dims[2],), np.dtype[dtype_in]]
        kv_l1_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[2],), np.dtype[dtype_out]]

        q_l1_ty_in = np.ndarray[(o1_matmul_dims[0][0] * o1_matmul_dims[0][1],), np.dtype[dtype_out]]
        k_l1_ty_in = np.ndarray[(o1_matmul_dims[1][0] * o1_matmul_dims[1][1],), np.dtype[dtype_out]]
        v_l1_ty_in = np.ndarray[(o3_matmul_dims[1] * o3_matmul_dims[2],), np.dtype[dtype_out]]
        Wo_l1_ty = np.ndarray[(o4_matmul_dims[1] * o4_matmul_dims[2],), np.dtype[dtype_in]]
        o1_l1_ty = np.ndarray[(o1_matmul_dims[0][0] * o1_matmul_dims[1][0],), np.dtype[dtype_out]]
        o2_l1_ty = np.ndarray[(o2_softmax_dims[0] * o2_softmax_dims[1],), np.dtype[dtype_out]]
        o3_l1_ty = np.ndarray[(o3_matmul_dims[0] * o3_matmul_dims[2],), np.dtype[dtype_out]]
        o4_l1_ty = np.ndarray[(o4_matmul_dims[0] * o4_matmul_dims[2],), np.dtype[dtype_out]]

        # AIE Core Function declarations
        # Last part of the name is whether the right matrix is row major
        # Need to declare the external functions for each object file used in the design 
        matmul_vectorized_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
        zero_projs = external_func(f"zero_{dtype_out_str}_{q_matmul_dims[0]}_{q_matmul_dims[1]}_{q_matmul_dims[2]}_1", inputs=[q_l1_ty])
        matmul_projs = external_func(
            matmul_vectorized_func_name + f"_{q_matmul_dims[0]}_{q_matmul_dims[1]}_{q_matmul_dims[2]}_1",
            inputs=[X_l1_ty, Wq_l1_ty, q_l1_ty],
        )
        zero_attn_score = external_func(f"zero_{dtype_out_str}_{o1_matmul_dims[0][0]}_{o1_matmul_dims[0][1]}_{o1_matmul_dims[1][0]}_0", inputs=[o1_l1_ty])
        matmul_attn_score = external_func(
            matmul_vectorized_func_name + f"_{o1_matmul_dims[0][0]}_{o1_matmul_dims[0][1]}_{o1_matmul_dims[1][0]}_0",
            inputs=[q_l1_ty_in, k_l1_ty_in, o1_l1_ty],
        )
        div_projs = external_func(f"div_2d_bf16", inputs=[o1_l1_ty, o1_l1_ty])
        softmax_o2 = external_func(
            f"softmax_{dtype_in_str}",
            inputs=[o2_l1_ty, o2_l1_ty],
        )
        zero_o3 = external_func(f"zero_{dtype_out_str}_{o3_matmul_dims[0]}_{o3_matmul_dims[1]}_{o3_matmul_dims[2]}_1", inputs=[o3_l1_ty])
        matmul_o3 = external_func(
            matmul_vectorized_func_name + f"_{o3_matmul_dims[0]}_{o3_matmul_dims[1]}_{o3_matmul_dims[2]}_1",
            inputs=[o2_l1_ty, v_l1_ty_in, o3_l1_ty],
        )
        zero_o4 = external_func(f"zero_{dtype_out_str}_{o4_matmul_dims[0]}_{o4_matmul_dims[1]}_{o4_matmul_dims[2]}_1", inputs=[o4_l1_ty])
        matmul_o4 = external_func(
            matmul_vectorized_func_name + f"_{o4_matmul_dims[0]}_{o4_matmul_dims[1]}_{o4_matmul_dims[2]}_1",
            inputs=[o3_l1_ty, Wo_l1_ty, o4_l1_ty],
        )

        if dev == "npu":
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 3rd columns
        else:
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 3rd columns
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # AIE-array data movement with object fifos
        Wq_l2l1_fifos = [None] * n_aie_rows_projs
        q_l1l2_fifos = [None] * n_aie_rows_projs
        Wk_l2l1_fifos = [None] * n_aie_rows_projs
        Wv_l2l1_fifos = [None] * n_aie_rows_projs
        k_l1l2_fifos = [None] * n_aie_rows_projs
        v_l1l2_fifos = [None] * n_aie_rows_projs

        X_l3l2_fifos = object_fifo(f"X_L3L2", shim_tiles[0], mem_tiles[0], fifo_depth, X_l2_ty)
        X_l2l1_fifos = object_fifo(f"X_L2L1", mem_tiles[0], [core_tiles[2 + row][0] for row in range(n_aie_rows_projs)] + [core_tiles[0 + row][0] for row in range(n_aie_rows_projs)] + [core_tiles[2 + row][1] for row in range(n_aie_rows_projs)], fifo_depth, X_l1_ty,
                                    [
                                        (q_matmul_dims[0] // r, r * q_matmul_dims[1]),
                                        (q_matmul_dims[1] // s, s),
                                        (r, q_matmul_dims[1]),
                                        (s, 1),
                                    ],)
        object_fifo_link(X_l3l2_fifos, X_l2l1_fifos)
        
        Wq_l3l2_fifos = object_fifo(f"Wq_L3L2", shim_tiles[0], mem_tiles[0], fifo_depth, Wq_l2_ty)
        for row in range(n_aie_rows_projs):
            Wq_l2l1_fifos[row] = object_fifo(f"Wq_L2L1_{row}", mem_tiles[0], core_tiles[2 + row][0], fifo_depth, Wq_l1_ty,
                                    [
                                        (q_matmul_dims[1] // s, s * q_matmul_dims[2]),
                                        (q_matmul_dims[2] // t, t),
                                        (s, q_matmul_dims[2]),
                                        (t, 1),
                                    ],)
        object_fifo_link(Wq_l3l2_fifos, Wq_l2l1_fifos, [], [q_matmul_dims[1] * q_matmul_dims[2] * i for i in range(n_aie_rows_projs)])
        
        Wk_l3l2_fifos = object_fifo(f"Wk_L3L2", shim_tiles[1], mem_tiles[0], fifo_depth, Wkv_l2_ty)
        for row in range(n_aie_rows_projs):
            Wk_l2l1_fifos[row] = object_fifo(f"Wk_L2L1_{row}", mem_tiles[0], core_tiles[0 + row][0], fifo_depth, Wkv_l1_ty,
                                    [
                                        (kv_matmul_dims[1] // s, s * kv_matmul_dims[2]),
                                        (kv_matmul_dims[2] // t, t),
                                        (s, kv_matmul_dims[2]),
                                        (t, 1),
                                    ],)
        object_fifo_link(Wk_l3l2_fifos, Wk_l2l1_fifos, [], [kv_matmul_dims[0] * kv_matmul_dims[2] * i for i in range(n_aie_rows_projs)])

        for row in range(n_aie_rows_projs):
            v_l1l2_fifos[row] = object_fifo(f"v_L1L2_{row}", core_tiles[2 + row][1], mem_tiles[1], fifo_depth, kv_l1_ty) 
        v_l2l3_fifos = object_fifo(f"v_L2L3", mem_tiles[1], shim_tiles[1], fifo_depth, kv_l2_ty,
                                    [
                                       (kv_matmul_dims[0] // r, r * kv_matmul_dims[2]),
                                       (r, t),
                                       (kv_matmul_dims[2] // t, r * t),
                                       (t, 1),
                                    ],)
        object_fifo_link(v_l1l2_fifos, v_l2l3_fifos, [kv_matmul_dims[1] * kv_matmul_dims[2] * i for i in range(n_aie_rows_projs)], [])

        for row in range(n_aie_rows_projs):
            q_l1l2_fifos[row] = object_fifo(f"q_L1L2_{row}", core_tiles[2 + row][0], mem_tiles[0], fifo_depth, q_l1_ty)
        q_l2l3_fifos = object_fifo(f"q_L2L3", mem_tiles[0], shim_tiles[0], fifo_depth, q_l2_ty,
                                    [
                                       (q_matmul_dims[0] // r, r * q_matmul_dims[2]),
                                       (r, t),
                                       (q_matmul_dims[2] // t, r * t),
                                       (t, 1),
                                    ],)
        object_fifo_link(q_l1l2_fifos, q_l2l3_fifos, [q_matmul_dims[0] * q_matmul_dims[2] * i for i in range(n_aie_rows_projs)], [])
        
        Wv_l3l2_fifos = object_fifo(f"Wv_L3L2", shim_tiles[1], mem_tiles[1], fifo_depth, Wkv_l2_ty)
        for row in range(n_aie_rows_projs):
            Wv_l2l1_fifos[row] = object_fifo(f"Wv_L2L1_{row}", mem_tiles[1], core_tiles[2 + row][1], fifo_depth, Wkv_l1_ty,
                                    [
                                        (kv_matmul_dims[1] // s, s * kv_matmul_dims[2]),
                                        (kv_matmul_dims[2] // t, t),
                                        (s, kv_matmul_dims[2]),
                                        (t, 1),
                                    ],)
        object_fifo_link(Wv_l3l2_fifos, Wv_l2l1_fifos, [], [kv_matmul_dims[1] * kv_matmul_dims[2] * i for i in range(n_aie_rows_projs)])

        for row in range(n_aie_rows_projs):
            k_l1l2_fifos[row] = object_fifo(f"k_L1L2_{row}", core_tiles[0 + row][0], mem_tiles[0], fifo_depth, kv_l1_ty) 
        k_l2l3_fifos = object_fifo(f"k_L2L3", mem_tiles[0], shim_tiles[0], fifo_depth, kv_l2_ty,
                                    [
                                       (kv_matmul_dims[0] // r, r * kv_matmul_dims[2]),
                                       (r, t),
                                       (kv_matmul_dims[2] // t, r * t),
                                       (t, 1),
                                    ],)
        object_fifo_link(k_l1l2_fifos, k_l2l3_fifos, [kv_matmul_dims[0] * kv_matmul_dims[2] * i for i in range(n_aie_rows_projs)], []) 

        q_l3l2_fifos = object_fifo(f"q_L3L2", shim_tiles[2], mem_tiles[2], fifo_depth, q_l1_ty_in,)
        q_l2l1_fifos = object_fifo(f"q_L2L1", mem_tiles[2], core_tiles[0][2], fifo_depth, q_l1_ty_in,
                                    [
                                        (o1_matmul_dims[0][0] // r, r * o1_matmul_dims[0][1]),
                                        (o1_matmul_dims[0][1] // s, s),
                                        (r, o1_matmul_dims[0][1]),
                                        (s, 1),
                                    ],)
        object_fifo_link(q_l3l2_fifos, q_l2l1_fifos)
        k_l3l2_fifos = object_fifo(f"k_L3L2", shim_tiles[3], mem_tiles[3], fifo_depth, k_l1_ty_in,)
        k_l2l1_fifos = object_fifo(f"k_L2L1", mem_tiles[3], core_tiles[0][2], fifo_depth, k_l1_ty_in,
                                    [
                                        (o1_matmul_dims[0][1] // t, t * o1_matmul_dims[1][1]),
                                        (o1_matmul_dims[1][1] // s, s),
                                        (t, o1_matmul_dims[1][1]),
                                        (s, 1),
                                    ],)
        object_fifo_link(k_l3l2_fifos, k_l2l1_fifos)
        o1_l1l2_fifos = object_fifo(f"o1_L1L2", core_tiles[0][2], mem_tiles[2], fifo_depth, o1_l1_ty)
        # Need to send tile back to L2 since we need 4-D transformation
        o1_l2l1_fifos = object_fifo(f"o1_L2L1", mem_tiles[2], core_tiles[1][2], fifo_depth, o2_l1_ty,
                                    [
                                        (o1_matmul_dims[0][0] // r, r * o1_matmul_dims[1][0]),
                                        (r, t),
                                        (o1_matmul_dims[1][0] // t, r * t),
                                        (t, 1),
                                    ],)
        object_fifo_link(o1_l1l2_fifos, o1_l2l1_fifos)
        o2_l1l1_fifos = object_fifo(f"o2_L1L1", core_tiles[1][2], core_tiles[2][2], fifo_depth, o2_l1_ty)
        v_l3l2_fifos = object_fifo(f"v_L3L2", shim_tiles[3], mem_tiles[3], fifo_depth, v_l1_ty_in,)
        v_l2l1_fifos = object_fifo(f"v_L2L1", mem_tiles[3], core_tiles[2][2], fifo_depth, v_l1_ty_in,
                                    [
                                        (o3_matmul_dims[1] // t, t * o3_matmul_dims[2]),
                                        (o3_matmul_dims[2] // s, s),
                                        (t, o3_matmul_dims[2]),
                                        (s, 1),
                                    ],)
        object_fifo_link(v_l3l2_fifos, v_l2l1_fifos)
        o3_l1l1_fifos = object_fifo(f"o3_L1L1", core_tiles[2][2], core_tiles[3][2], fifo_depth, o3_l1_ty)
        Wo_l3l2_fifos = object_fifo(f"Wo_L3L2", shim_tiles[2], mem_tiles[2], fifo_depth, Wo_l1_ty)
        Wo_l2l1_fifos = object_fifo(f"Wo_L2L1", mem_tiles[2], core_tiles[3][2], fifo_depth, Wo_l1_ty,
                                    [
                                        (o4_matmul_dims[1] // s, s * o4_matmul_dims[2]),
                                        (o4_matmul_dims[2] // t, t),
                                        (s, o4_matmul_dims[2]),
                                        (t, 1),
                                    ],) 
        object_fifo_link(Wo_l3l2_fifos, Wo_l2l1_fifos)
        o4_l1l2_fifos = object_fifo(f"o4_L1L2", core_tiles[3][2], mem_tiles[2], fifo_depth, o4_l1_ty)
        o4_l2l3_fifos = object_fifo(f"o4_L2L3", mem_tiles[2], shim_tiles[2], fifo_depth, o4_l1_ty,
                                    [
                                       (o4_matmul_dims[0] // r, r * o4_matmul_dims[2]),
                                       (r, t),
                                       (o4_matmul_dims[2] // t, r * t),
                                       (t, 1),
                                    ],)
        object_fifo_link(o4_l1l2_fifos, o4_l2l3_fifos)

        # Set up compute tiles
        for row in range(n_aie_rows_projs):
            @core(core_tiles[2 + row][0], f"mha_mm_{q_matmul_dims[0]}x{q_matmul_dims[1]}x{q_matmul_dims[2]}_row_major.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(H):
                        for _ in range_(head_dim // q_matmul_dims[2]):
                            elem_q = q_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                            zero_projs(elem_q)
                            for _ in range_(K // q_matmul_dims[1]):
                                elem_in_X = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                                elem_in_wq = Wq_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                                matmul_projs(elem_in_X, elem_in_wq, elem_q)
                                Wq_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                            q_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        for row in range(n_aie_rows_projs):
            @core(core_tiles[0 + row][0], f"mha_mm_{kv_matmul_dims[0]}x{kv_matmul_dims[1]}x{kv_matmul_dims[2]}_row_major.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(H):
                        for _ in range_(head_dim // kv_matmul_dims[2]):
                            elem_k = k_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                            zero_projs(elem_k)
                            for _ in range_(K // kv_matmul_dims[1]):
                                elem_in_xk = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                                elem_in_wk = Wk_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                                matmul_projs(elem_in_xk, elem_in_wk, elem_k)
                                Wk_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                            k_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        for row in range(n_aie_rows_projs):
            @core(core_tiles[2 + row][1], f"mha_mm_{kv_matmul_dims[0]}x{kv_matmul_dims[1]}x{kv_matmul_dims[2]}_row_major.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(H):
                        for _ in range_(head_dim // kv_matmul_dims[2]):
                            elem_v = v_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                            zero_projs(elem_v)
                            for _ in range_(K // kv_matmul_dims[1]):
                                elem_in_xv = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                                elem_in_wv = Wv_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                                matmul_projs(elem_in_xv, elem_in_wv, elem_v)
                                Wv_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                            v_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[0][2], f"mha_mm_{o1_matmul_dims[0][0]}x{o1_matmul_dims[0][1]}x{o1_matmul_dims[1][0]}_col_major.o", stack_size=0xD00)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    elem_o1 = o1_l1l2_fifos.acquire(ObjectFifoPort.Produce, 1)
                    zero_attn_score(elem_o1)
                    for _ in range_(head_dim // o1_matmul_dims[0][1]):
                        elem_in_q = q_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_k = k_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        matmul_attn_score(elem_in_q, elem_in_k, elem_o1)
                        q_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        k_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                    div_projs(elem_o1, elem_o1)
                    o1_l1l2_fifos.release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[1][2], f"mha_softmax.o") # Make sure to use the bundled obj file, not the softmax-only obj file
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    elem_o2 = o2_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                    elem_o1 = o1_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                    softmax_o2(elem_o1, elem_o2)
                    o1_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                    o2_l1l1_fifos.release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[2][2], f"mha_mm_{o3_matmul_dims[0]}x{o3_matmul_dims[1]}x{o3_matmul_dims[2]}_row_major.o", stack_size=0xD00)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    elem_in_o2 = o2_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                    for _ in range_(head_dim // o3_matmul_dims[2]):
                        for _ in range_(M // o3_matmul_dims[1]):
                            elem_o3 = o3_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                            zero_o3(elem_o3)
                            elem_in_v = v_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            matmul_o3(elem_in_o2, elem_in_v, elem_o3)
                            v_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                            o3_l1l1_fifos.release(ObjectFifoPort.Produce, 1)
                    o2_l1l1_fifos.release(ObjectFifoPort.Consume, 1)

        @core(core_tiles[3][2], f"mha_mm_{o4_matmul_dims[0]}x{o4_matmul_dims[1]}x{o4_matmul_dims[2]}_row_major.o", stack_size=0xD00)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                elem_o4 = o4_l1l2_fifos.acquire(ObjectFifoPort.Produce, 1)
                zero_o4(elem_o4)
                for _ in range_(K // o4_matmul_dims[1]):
                    elem_in_o3 = o3_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                    elem_in_wo = Wo_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                    matmul_o4(elem_in_o3, elem_in_wo, elem_o4)
                    Wo_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                    o3_l1l1_fifos.release(ObjectFifoPort.Consume, 1)
                o4_l1l2_fifos.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[((M * K) + (2 * K * N),), np.dtype[dtype_in]], # Order: X, W_Q, W_K
            np.ndarray[(2 * K * N,), np.dtype[dtype_in]], # Order: W_V, W_O
            np.ndarray[((4 * M * N),), np.dtype[dtype_out]], # Order: Q, K, V, o4
        )
        def sequence(A, B, C):
                # One iteration generates the output for 32 rows, so need to repeat
                # for the rest of the output's sequence
                for row_offset in range(0, M, q_matmul_dims[0]):
                    npu_dma_memcpy_nd(
                        metadata=X_l3l2_fifos,
                        bd_id=0,
                        mem=A,
                        offsets=[0, 0, 0, row_offset * K],
                        sizes=[N // q_matmul_dims[2], K // q_matmul_dims[1], q_matmul_dims[0], q_matmul_dims[1]],
                        strides=[0, q_matmul_dims[1], K, 1],
                    )

                    for col_offset in range(0, N, q_matmul_dims[2] * n_aie_cols_projs):
                        npu_dma_memcpy_nd(
                            metadata=Wq_l3l2_fifos,
                            bd_id=1,
                            mem=A,
                            offsets=[0, 0, 0, M * K + col_offset],
                            sizes=[K // q_matmul_dims[1], n_aie_rows_projs, q_matmul_dims[1], q_matmul_dims[2]],
                            strides=[q_matmul_dims[1] * N, q_matmul_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=Wk_l3l2_fifos,
                            bd_id=2,
                            mem=A,
                            offsets=[0, 0, 0, M * K + K * N + col_offset],
                            sizes=[K // kv_matmul_dims[1], n_aie_rows_projs, kv_matmul_dims[1], kv_matmul_dims[2]],
                            strides=[kv_matmul_dims[1] * N, kv_matmul_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=Wv_l3l2_fifos,
                            bd_id=3,
                            mem=B,
                            offsets=[0, 0, 0, col_offset],
                            sizes=[K // kv_matmul_dims[1], n_aie_rows_projs, kv_matmul_dims[1], kv_matmul_dims[2]],
                            strides=[kv_matmul_dims[1] * N, kv_matmul_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=q_l2l3_fifos,
                            bd_id=4,
                            mem=C,
                            offsets=[0, 0, 0, row_offset * N + col_offset],
                            sizes=[1, n_aie_rows_projs, q_matmul_dims[0], q_matmul_dims[2]],
                            strides=[0, q_matmul_dims[0] * q_matmul_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=k_l2l3_fifos,
                            bd_id=5,
                            mem=C,
                            offsets=[0, 0, 0, M * N + row_offset * N + col_offset],
                            sizes=[1, n_aie_rows_projs, kv_matmul_dims[0], kv_matmul_dims[2]],
                            strides=[0, kv_matmul_dims[0] * kv_matmul_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=v_l2l3_fifos,
                            bd_id=6,
                            mem=C,
                            offsets=[0, 0, 0, 2 * M * N + row_offset * N + col_offset],
                            sizes=[1, n_aie_rows_projs, kv_matmul_dims[0], kv_matmul_dims[2]],
                            strides=[0, kv_matmul_dims[0] * kv_matmul_dims[2], N, 1],
                        )

                        dma_wait(q_l2l3_fifos, k_l2l3_fifos, v_l2l3_fifos)
                
                for row_offset in range(0, M, o4_matmul_dims[0]):
                    for col_offset in range(0, N, o4_matmul_dims[2]):
                        for head in range(H):
                            npu_dma_memcpy_nd(
                                metadata=q_l3l2_fifos,
                                bd_id=0,
                                mem=C,
                                offsets=[0, 0, 0, row_offset * N + head * head_dim],
                                sizes=[M // o1_matmul_dims[1][0], head_dim // o1_matmul_dims[0][1], o1_matmul_dims[0][0], o1_matmul_dims[0][1]],
                                strides=[0, o1_matmul_dims[0][1], N, 1],
                                issue_token=True,
                            )

                            npu_dma_memcpy_nd(
                                metadata=k_l3l2_fifos,
                                bd_id=1,
                                mem=C,
                                offsets=[0, 0, 0, M * N + head * head_dim],
                                sizes=[M // o1_matmul_dims[1][0], head_dim // o1_matmul_dims[1][1], o1_matmul_dims[1][0], o1_matmul_dims[1][1]],
                                strides=[o1_matmul_dims[1][0] * N, o1_matmul_dims[1][1], N, 1],
                                issue_token=True,
                            )

                            npu_dma_memcpy_nd(
                                metadata=v_l3l2_fifos,
                                bd_id=2,
                                mem=C,
                                offsets=[0, 0, 0, 2 * M * N + head * head_dim],
                                sizes=[head_dim // o3_matmul_dims[2], M // o3_matmul_dims[1], o3_matmul_dims[1], o3_matmul_dims[2]],
                                strides=[o3_matmul_dims[2], o3_matmul_dims[1] * N, N, 1],
                                issue_token=True,
                            )

                            npu_dma_memcpy_nd(
                                metadata=Wo_l3l2_fifos,
                                bd_id=3,
                                mem=B,
                                offsets=[0, 0, 0, K * N + head * head_dim * N + col_offset],
                                sizes=[1, head_dim // o4_matmul_dims[1], o4_matmul_dims[1], o4_matmul_dims[2]],
                                strides=[0, o4_matmul_dims[1] * N, N, 1],
                                issue_token=True,
                            )
                            dma_wait(q_l3l2_fifos, k_l3l2_fifos, v_l3l2_fifos, Wo_l3l2_fifos)

                        npu_dma_memcpy_nd(
                            metadata=o4_l2l3_fifos,
                            bd_id=4,
                            mem=C,
                            offsets=[0, 0, 0, 3 * M * N + row_offset * N + col_offset],
                            sizes=[1, 1, o4_matmul_dims[0], o4_matmul_dims[2]],
                            strides=[0, 0, N, 1],
                        )

                        dma_wait(o4_l2l3_fifos)


if __name__ == "__main__":
    main()
