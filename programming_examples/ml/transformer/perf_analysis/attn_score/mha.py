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
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 3, 4, 8], default=4)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
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
    trace_size,
    generate_taps=False,
):
    o4_rows = 4
    head_dim = N // H

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    q_matmul_dims = (32, 192, 16)
    kv_matmul_dims = (256, 32, 16)
    o1_matmul_dims = (q_matmul_dims[0], q_matmul_dims[2], kv_matmul_dims[0]) # (32, 16, 256)
    o3_matmul_dims = (q_matmul_dims[0], kv_matmul_dims[0], kv_matmul_dims[2]) # (32, 256, 16)
    o4_matmul_dims = (q_matmul_dims[0], kv_matmul_dims[2], 192) # (32, 16, 192)
    o2_softmax_dims = (q_matmul_dims[0], kv_matmul_dims[0]) # (32, 256)
    matmul_dims = [q_matmul_dims, kv_matmul_dims, o1_matmul_dims,
                   o3_matmul_dims, o4_matmul_dims]
    
    if dev == "npu":
        if dtype_in_str == "bf16":
            r = 4
            s = 8
            t = 4
        elif dtype_in_str == "i8":
            r = 4
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 4
    else:
        if dtype_in_str == "bf16":
            r = 8
            s = 8
            t = 8
        elif dtype_in_str == "i8":
            r = 8
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 8

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(dtype_out, np.integer), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    assert (N % H == 0), """Embedding dimension must be divisible by number of heads"""
    for dims in matmul_dims:
        assert (dims[0] * dims[1] * dims[2] > 0), f"Matmul dimensions {dims} must be non-zero"
        assert (dims[0] % r == 0), f"Matmul dimension {dims[0]} must be divisible by {r}"
        assert (dims[1] % s == 0), f"Matmul dimension {dims[1]} must be divisible by {s}"
        assert (dims[2] % t == 0), f"Matmul dimension {dims[2]} must be divisible by {t}"

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    if dev == "npu":
        if n_aie_cols == 3:
            # Use this virtualization to generate the xclbin, but the 
            # design will only use one column of cores.
            dev_ty = AIEDevice.npu1_4col
    else:
        if n_aie_cols == 3:
            dev_ty = AIEDevice.npu2

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    @device(dev_ty)
    def device_body():
        Xq_l1_ty = np.ndarray[(q_matmul_dims[0] * q_matmul_dims[1],), np.dtype[dtype_in]]
        Wq_l1_ty = np.ndarray[(q_matmul_dims[1] * q_matmul_dims[2],), np.dtype[dtype_in]]
        q_l1_ty = np.ndarray[(q_matmul_dims[0] * q_matmul_dims[2],), np.dtype[dtype_out]]

        Xkv_l1_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[1],), np.dtype[dtype_in]]
        Wkv_l1_ty = np.ndarray[(kv_matmul_dims[1] * kv_matmul_dims[2],), np.dtype[dtype_in]]
        kv_l1_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[2],), np.dtype[dtype_out]]

        o1_l1_ty = np.ndarray[(o1_matmul_dims[0] * o1_matmul_dims[2],), np.dtype[dtype_out]]
        o2_l1_ty = np.ndarray[(o2_softmax_dims[0] * o2_softmax_dims[1],), np.dtype[dtype_out]]

        o3_l1_ty = np.ndarray[(o3_matmul_dims[0] * o3_matmul_dims[2],), np.dtype[dtype_out]]

        Wo_l1_ty = np.ndarray[(o4_matmul_dims[1] * o4_matmul_dims[2],), np.dtype[dtype_in]]
        o4_l1_ty = np.ndarray[(o4_matmul_dims[0] * o4_matmul_dims[2],), np.dtype[dtype_out]]
        Wo_l2_ty = np.ndarray[(o4_matmul_dims[1] * o4_matmul_dims[2] * o4_rows,), np.dtype[dtype_in]]
        o4_l2_ty = np.ndarray[(o4_matmul_dims[0] * o4_matmul_dims[2] * o4_rows,), np.dtype[dtype_out]]


        # AIE Core Function declarations
        # TODO: The matmul functions should have different input data types based 
        # on which operation is being performed, if the Makefile specifies 
        # differing input and output data types. Currently this isn't being handled,
        # i.e. matmul for o1 and o3 should be i16xi16->i16 and o4 i16xi8->i16.
        # The former is easy to add, but latter requires a new matmul function.
        zero_q = external_func(f"zero_{dtype_out_str}_{q_matmul_dims[0]}_{q_matmul_dims[1]}_{q_matmul_dims[2]}", inputs=[q_l1_ty])
        zero_kv = external_func(f"zero_{dtype_out_str}_{kv_matmul_dims[0]}_{kv_matmul_dims[1]}_{kv_matmul_dims[2]}", inputs=[kv_l1_ty])

        matmul_vectorized_func_name = (
            f"matmul_{dtype_in_str}_{dtype_out_str}"
            if not b_col_maj
            else f"matmul_{dtype_in_str}_{dtype_out_str}_b_col_maj"
        )
        matmul_q = external_func(
            matmul_vectorized_func_name + f"_{q_matmul_dims[0]}_{q_matmul_dims[1]}_{q_matmul_dims[2]}",
            inputs=[Xq_l1_ty, Wq_l1_ty, q_l1_ty],
        )
        matmul_kv = external_func(
            matmul_vectorized_func_name + f"_{kv_matmul_dims[0]}_{kv_matmul_dims[1]}_{kv_matmul_dims[2]}",
            inputs=[Xkv_l1_ty, Wkv_l1_ty, kv_l1_ty],
        )
        matmul_o1 = external_func(
            matmul_vectorized_func_name + f"_{o1_matmul_dims[0]}_{o1_matmul_dims[1]}_{o1_matmul_dims[2]}",
            inputs=[q_l1_ty, kv_l1_ty, o1_l1_ty],
        )
        softmax_o2 = external_func(
            f"softmax_{dtype_in_str}",
            inputs=[o1_l1_ty, o2_l1_ty, np.int32],
        )
        matmul_o3 = external_func(
            matmul_vectorized_func_name + f"_{o3_matmul_dims[0]}_{o3_matmul_dims[1]}_{o3_matmul_dims[2]}",
            inputs=[o2_l1_ty, kv_l1_ty, o3_l1_ty],
        )
        matmul_o4 = external_func(
            matmul_vectorized_func_name + f"_{o4_matmul_dims[0]}_{o4_matmul_dims[1]}_{o4_matmul_dims[2]}",
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
        # TODO: Will likely need to adjust the memory access patterns based on the matrix mult API tiling
        Xq_l3l2_fifos = object_fifo(f"Xq_L3L2", shim_tiles[2], mem_tiles[2], fifo_depth, Xq_l1_ty)
        Xq_l2l1_fifos = object_fifo(f"Xq_L2L1", mem_tiles[2], core_tiles[0][1], fifo_depth, Xq_l1_ty,
                                    [
                                        (q_matmul_dims[0] // r, r * q_matmul_dims[1]),
                                        (q_matmul_dims[1] // s, s),
                                        (r, q_matmul_dims[1]),
                                        (s, 1),
                                    ],)
        object_fifo_link(Xq_l3l2_fifos, Xq_l2l1_fifos)
        
        Wq_l3l2_fifos = object_fifo(f"Wq_L3L2", shim_tiles[1], mem_tiles[1], fifo_depth, Wq_l1_ty)
        Wq_l2l1_fifos = object_fifo(f"Wq_L2L1", mem_tiles[1], core_tiles[0][1], fifo_depth, Wq_l1_ty,
                                    [
                                        (q_matmul_dims[1] // s, s * q_matmul_dims[2]),
                                        (q_matmul_dims[2] // t, t),
                                        (s, q_matmul_dims[2]),
                                        (t, 1),
                                    ],)
        object_fifo_link(Wq_l3l2_fifos, Wq_l2l1_fifos)
        Xkv_l3l2_fifos = object_fifo(f"Xkv_L3L2", shim_tiles[0], mem_tiles[0], fifo_depth, Xkv_l1_ty)
        Xkv_l2l1_fifos = object_fifo(f"Xkv_L2L1", mem_tiles[0], [core_tiles[1][0], core_tiles[2][0]], fifo_depth, Xkv_l1_ty,
                                    [
                                        (kv_matmul_dims[0] // r, r * kv_matmul_dims[1]),
                                        (kv_matmul_dims[1] // s, s),
                                        (r, kv_matmul_dims[1]),
                                        (s, 1),
                                    ],)
        object_fifo_link(Xkv_l3l2_fifos, Xkv_l2l1_fifos)
        Wk_l3l2_fifos = object_fifo(f"Wk_L3L2", shim_tiles[1], mem_tiles[1], fifo_depth, Wkv_l1_ty)
        Wk_l2l1_fifos = object_fifo(f"Wk_L2L1", mem_tiles[1], core_tiles[1][0], fifo_depth, Wkv_l1_ty,
                                    [
                                        (kv_matmul_dims[1] // s, s * kv_matmul_dims[2]),
                                        (kv_matmul_dims[2] // t, t),
                                        (s, kv_matmul_dims[2]),
                                        (t, 1),
                                    ],)
        object_fifo_link(Wk_l3l2_fifos, Wk_l2l1_fifos)
        Wv_l3l2_fifos = object_fifo(f"Wv_L3L2", shim_tiles[0], mem_tiles[0], fifo_depth, Wkv_l1_ty)
        Wv_l2l1_fifos = object_fifo(f"Wv_L2L1", mem_tiles[0], core_tiles[2][0], fifo_depth, Wkv_l1_ty,
                                    [
                                        (kv_matmul_dims[1] // s, s * kv_matmul_dims[2]),
                                        (kv_matmul_dims[2] // t, t),
                                        (s, kv_matmul_dims[2]),
                                        (t, 1),
                                    ],)
        object_fifo_link(Wv_l3l2_fifos, Wv_l2l1_fifos)
        q_l1l1_fifos = object_fifo(f"q_L1L1", core_tiles[0][1], core_tiles[1][1], fifo_depth, q_l1_ty)
        k_l1l1_fifos = object_fifo(f"k_L1L1", core_tiles[1][0], core_tiles[1][1], fifo_depth, kv_l1_ty)
        v_l1l1_fifos = object_fifo(f"v_L1L1", core_tiles[2][0], core_tiles[3][1], fifo_depth, kv_l1_ty)
        o1_l1l1_fifos = object_fifo(f"o1_L1L1", core_tiles[1][1], core_tiles[2][1], fifo_depth, o1_l1_ty)
        o2_l1l1_fifos = object_fifo(f"o2_L1L1", core_tiles[2][1], core_tiles[3][1], fifo_depth, o2_l1_ty)
        o3_l1l1_fifos = object_fifo(f"o3_L1L1", core_tiles[3][1], [core_tiles[j][2] for j in range(o4_rows)], fifo_depth, o3_l1_ty) # broadcast along one column
        Wo_l3l2_fifos = object_fifo(f"Wo_L3L2", shim_tiles[2], mem_tiles[2], fifo_depth, Wo_l2_ty)
        Wo_l2l1_fifos = [None] * o4_rows
        for row in range(o4_rows):
            Wo_l2l1_fifos[row] = object_fifo(f"Wo_L2L1_{row}", mem_tiles[2], core_tiles[row][2], fifo_depth, Wo_l1_ty)
        object_fifo_link(Wo_l3l2_fifos, [Wo_l2l1_fifos[i] for i in range(o4_rows)], [], [o4_matmul_dims[1] * o4_matmul_dims[2] * i for i in range(o4_rows)])
        o4_l1l2_fifos = [None] * o4_rows
        for row in range(o4_rows):
            o4_l1l2_fifos[row] = object_fifo(f"o4_L1L2_{row}", core_tiles[row][2], mem_tiles[2], fifo_depth, o4_l1_ty)
        o4_l2l3_fifos = object_fifo(f"o4_L2L3", mem_tiles[2], shim_tiles[2], fifo_depth, o4_l2_ty)
        object_fifo_link([o4_l1l2_fifos[i] for i in range(o4_rows)], o4_l2l3_fifos, [o4_matmul_dims[0] * o4_matmul_dims[2] * i for i in range(o4_rows)], [])
        

        # Set up compute tiles  
        @core(core_tiles[1][0], f"mha_mm_{kv_matmul_dims[0]}x{kv_matmul_dims[1]}x{kv_matmul_dims[2]}.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    for _ in range_(head_dim // kv_matmul_dims[2]):
                        elem_k = k_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                        zero_kv(elem_k)
                        for _ in range_(K // kv_matmul_dims[1]):
                            elem_in_xk = Xkv_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wk = Wk_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            matmul_kv(elem_in_xk, elem_in_wk, elem_k)
                            Wk_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                            Xkv_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        k_l1l1_fifos.release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[2][0], f"mha_mm_{kv_matmul_dims[0]}x{kv_matmul_dims[1]}x{kv_matmul_dims[2]}.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    for _ in range_(head_dim // kv_matmul_dims[2]):
                        elem_v = v_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                        zero_kv(elem_v)
                        for _ in range_(K // kv_matmul_dims[1]):
                            elem_in_xv = Xkv_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wv = Wv_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            matmul_kv(elem_in_xv, elem_in_wv, elem_v)
                            Wv_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                            Xkv_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        v_l1l1_fifos.release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[0][1], f"mha_mm_{q_matmul_dims[0]}x{q_matmul_dims[1]}x{q_matmul_dims[2]}.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    for _ in range_(head_dim // q_matmul_dims[2]):
                        elem_q = q_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                        zero_q(elem_q)
                        for _ in range_(K // q_matmul_dims[1]):
                            elem_in_xq = Xq_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wq = Wq_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            matmul_q(elem_in_xq, elem_in_wq, elem_q)
                            Wq_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                            Xq_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        q_l1l1_fifos.release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[1][1], f"mha_mm_{o1_matmul_dims[0]}x{o1_matmul_dims[1]}x{o1_matmul_dims[2]}.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    elem_o1 = o1_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                    # zero_o1(elem_o1)
                    for _ in range_(head_dim // o1_matmul_dims[1]):
                        elem_in_q = q_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_k = k_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        matmul_o1(elem_in_q, elem_in_k, elem_o1)
                        q_l1l1_fifos.release(ObjectFifoPort.Consume, 1)
                        k_l1l1_fifos.release(ObjectFifoPort.Consume, 1)
                    o1_l1l1_fifos.release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[2][1], f"mha_softmax.o") # Make sure to use the bundled obj file, not the softmax-only obj file
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    elem_o2 = o2_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                    elem_o1 = o1_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                    softmax_o2(elem_o1, elem_o2, 32 * 256)
                    o1_l1l1_fifos.release(ObjectFifoPort.Consume, 1)
                    o2_l1l1_fifos.release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[3][1], f"mha_mm_{o3_matmul_dims[0]}x{o3_matmul_dims[1]}x{o3_matmul_dims[2]}.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    elem_in_o2 = o2_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                    for _ in range_(head_dim // o3_matmul_dims[2]):
                        elem_o3 = o3_l1l1_fifos.acquire(ObjectFifoPort.Produce, 1)
                        # zero_o3(elem_o3)
                        elem_in_v = v_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        matmul_o3(elem_in_o2, elem_in_v, elem_o3)
                        v_l1l1_fifos.release(ObjectFifoPort.Consume, 1)
                        o3_l1l1_fifos.release(ObjectFifoPort.Produce, 1)
                    o2_l1l1_fifos.release(ObjectFifoPort.Consume, 1)

        for row in range(o4_rows):
            @core(core_tiles[row][2], f"mha_mm_{o4_matmul_dims[0]}x{o4_matmul_dims[1]}x{o4_matmul_dims[2]}.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_o4 = o4_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                    # zero_o4(elem_o4)
                    for _ in range_(K // o4_matmul_dims[1]):
                        elem_in_o3 = o3_l1l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_wo = Wo_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                        matmul_o4(elem_in_o3, elem_in_wo, elem_o4)
                        Wo_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                        o3_l1l1_fifos.release(ObjectFifoPort.Consume, 1)
                    o4_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)


        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[((M * K) + (2 * K * N),), np.dtype[dtype_in]], # Order: X, W_Q, W_K
            np.ndarray[(2 * K * N,), np.dtype[dtype_in]], # Order: W_V, W_O 
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
                # One iteration generates the output for 32 rows, so need to repeat
                # for the rest of the output's sequence
                for row_offset in range(0, M, o4_matmul_dims[0]):
                    npu_dma_memcpy_nd(
                        metadata=Xq_l3l2_fifos,
                        bd_id=0,
                        mem=A,
                        offsets=[0, 0, 0, row_offset * K],
                        sizes=[N // q_matmul_dims[2], K // q_matmul_dims[1], q_matmul_dims[0], q_matmul_dims[1]],
                        strides=[0, q_matmul_dims[1], K, 1],
                    )

                    npu_dma_memcpy_nd(
                        metadata=Wq_l3l2_fifos,
                        bd_id=2,
                        mem=A,
                        offsets=[0, 0, 0, M * K],
                        sizes=[N // q_matmul_dims[2], K // q_matmul_dims[1], q_matmul_dims[1], q_matmul_dims[2]],
                        strides=[q_matmul_dims[2], q_matmul_dims[1] * N, N, 1],
                    )

                    npu_dma_memcpy_nd(
                        metadata=Xkv_l3l2_fifos,
                        bd_id=1,
                        mem=A,
                        offsets=[0, 0, 0, 0],
                        sizes=[N // kv_matmul_dims[2], K // kv_matmul_dims[1], M, kv_matmul_dims[1]],
                        strides=[0, kv_matmul_dims[1], K, 1],
                    )

                    npu_dma_memcpy_nd(
                        metadata=Wk_l3l2_fifos,
                        bd_id=3,
                        mem=A,
                        offsets=[0, 0, 0, M * K + K * N],
                        sizes=[N // kv_matmul_dims[2], K // kv_matmul_dims[1], kv_matmul_dims[1], kv_matmul_dims[2]],
                        strides=[kv_matmul_dims[2], kv_matmul_dims[1] * N, N, 1],
                    )

                    # B input transfer:
                    npu_dma_memcpy_nd(
                        metadata=Wv_l3l2_fifos,
                        bd_id=4,
                        mem=B,
                        offsets=[0, 0, 0, 0],
                        sizes=[N // kv_matmul_dims[2], K // kv_matmul_dims[1], kv_matmul_dims[1], kv_matmul_dims[2]],
                        strides=[kv_matmul_dims[2], kv_matmul_dims[1] * N, N, 1],
                    )

                    npu_dma_memcpy_nd(
                        metadata=Wo_l3l2_fifos,
                        bd_id=5,
                        mem=B,
                        offsets=[0, 0, 0, K * N],
                        sizes=[1, K // o4_matmul_dims[1], o4_matmul_dims[0], N],
                        strides=[0, o4_matmul_dims[0] * N, N, 1],
                    )

                    # Output transfer:
                    npu_dma_memcpy_nd(
                        metadata=o4_l2l3_fifos,
                        bd_id=6,
                        mem=C,
                        offsets=[0, 0, 0, row_offset * N],
                        sizes=[1, 1, o4_matmul_dims[0], N],
                        strides=[0, 0, N, 1],
                    )
                    dma_wait(o4_l2l3_fifos)


if __name__ == "__main__":
    main()
