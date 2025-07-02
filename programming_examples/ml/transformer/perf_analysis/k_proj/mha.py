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
        start_row_k = 0
    else:
        n_aie_rows_projs = 2
        start_row_k = 0
    o4_rows = 4
    head_dim = N // H

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    kv_matmul_dims = (64, 64, 64)
    matmul_dims = [kv_matmul_dims]
    
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
            M % dims[0]  == 0
        ), """A must be tileable into (m * n_aie_rows_projs, dims[1])-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    for dims in matmul_dims:
        assert K % dims[1] == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    for dims in matmul_dims:
        assert (
            N % (dims[2] * n_aie_rows_projs) == 0
        ), """B must be tileable into (k, dims[2] * n_aie_rows_projs)-sized blocks"""

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
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu1_4col
    else:
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        X_l2_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[1],), np.dtype[dtype_in]]
        Wkv_l2_ty = np.ndarray[(kv_matmul_dims[1] * kv_matmul_dims[2] * n_aie_rows_projs,), np.dtype[dtype_in]]
        kv_l2_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[2] * n_aie_rows_projs,), np.dtype[dtype_out]]
        X_l1_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[1],), np.dtype[dtype_in]]
        Wkv_l1_ty = np.ndarray[(kv_matmul_dims[1] * kv_matmul_dims[2],), np.dtype[dtype_in]]
        kv_l1_ty = np.ndarray[(kv_matmul_dims[0] * kv_matmul_dims[2],), np.dtype[dtype_out]]

        # AIE Core Function declarations
        zero_kv = external_func(f"zero_{dtype_out_str}_{kv_matmul_dims[0]}_{kv_matmul_dims[1]}_{kv_matmul_dims[2]}", inputs=[kv_l1_ty])
        matmul_vectorized_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
        matmul_kv = external_func(
            matmul_vectorized_func_name + f"_{kv_matmul_dims[0]}_{kv_matmul_dims[1]}_{kv_matmul_dims[2]}",
            inputs=[X_l1_ty, Wkv_l1_ty, kv_l1_ty],
        )

        if dev == "npu":
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 3rd columns
        else:
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 3rd columns
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # AIE-array data movement with object fifos
        Wk_l2l1_fifos = [None] * n_aie_rows_projs
        kv_l1l2_fifos = [None] * n_aie_rows_projs

        X_l3l2_fifos = object_fifo(f"X_L3L2", shim_tiles[0], mem_tiles[0], fifo_depth, X_l2_ty)
        X_l2l1_fifos = object_fifo(f"X_L2L1", mem_tiles[0], [core_tiles[start_row_k + row][0] for row in range(n_aie_rows_projs)], fifo_depth, X_l1_ty,
                                    [
                                        (kv_matmul_dims[0] // r, r * kv_matmul_dims[1]),
                                        (kv_matmul_dims[1] // s, s),
                                        (r, kv_matmul_dims[1]),
                                        (s, 1),
                                    ],)
        object_fifo_link(X_l3l2_fifos, X_l2l1_fifos)
        
        Wk_l3l2_fifos = object_fifo(f"Wk_L3L2", shim_tiles[1], mem_tiles[0], fifo_depth, Wkv_l2_ty,)
        for row in range(n_aie_rows_projs):
            Wk_l2l1_fifos[row] = object_fifo(f"Wk_L2L1_{row}", mem_tiles[0], core_tiles[start_row_k + row][0], fifo_depth, Wkv_l1_ty,
                                            [
                                                (kv_matmul_dims[1] // s, s * kv_matmul_dims[2]),
                                                (kv_matmul_dims[2] // t, t),
                                                (s, kv_matmul_dims[2]),
                                                (t, 1),
                                            ],)
        object_fifo_link(Wk_l3l2_fifos, Wk_l2l1_fifos, [], [kv_matmul_dims[1] * kv_matmul_dims[2] * i for i in range(n_aie_rows_projs)])

        for row in range(n_aie_rows_projs):
            kv_l1l2_fifos[row] = object_fifo(f"k_L1L2_{row}", core_tiles[start_row_k + row][0], mem_tiles[0], fifo_depth, kv_l1_ty,)
        k_l2l3_fifos = object_fifo(f"k_L2L3", mem_tiles[0], shim_tiles[0], fifo_depth, kv_l2_ty,
                                    [
                                       (kv_matmul_dims[0] // r, r * kv_matmul_dims[2]),
                                       (r, t),
                                       (kv_matmul_dims[2] // t, r * t),
                                       (t, 1),
                                    ],)
        object_fifo_link(kv_l1l2_fifos, k_l2l3_fifos, [kv_matmul_dims[0] * kv_matmul_dims[2] * i for i in range(n_aie_rows_projs)], [])

        for row in range(n_aie_rows_projs):
            @core(core_tiles[start_row_k + row][0], f"mha_mm_{kv_matmul_dims[0]}x{kv_matmul_dims[1]}x{kv_matmul_dims[2]}.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(N // kv_matmul_dims[2] // n_aie_rows_projs):
                        elem_k = kv_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                        zero_kv(elem_k)
                        for _ in range_(K // kv_matmul_dims[1]):
                            elem_in_xk = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wk = Wk_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                            matmul_kv(elem_in_xk, elem_in_wk, elem_k)
                            Wk_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                            X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        kv_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[((M * K) + (K * N),), np.dtype[dtype_in]], # Order: X, W_K
            np.ndarray[(M * N,), np.dtype[dtype_in]], # Not used
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
                # One iteration generates the output for 32 rows, so need to repeat
                # for the rest of the output's sequence
                for row_offset in range(0, M, kv_matmul_dims[0]):
                    npu_dma_memcpy_nd(
                        metadata=X_l3l2_fifos,
                        bd_id=1,
                        mem=A,
                        offsets=[0, 0, 0, row_offset * K],
                        sizes=[N // kv_matmul_dims[2] // n_aie_rows_projs, K // kv_matmul_dims[1], kv_matmul_dims[0], kv_matmul_dims[1]],
                        strides=[0, kv_matmul_dims[1], K, 1],
                    )

                    for col_offset in range(0, N, kv_matmul_dims[2] * n_aie_rows_projs):
                        npu_dma_memcpy_nd(
                            metadata=Wk_l3l2_fifos,
                            bd_id=3,
                            mem=A,
                            offsets=[0, 0, 0, M * K + col_offset],
                            sizes=[K // kv_matmul_dims[1], n_aie_rows_projs, kv_matmul_dims[1], kv_matmul_dims[2]],
                            strides=[kv_matmul_dims[1] * N, kv_matmul_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=k_l2l3_fifos,
                            bd_id=6,
                            mem=C,
                            offsets=[0, 0, 0, row_offset * N + col_offset],
                            sizes=[1, n_aie_rows_projs, kv_matmul_dims[0], kv_matmul_dims[2]],
                            strides=[0, kv_matmul_dims[2], N, 1],
                        )
                        dma_wait(k_l2l3_fifos)


if __name__ == "__main__":
    main()
