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
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
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
        maybe_taps = my_addandnorm(
            args.dev,
            args.M,
            args.N,
            args.m,
            args.n,
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


def my_addandnorm(
    dev,
    M,
    N,
    m,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    trace_size,
    generate_taps=False,
):
    # Using 2 rows because there's not enough channels to use 4 cores and 3 cores makes the tiling uneven. 
    # But Layernorm is much faster than the other transformer operations even with 2 rows, so will likely 
    # move to 1 row at some point.
    n_aie_rows = 2 
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    # npu is a 4 row x 4 col array
    if dev == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    # npu2 is a 4 row x 8 col array
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols

    if dev == "npu":
        if n_aie_cols == 1:
            dev_ty = AIEDevice.npu1
    else:
        if n_aie_cols == 1:
            dev_ty = AIEDevice.npu2

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    @device(dev_ty)
    def device_body():
        in_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_in]]
        in_l1_ty = np.ndarray[(m, n), np.dtype[dtype_in]]
        out_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
        out_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        # AIE Core Function declarations
        add = external_func(f"eltwise_add_{dtype_in_str}_vector", inputs=[in_l1_ty, in_l1_ty, in_l1_ty])
        norm = external_func(f"layer_norm_{dtype_in_str}_vector", inputs=[in_l1_ty, out_l1_ty])

        # Tile declarations as tile[row][col]
        if dev == "npu":
            tiles = [
                [tile(2, row)] for row in range(0, 6) # 3rd column only
            ]
        else:
            tiles = [
                [tile(7, row)] for row in range(0, 6) # 8th column only
            ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # AIE-array data movement with object fifos
        A_l3l2_fifos = [None] * n_aie_cols
        A_l2l1_fifos = [None] * n_aie_rows

        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_rows

        C_l1_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols

        # Input A
        for row in range(n_aie_rows):
            A_l2l1_fifos[row] = object_fifo(
                f"A_L2L1_{row}",
                mem_tiles[row // n_A_tiles_per_shim],
                core_tiles[row][0:n_aie_cols],  # broadcast along one row
                fifo_depth,
                in_l1_ty,
            )
        for col in range(n_aie_cols):
            A_l3l2_fifos[col] = object_fifo(
                f"A_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                in_l2_ty,
            )
            # If n_cols == n_rows, n_A_tiles_per_shim is 1 and
            # this simply links a_l3l2_fifos[col] to a_l2l1_fifos[row] directly,
            # where col == row.
            # If n_cols < n_rows, each column receives multiple rows of
            # tiles; distribute it along rows of AIE cores.
            start_row = col * n_A_tiles_per_shim
            stop_row = start_row + n_A_tiles_per_shim
            if stop_row - start_row > 1:
                of_offsets = [m * n * i for i in range(stop_row - start_row)]
            else:
                of_offsets = []
            object_fifo_link(
                A_l3l2_fifos[col],
                [A_l2l1_fifos[row] for row in range(start_row, stop_row)],
                [],
                of_offsets,
            )

        # Input B
        for row in range(n_aie_rows):
            B_l2l1_fifos[row] = object_fifo(
                f"B_L2L1_{row}",
                mem_tiles[row // n_A_tiles_per_shim],
                core_tiles[row][0:n_aie_cols],  # broadcast along one row
                fifo_depth,
                in_l1_ty,
            )
        for col in range(n_aie_cols):
            B_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                in_l2_ty,
            )
            start_row = col * n_A_tiles_per_shim
            stop_row = start_row + n_A_tiles_per_shim
            if stop_row - start_row > 1:
                of_offsets = [m * n * i for i in range(stop_row - start_row)]
            else:
                of_offsets = []
            object_fifo_link(
                B_l3l2_fifos[col],
                [B_l2l1_fifos[row] for row in range(start_row, stop_row)],
                [],
                of_offsets,
            )

        # Output C
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1_fifos[row][col] = object_fifo(
                    f"C_L1_{col}_{row}",
                    core_tiles[row][col],
                    core_tiles[row][col],
                    fifo_depth,
                    in_l1_ty,
                )
                C_l1l2_fifos[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    fifo_depth,
                    out_l1_ty,
                )
            C_l2l3_fifos[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col],
                shim_tiles[col],
                fifo_depth,
                out_l2_ty,
            )
            if n_aie_rows > 1:
                of_offsets = [m * n * i for i in range(n_aie_rows)]
            else:
                of_offsets = []
            object_fifo_link(
                [C_l1l2_fifos[j][col] for j in range(n_aie_rows)],
                C_l2l3_fifos[col],
                of_offsets,
                [],
            )  # join along one column


        # Set up compute tiles
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                @core(core_tiles[row][col], f"addandnorm_{m}x{n}.o", stack_size=0xD00)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        loop = (
                            range_(n_tiles_per_core)
                            if n_tiles_per_core > 1
                            else range(1)
                        )  # Workaround for issue #1547
                        for _ in loop:
                            elem_in_a = A_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = B_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                            elem_c = C_l1_fifos[row][col].acquire(ObjectFifoPort.Produce, 1)
                            add(elem_in_a, elem_in_b, elem_c)
                            A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                            B_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                            C_l1_fifos[row][col].release(ObjectFifoPort.Produce, 1)

                            elem_c = C_l1_fifos[row][col].acquire(ObjectFifoPort.Consume, 1)
                            elem_out = C_l1l2_fifos[row][col].acquire(ObjectFifoPort.Produce, 1)
                            # TODO: Split up the LayerNorm into a step to generate
                            # the sums of each row and then a step to normalize.
                            # This would remove the need to tile across the whole
                            # workload's rows.
                            norm(elem_c, elem_out)
                            C_l1_fifos[row][col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2_fifos[row][col].release(ObjectFifoPort.Produce, 1)


        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[(M * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            # for row_tile in range(M // m // n_aie_rows):
            #     for col_tile in range(N // n // n_aie_cols):
            for col in range(n_aie_cols):
                # C Output Transfer:
                # The smallest transfer unit is a (m*n_aie_rows)-x-(n)-sized sub-tile of the matrix.
                # Transfer one such tile for every (n_aie_cols)-th column, evenly spaced,
                # then repeat that (tb_n_rows) times for the next contiguous blocks of rows.
                # Each shim will start at a different column offset, transferring interleaved
                # columns. For example, shim 0 may transfer the blocks marked 0 below, and shim 1
                # may transfer the blocks marked 1.
                #
                #             N
                #      ----------------
                #     |0011    0011    |
                #     |0011    0011    |
                #     |0011    0011    |
                # M   |0011    0011    |
                #     |                |
                #     |                |
                #     |                |
                #     |                |
                #      ----------------
                # row_offset = row_tile * m * n_aie_rows * N
                # col_offset = col_tile * n + col * n
                # offset = col_offset + row_offset
                offset = 0
                sizes = [M // m // n_aie_rows, N // n // n_aie_cols, m * n_aie_rows, n * n_aie_cols]
                strides = [m * n_aie_rows * N, n * n_aie_cols, N, 1]
                npu_dma_memcpy_nd(
                    metadata=C_l2l3_fifos[col],
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, offset],
                    sizes=sizes,
                    strides=strides,
                )

                # A input transfer:
                npu_dma_memcpy_nd(
                    metadata=A_l3l2_fifos[col],
                    bd_id=1,
                    mem=A,
                    offsets=[0, 0, 0, offset],
                    sizes=sizes,
                    strides=strides,
                )

                # B input transfer:
                npu_dma_memcpy_nd(
                    metadata=B_l3l2_fifos[col],
                    bd_id=2,
                    mem=B,
                    offsets=[0, 0, 0, offset],
                    sizes=sizes,
                    strides=strides,
                )
                dma_wait(*C_l2l3_fifos)


if __name__ == "__main__":
    main()
