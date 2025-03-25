#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys
import argparse

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.dialects._arith_ops_gen import AddIOp
import aie.dialects.index as index_dialect
import aie.dialects.arith as arith_dialect
import aie.dialects.memref as memref_dialect
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from ml_dtypes import bfloat16 

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

def get_memref_len_elems(memref):
    out = 1
    for s in memref.shape:
        out *= s
    return out

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out", type=str, choices=["bf16", "i16", "f32", "i32"], default="i16"
    )
    argparser.add_argument(
        "--trace_size", type=int, default=0, help="Size of the trace buffer"
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(
            args.M,
            args.K,
            args.m,
            args.k,
            args.dtype_in,
            args.dtype_out,
            args.trace_size,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, m, k, dtype_in_str, dtype_out_str, trace_size):
    enableTrace = trace_size > 0

    # NOTE: The matrix_vector example in the programming_examples directory doesn't
    # work with n_aie_rows > 1. Have to try to get this to work with n_aie_rows > 1 
    # after fixing n_aie_rows = 1

    # FIXME vectorized kernel is currently erroneous
    vectorized = True

    A_sz = M * K
    B_sz = K
    C_sz = M
    m_x_K = m * K

    dtype_in = np.dtype[dtype_map[dtype_in_str]]
    dtype_out = np.dtype[dtype_map[dtype_out_str]]
    assert dtype_in == np.dtype[dtype_map["bf16"]]
    assert dtype_out == np.dtype[dtype_map["f32"]]

    if dtype_in_str == "bf16":
        r = 4
        s = 8
    elif dtype_in_str == "i16":
        r = 4
        s = 4

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % m == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    assert m % r == 0
    assert k % s == 0

    M_div_m = M // m
    K_div_k = K // k

    @device(AIEDevice.npu1_4col)
    def device_body():
        inA_ty = np.ndarray[(m * k,), dtype_in]
        inB_ty = np.ndarray[(k,), dtype_in]
        outC_ty = np.ndarray[(m,), dtype_out]
        A_ty = np.ndarray[(m, k), dtype_in]
        rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

        # AIE Core Function declarations
        func_type = "vectorized" if vectorized else "scalar"
        zero = external_func(f"zero_{func_type}_{dtype_out_str}", inputs=[outC_ty])
        gemv = external_func(f"gemv_{func_type}_{dtype_in_str}_{dtype_out_str}", inputs=[A_ty, inB_ty, outC_ty])

        # Tile declarations as tile[row][col]
        tiles = [tile(0, row) for row in range(0, 6)]
        shim_tile = tiles[0]
        mem_tile = tiles[1]
        core_tile = tiles[2]

        # RTP index 0: "ready" signal
        # RTP index 1: K // k // 2
        # RTP index 2: Number of tiles per core
        rtp_buf = buffer(
            core_tile,
            datatype=rtp_ty,
            name=f"rtp",
            use_write_rtp=True,
        )

        # AIE-array data movement with object fifos
        A_l3l2_fifos = [None] 
        A_l2l1_fifos = [None]
        B_l3l1_fifos = [None] 
        C_l1l3_fifos = [None]

        # Input A, L2 -> L1
        A_l2l1_fifos[0] = object_fifo(
                        f"inA{0}",
                        mem_tile,
                        core_tile,
                        2,
                        A_ty,
                        (
                            [
                                (k // 2 // 2, 2),
                                (m, k),
                                (2, 1),
                            ]
                            if vectorized
                            else []
                        ),  # transpose at 4-byte (2xbf16) granularity
                    )

        # Input A, L3 -> L2
        A_l3l2_fifos[0] = object_fifo(f"memA{0}", shim_tile, mem_tile, 2, inA_ty)
        object_fifo_link(A_l3l2_fifos[0], A_l2l1_fifos[0])

        # Input B, L3 -> L1
        B_l3l1_fifos[0] = object_fifo("inB", shim_tile, core_tile, 2, inB_ty)

        # Output C, L1 -> L3
        C_l1l3_fifos[0] = object_fifo(f"outC{0}", core_tile, shim_tile, 2, outC_ty)

        # Set up compute tiles
        @core(core_tile, f"mv_{m}x{k}.o")
        def core_body():
            for _ in range_(0xFFFFFFFF):
                c_1 = index_dialect.constant(1)
                # while_M_div_m = WhileOp(results_=[T.index()], inits=[c_1])
                # while_M_div_m.regions[0].blocks.append(while_M_div_m.operands[0].type)
                # while_M_div_m.regions[1].blocks.append(while_M_div_m.operands[0].type)
                # with InsertionPoint(while_M_div_m.regions[0].blocks[0]):
                elem_out = C_l1l3_fifos[0].acquire(
                    ObjectFifoPort.Produce,
                    1,
                )
                zero(elem_out)

                while_K_div_k = WhileOp(results_=[T.index()], inits=[c_1])
                while_K_div_k.regions[0].blocks.append(while_K_div_k.operands[0].type)
                while_K_div_k.regions[1].blocks.append(while_K_div_k.operands[0].type)
                with InsertionPoint(while_K_div_k.regions[0].blocks[0]):
                    elem_in_a = A_l2l1_fifos[0].acquire(ObjectFifoPort.Consume, 1)
                    elem_in_b = B_l3l1_fifos[0].acquire(ObjectFifoPort.Consume, 1)
                    gemv(elem_in_a, elem_in_b, elem_out)
                    A_l2l1_fifos[0].release(ObjectFifoPort.Consume, 1)
                    B_l3l1_fifos[0].release(ObjectFifoPort.Consume, 1)

                    total_col_tiles_i32 = rtp_buf[1] 
                    total_col_tiles = index_dialect.castu(T.index(), total_col_tiles_i32)
                    add_col_tile_iter = AddIOp(lhs=while_K_div_k.regions[0].blocks[0].arguments[0], rhs=c_1)
                    next_col_tile_iter = add_col_tile_iter.results[0]
                    exit_col_tile_cond = index_dialect.cmp("slt", next_col_tile_iter, total_col_tiles) # Proceed to the "after" region if true
                    ConditionOp(condition=exit_col_tile_cond, args=[next_col_tile_iter])
                with InsertionPoint(while_K_div_k.regions[1].blocks[0]):
                    yield_([while_K_div_k.regions[1].blocks[0].arguments[0]])
                    
                C_l1l3_fifos[0].release(ObjectFifoPort.Produce, 1)
                #     total_row_tiles_i32 = rtp_buf[0] 
                #     total_row_tiles = index_dialect.castu(T.index(), total_row_tiles_i32)
                #     add_row_tile_iter = AddIOp(lhs=while_M_div_m.regions[0].blocks[0].arguments[0], rhs=c_1)
                #     next_row_tile_iter = add_row_tile_iter.results[0]
                #     exit_row_tile_cond = index_dialect.cmp("slt", next_row_tile_iter, total_row_tiles) # Proceed to the "after" region if true
                #     ConditionOp(condition=exit_row_tile_cond, args=[next_row_tile_iter])
                # with InsertionPoint(while_M_div_m.regions[1].blocks[0]):
                #     yield_([while_M_div_m.regions[1].blocks[0].arguments[0]])

        # tiles_to_trace = [core_tiles[0][0]]
        # if enableTrace:
        #     trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tiles[0])

        # To/from AIE-array data movement
        @runtime_sequence(
                np.ndarray[(A_sz,), dtype_in],
                np.ndarray[(B_sz,), dtype_in],
                np.ndarray[(C_sz,), dtype_out],
            )
        def sequence(A, B, C):
            # if enableTrace:
            #     trace_utils.configure_packet_tracing_aie2(
            #         tiles_to_trace, shim_tiles[0], trace_size, 4096 * 4
            #     )

            # Write number of inner loop iterations for cores to use as run-time parameter.
            # This allows for processing different problem sizes by only swapping the insts.txt.
            rtp_M_div_m= M_div_m
            rtp_K_div_k = K_div_k
            rtp_buf[0] = rtp_M_div_m
            rtp_buf[1] = rtp_K_div_k

            for row_tile in range(M_div_m):
                npu_dma_memcpy_nd(
                    metadata=B_l3l1_fifos[0],
                    bd_id=2,
                    mem=B,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, K_div_k, 1, k],
                    strides=[0, k, 0, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=A_l3l2_fifos[0],
                    bd_id=1,
                    mem=A,
                    offsets=[0, 0, 0, row_tile * m * K],
                    sizes=[1, K_div_k, m, k],
                    strides=[0, k, K, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=C_l1l3_fifos[0],
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, row_tile * m],
                    sizes=[1, 1, 1, m],
                    strides=[0, 0, 0, 1],
                )
                dma_wait(*C_l1l3_fifos)

if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
