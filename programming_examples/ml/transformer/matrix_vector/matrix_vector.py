#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import sys
import argparse
import numpy as np

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Vector MLIR Design",
        description="Emits MLIR code for a matrix vector design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["i8", "bf16", "i16"], default="i16"
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

# TODOs:
# 1) Modify external memory transfer to a loop based on the tiling
# 2) Extend support for data type combinations other than i16/i32 in design and kernel
#    - Make sure to adjust the combinations in the kernel code
# 3) Add support for more than 1 core
def my_matmul(M, K, m, k, dtype_in_str, dtype_out_str, trace_size):
    enableTrace = trace_size > 0

    n_cores = 1

    A_sz = M * K
    B_sz = K
    C_sz = M
    C_sz_div_n_cores = C_sz // n_cores

    M_div_m = M // m
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    m_x_k = m * k
    m_x_K = m * K

    vectorized = True

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]
    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"
    
    @device(AIEDevice.npu1_4col)
    def device_body():
        inA_ty = np.ndarray[(m * k,), np.dtype[dtype_in]]
        inB_ty = np.ndarray[(k,), np.dtype[dtype_in]]
        outC_ty = np.ndarray[(m,), np.dtype[dtype_out]]
        A_ty = np.ndarray[(m, k), np.dtype[dtype_in]]

        # AIE Core Function declarations
        func_type = "vectorized" if vectorized else "scalar"
        zero = external_func(f"zero_{func_type}_{dtype_out_str}", inputs=[outC_ty])
        matvec = external_func(
            f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
            inputs=[A_ty, inB_ty, outC_ty],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile0 = tile(0, 2)
        ComputeTile1 = tile(0, 3)
        ComputeTile2 = tile(0, 4)
        ComputeTile3 = tile(0, 5)
        cores = [ComputeTile0, ComputeTile1, ComputeTile2, ComputeTile3]
        memA_fifos = []
        inA_fifos = []
        memB_fifos = []
        inB_fifos = []
        outC_fifos = []
        memC_fifos = []

        # AIE-array data movement with object fifos
        # Input A
        for i in range(n_cores):
            memA_fifos.append(
                object_fifo(f"memA{i}", ShimTile, MemTile, 2, inA_ty)
            )
            inA_fifos.append(
                object_fifo(
                    f"inA{i}",
                    MemTile,
                    cores[i],
                    2,
                    A_ty,
                    (
                        [
                            (k // 2, 2),
                            (m, k),
                            (2, 1),
                        ]
                        if vectorized
                        else []
                    ),  # transpose at 4-byte (2xbf16) granularity
                )
            )
            object_fifo_link(memA_fifos[i], inA_fifos[i])

            # Output C
            memC_fifos.append(
                object_fifo(f"outC{i}", MemTile, ShimTile, 2, outC_ty)
            )
            outC_fifos.append(
                object_fifo(f"memC{i}", cores[i], MemTile, 2, outC_ty)
            )
            object_fifo_link(outC_fifos[i], memC_fifos[i])

            # Input B
            memB_fifos.append(
                object_fifo(f"memB{i}", ShimTile, MemTile, 2, inB_ty)
            )
            inB_fifos.append(
                object_fifo(f"inB{i}", MemTile, cores[i], 2, inB_ty)
            )
            object_fifo_link(memB_fifos[i], inB_fifos[i])

        # Set up compute tiles
        for i in range(n_cores):
            # Compute tile i
            @core(cores[i], f"mv_{m}x{k}.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_out = outC_fifos[i].acquire(
                        ObjectFifoPort.Produce,
                        1,
                    )
                    zero(elem_out)

                    for _ in range_(K_div_k):
                        elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = inB_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                        matvec(elem_in_a, elem_in_b, elem_out)
                        inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                        inB_fifos[i].release(ObjectFifoPort.Consume, 1)

                    outC_fifos[i].release(ObjectFifoPort.Produce, 1)

        tiles_to_trace = [cores[0]]
        if enableTrace:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

        # To/from AIE-array data movement

        @runtime_sequence(
            np.ndarray[(A_sz,), np.dtype[dtype_in]],
            np.ndarray[(B_sz,), np.dtype[dtype_in]],
            np.ndarray[(C_sz,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            if enableTrace:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace, ShimTile, trace_size, M * 4
                )
            for i in range(n_cores):
                npu_dma_memcpy_nd(
                    metadata=memB_fifos[i],
                    bd_id=2,
                    mem=B,
                    sizes=[M_div_m_div_n_cores, 1, 1, K],
                    strides=[0, 0, 0, 1],
                )
                A_offset = i * M_div_m_div_n_cores * m * K
                C_offset = i * M_div_m_div_n_cores * m
                npu_dma_memcpy_nd(
                    metadata=memA_fifos[i],
                    bd_id=1,
                    mem=A,
                    offsets=[0, 0, 0, A_offset],
                    sizes=[M_div_m_div_n_cores, K_div_k, m, k],
                    strides=[m_x_K, k, K, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=memC_fifos[i],
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, C_offset],
                    sizes=[1, 1, 1, C_sz_div_n_cores],
                    strides=[0, 0, 0, 1],
                )
            dma_wait(*memC_fifos)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)

