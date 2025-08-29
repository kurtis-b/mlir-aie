# passthrough_pykernel/passthrough_pykernel_placed.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.func import func
from aie.helpers.dialects.ext.scf import _for as range_


dev = AIEDevice.npu1_1col

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))


def passthroughKernel(vector_size):
    N = vector_size
    lineWidthInBytes = N // 4  # chop input in 4 sub-tensors

    @device(dev)
    def device_body():
        # define types
        line_ty = np.ndarray[(lineWidthInBytes,), np.dtype[np.uint16]]

        # AIE Core Python Function declarations
        @func(emit=True)
        def passThroughLine(input: line_ty, output: line_ty, lineWidth: np.int32):
            for i in range_(lineWidth):
                output[i] = input[i]

        @func(emit=True)
        def passThroughLine1(input: np.ndarray[(lineWidthInBytes // 2,), np.dtype[np.uint16]], output: np.ndarray[(lineWidthInBytes // 2,), np.dtype[np.uint16]], lineWidth: np.int32):
            for i in range_(lineWidth):
                output[i] = input[i]

        # Tile declarations
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)

        # AIE-array data movement with object fifos
        of_in1 = object_fifo("in1", ShimTile, MemTile, [4, 4], np.ndarray[(lineWidthInBytes,), np.dtype[np.uint16]])
        of_in2 = object_fifo("in2", MemTile, ComputeTile2, [4, 4], np.ndarray[(lineWidthInBytes,), np.dtype[np.uint16]])
        of_out1 = object_fifo("out1", ComputeTile2, MemTile, [4, 4], np.ndarray[(lineWidthInBytes,), np.dtype[np.uint16]])
        of_out2 = object_fifo("out2", MemTile, ComputeTile3, [8, 4], np.ndarray[(lineWidthInBytes // 2,), np.dtype[np.uint16]])
        of_out3 = object_fifo("out3", ComputeTile3, MemTile, [4, 4], np.ndarray[(lineWidthInBytes // 2,), np.dtype[np.uint16]])
        of_out4 = object_fifo("out4", MemTile, ShimTile, [2, 2], np.ndarray[(lineWidthInBytes,), np.dtype[np.uint16]])
        object_fifo_link(of_in1, of_in2)
        object_fifo_link(of_out1, of_out2)
        object_fifo_link(of_out3, of_out4)

        # Set up compute tiles
        # By changing the array dimensions in out2, it seems like we essentially have an implicit offset when out1 produces data
        # and out2 consumes it. This also seems to be the case when out3 produces data and out4 consumes it.
        # The data movement here is in 2 stages:
        # Stage 1: Get 4 tiles of lineWidthInBytes from shim to compute tile 2 and send all 4 tiles to mem tile
        # Stage 2: Get 2 tiles of lineWidthInBytes from the same mem tile and send 1 tile of lineWidthInBytes to shim for 2 iterations

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            for _ in range_(sys.maxsize): 
                # This seems to fill the 8 objfifos in out2
                elemOut = of_out1.acquire(ObjectFifoPort.Produce, 4)
                elemIn = of_in2.acquire(ObjectFifoPort.Consume, 4)
                passThroughLine(elemIn[0], elemOut[0], lineWidthInBytes)
                passThroughLine(elemIn[1], elemOut[1], lineWidthInBytes)
                passThroughLine(elemIn[2], elemOut[2], lineWidthInBytes)
                passThroughLine(elemIn[3], elemOut[3], lineWidthInBytes)
                of_in2.release(ObjectFifoPort.Consume, 4)
                of_out1.release(ObjectFifoPort.Produce, 4)

                # # Generating the data in 4 iterations works too to
                # # provide of_out2 with data for 4 buffers 2 times
                # # Comment the above before uncommenting below
                # elemOut = of_out1.acquire(ObjectFifoPort.Produce, 1)
                # elemIn = of_in2.acquire(ObjectFifoPort.Consume, 1)
                # passThroughLine(elemIn, elemOut, lineWidthInBytes)
                # of_in2.release(ObjectFifoPort.Consume, 1)
                # of_out1.release(ObjectFifoPort.Produce, 1)

        # Compute tile 3
        @core(ComputeTile3)
        def core_body():
            for _ in range_(sys.maxsize): 
                # # This seems to write the correct data at the correct offset implicitly
                elemIn = of_out2.acquire(ObjectFifoPort.Consume, 4)
                elemOut = of_out3.acquire(ObjectFifoPort.Produce, 2)
                passThroughLine1(elemIn[0], elemOut[0], lineWidthInBytes // 2)
                passThroughLine1(elemIn[1], elemOut[1], lineWidthInBytes // 2)
                of_out3.release(ObjectFifoPort.Produce, 2)
                elemOut = of_out3.acquire(ObjectFifoPort.Produce, 2)
                passThroughLine1(elemIn[2], elemOut[0], lineWidthInBytes // 2)
                passThroughLine1(elemIn[3], elemOut[1], lineWidthInBytes // 2)
                of_out3.release(ObjectFifoPort.Produce, 2)
                of_out2.release(ObjectFifoPort.Consume, 4)
        #    print(ctx.module.operation.verify())

        vector_ty = np.ndarray[(N,), np.dtype[np.uint16]]

        @runtime_sequence(vector_ty, vector_ty, vector_ty)
        def sequence(inTensor, outTensor, notUsed):
            in_task = shim_dma_single_bd_task(
                of_in1, inTensor, sizes=[1, 1, 1, N], issue_token=True
            )
            out_task = shim_dma_single_bd_task(
                of_out4, outTensor, sizes=[1, 1, 1, N], issue_token=True
            )

            dma_start_task(in_task, out_task)
            dma_await_task(in_task, out_task)


try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    passthroughKernel(vector_size)
    print(ctx.module)
