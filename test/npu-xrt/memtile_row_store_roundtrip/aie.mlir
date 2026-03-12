//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    %input = aie.buffer(%tile_0_2) {sym_name = "input"} : memref<2xi32>
    %output = aie.buffer(%tile_0_2) {sym_name = "output"} : memref<2xi32>
    %input_empty = aie.lock(%tile_0_2) {init = 1 : i32, sym_name = "input_empty"}
    %input_full = aie.lock(%tile_0_2) {init = 0 : i32, sym_name = "input_full"}
    %output_empty = aie.lock(%tile_0_2) {init = 1 : i32, sym_name = "output_empty"}
    %output_full = aie.lock(%tile_0_2) {init = 0 : i32, sym_name = "output_full"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    aie.memtile_row_store @row_store(%tile_0_2, %tile_0_1) {
      part_count = 8 : i32,
      compute_mm2s_channel = 1 : i32,
      compute_s2mm_channel = 1 : i32
    } : memref<2xi32>

    %mem_0_2 = aie.mem(%tile_0_2) {
      %inDma = aie.dma_start(S2MM, 0, ^input_bd, ^out_dma)
    ^input_bd:  // 2 preds: ^bb0, ^input_bd
      aie.use_lock(%input_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%input : memref<2xi32>, 0, 2)
      aie.use_lock(%input_full, Release, 1)
      aie.next_bd ^input_bd
    ^out_dma:  // pred: ^bb0
      %outDma = aie.dma_start(MM2S, 0, ^output_bd, ^end)
    ^output_bd:  // 2 preds: ^out_dma, ^output_bd
      aie.use_lock(%output_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%output : memref<2xi32>, 0, 2)
      aie.use_lock(%output_empty, Release, 1)
      aie.next_bd ^output_bd
    ^end:  // pred: ^out_dma
      aie.end
    }

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c100 = arith.constant 100 : i32
      %c1_i32 = arith.constant 1 : i32

      affine.for %part = 0 to 8 {
        aie.use_lock(%input_full, AcquireGreaterEqual, 1)
        %src = aie.memtile_row_store.acquire @row_store(Produce) : memref<2xi32>
        affine.for %i = 0 to 2 {
          %v = memref.load %input[%i] : memref<2xi32>
          %w = arith.addi %v, %c100 : i32
          memref.store %w, %src[%i] : memref<2xi32>
        }
        aie.memtile_row_store.release @row_store(Produce)
        aie.use_lock(%input_empty, Release, 1)
      }

      affine.for %part = 0 to 8 {
        %dst = aie.memtile_row_store.acquire @row_store(Consume) : memref<2xi32>
        aie.use_lock(%output_empty, AcquireGreaterEqual, 1)
        affine.for %i = 0 to 2 {
          %v = memref.load %dst[%i] : memref<2xi32>
          %w = arith.addi %v, %c1_i32 : i32
          memref.store %w, %output[%i] : memref<2xi32>
        }
        aie.use_lock(%output_full, Release, 1)
        aie.memtile_row_store.release @row_store(Consume)
      }
      aie.end
    }

    aie.shim_dma_allocation @data_in (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @data_out (%tile_0_0, S2MM, 0)
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c16_i64 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd (%arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c16_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 0 : i64, metadata = @data_in} : memref<16xi32>
      aiex.npu.dma_memcpy_nd (%arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c1_i64, %c16_i64][%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {id = 1 : i64, metadata = @data_out, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @data_out}
    }
  }
}
