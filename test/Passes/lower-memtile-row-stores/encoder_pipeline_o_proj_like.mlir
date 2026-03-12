//===- encoder_pipeline_o_proj_like.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Models the encoder_pipeline O-proj to LN1 handoff when proj_acc_depth = 8.
//
// RUN: aie-opt --aie-lower-memtile-row-stores %s | FileCheck %s

module {
  aie.device(npu2) {
    %mem = aie.tile(2, 1)
    %core = aie.tile(2, 5)

    aie.memtile_row_store @o_proj_row_store(%core, %mem) {part_count = 8 : i32} : memref<32x96xbf16>

    aie.core(%core) {
      %src = aie.memtile_row_store.acquire @o_proj_row_store(Produce) : memref<32x96xbf16>
      aie.memtile_row_store.release @o_proj_row_store(Produce)
      %dst = aie.memtile_row_store.acquire @o_proj_row_store(Consume) : memref<32x96xbf16>
      aie.memtile_row_store.release @o_proj_row_store(Consume)
      aie.end
    }
  }
}

// CHECK-LABEL: module {
// CHECK: aie.device(npu2) {
// CHECK: %[[MEM:.*]] = aie.tile(2, 1)
// CHECK: %[[CORE:.*]] = aie.tile(2, 5)
// CHECK: %[[ROW:.*]] = aie.buffer(%[[MEM]]) {sym_name = "o_proj_row_store_row"} : memref<24576xbf16>
// CHECK: aie.flow(%[[CORE]], DMA : 0, %[[MEM]], DMA : 0)
// CHECK: aie.flow(%[[MEM]], DMA : 1, %[[CORE]], DMA : 0)
// CHECK: aie.core(%[[CORE]]) {
// CHECK: aie.use_lock(%{{.*}}_src_empty, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_src_full, Release, 1)
// CHECK: aie.use_lock(%{{.*}}_dst_full, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}_dst_empty, Release, 1)
// CHECK: aie.mem(%[[CORE]]) {
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.memtile_dma(%[[MEM]]) {
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 3072, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 6144, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 9216, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 12288, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 15360, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 18432, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 21504, 3072)
// CHECK: aie.dma_start(MM2S, 1
