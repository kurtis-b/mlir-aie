//===- encoder_pipeline_o_proj_like_finer_tiles.mlir -----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Same total memtile row footprint as encoder_pipeline_o_proj_like.mlir, but
// with more parts and smaller per-part tiles.
//
// RUN: aie-opt --aie-lower-memtile-row-stores %s | FileCheck %s

module {
  aie.device(npu2) {
    %mem = aie.tile(2, 1)
    %core = aie.tile(2, 5)

    aie.memtile_row_store @o_proj_row_store(%core, %mem) {part_count = 16 : i32} : memref<24x64xbf16>

    aie.core(%core) {
      %src = aie.memtile_row_store.acquire @o_proj_row_store(Produce) : memref<24x64xbf16>
      aie.memtile_row_store.release @o_proj_row_store(Produce)
      %dst = aie.memtile_row_store.acquire @o_proj_row_store(Consume) : memref<24x64xbf16>
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
// CHECK: aie.mem(%[[CORE]]) {
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.dma_bd(%{{.*}} : memref<24x64xbf16>, 0, 1536)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.memtile_dma(%[[MEM]]) {
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 0, 1536)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 1536, 1536)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 23040, 1536)
// CHECK: aie.dma_start(MM2S, 1
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 0, 1536)
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 23040, 1536)

