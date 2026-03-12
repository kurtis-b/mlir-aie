//===- memtile_row_store_encoder_pipeline_o_proj_like_finer_tiles.mlir -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Same total memtile row footprint as the part_count=8 encoder-style test, but
// split into more parts.
//
// RUN: aie-opt --aie-lower-memtile-row-stores --aie-assign-bd-ids %s | FileCheck %s

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
// CHECK: %[[ROW:.*]] = aie.buffer(%{{.*}}) {sym_name = "o_proj_row_store_row"} : memref<24576xbf16>
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 0, 1536) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 1536, 1536) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 3072, 1536) {bd_id = 2 : i32, next_bd_id = 3 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 21504, 1536) {bd_id = 14 : i32, next_bd_id = 15 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 23040, 1536) {bd_id = 15 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 0, 1536) {bd_id = 24 : i32, next_bd_id = 25 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 1536, 1536) {bd_id = 25 : i32, next_bd_id = 26 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 3072, 1536) {bd_id = 26 : i32, next_bd_id = 27 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 21504, 1536) {bd_id = 38 : i32, next_bd_id = 39 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 23040, 1536) {bd_id = 39 : i32, next_bd_id = 24 : i32}

