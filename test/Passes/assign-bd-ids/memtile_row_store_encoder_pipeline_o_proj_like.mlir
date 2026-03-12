//===- memtile_row_store_encoder_pipeline_o_proj_like.mlir -----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Models the encoder_pipeline O-proj to LN1 handoff when proj_acc_depth = 8.
//
// RUN: aie-opt --aie-lower-memtile-row-stores --aie-assign-bd-ids %s | FileCheck %s

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
// CHECK: %[[SRC:.*]] = aie.buffer(%{{.*}}) {sym_name = "o_proj_row_store_src"} : memref<32x96xbf16>
// CHECK: %[[DST:.*]] = aie.buffer(%{{.*}}) {sym_name = "o_proj_row_store_dst"} : memref<32x96xbf16>
// CHECK: %[[ROW:.*]] = aie.buffer(%{{.*}}) {sym_name = "o_proj_row_store_row"} : memref<24576xbf16>
// CHECK: aie.dma_bd(%[[SRC]] : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%[[DST]] : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 3072, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 6144, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 9216, 3072) {bd_id = 3 : i32, next_bd_id = 4 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 12288, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 15360, 3072) {bd_id = 5 : i32, next_bd_id = 6 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 18432, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 21504, 3072) {bd_id = 7 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 3072, 3072) {bd_id = 25 : i32, next_bd_id = 26 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 6144, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 9216, 3072) {bd_id = 27 : i32, next_bd_id = 28 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 12288, 3072) {bd_id = 28 : i32, next_bd_id = 29 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 15360, 3072) {bd_id = 29 : i32, next_bd_id = 30 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 18432, 3072) {bd_id = 30 : i32, next_bd_id = 31 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<24576xbf16>, 21504, 3072) {bd_id = 31 : i32, next_bd_id = 24 : i32}
