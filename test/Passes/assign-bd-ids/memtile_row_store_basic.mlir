//===- memtile_row_store_basic.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memtile-row-stores --aie-assign-bd-ids %s | FileCheck %s

module {
  aie.device(xcve2302) {
    %mem = aie.tile(2, 1)
    %core = aie.tile(2, 2)

    aie.memtile_row_store @row_store0(%core, %mem) {part_count = 4 : i32} : memref<32x96xbf16>

    aie.core(%core) {
      %src = aie.memtile_row_store.acquire @row_store0(Produce) : memref<32x96xbf16>
      aie.memtile_row_store.release @row_store0(Produce)
      %dst = aie.memtile_row_store.acquire @row_store0(Consume) : memref<32x96xbf16>
      aie.memtile_row_store.release @row_store0(Consume)
      aie.end
    }
  }
}

// CHECK-LABEL: aie.device(xcve2302) {
// CHECK: %[[MEM:.*]] = aie.tile(2, 1)
// CHECK: %[[CORE:.*]] = aie.tile(2, 2)
// CHECK: %[[SRC:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_src"} : memref<32x96xbf16>
// CHECK: %[[DST:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_dst"} : memref<32x96xbf16>
// CHECK: %[[ROW:.*]] = aie.buffer(%[[MEM]]) {sym_name = "row_store0_row"} : memref<12288xbf16>
// CHECK: aie.dma_bd(%[[SRC]] : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%[[DST]] : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 3072, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 6144, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 9216, 3072) {bd_id = 3 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 0, 3072) {bd_id = 24 : i32, next_bd_id = 25 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 3072, 3072) {bd_id = 25 : i32, next_bd_id = 26 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 6144, 3072) {bd_id = 26 : i32, next_bd_id = 27 : i32}
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 9216, 3072) {bd_id = 27 : i32, next_bd_id = 24 : i32}
