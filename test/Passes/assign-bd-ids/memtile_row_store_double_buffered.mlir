//===- memtile_row_store_double_buffered.mlir ------------------*- MLIR -*-===//
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

    aie.memtile_row_store @row_store0(%core, %mem) {part_count = 2 : i32, buffer_count = 2 : i32} : memref<32x96xbf16>

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
// CHECK: %[[ROW0:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_row_0"} : memref<6144xbf16>
// CHECK: %[[ROW1:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_row_1"} : memref<6144xbf16>
// CHECK: aie.dma_bd(%[[ROW0]] : memref<6144xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%[[ROW1]] : memref<6144xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%[[ROW0]] : memref<6144xbf16>, 0, 6144) {bd_id = 24 : i32, next_bd_id = 25 : i32}
// CHECK: aie.dma_bd(%[[ROW1]] : memref<6144xbf16>, 0, 6144) {bd_id = 25 : i32, next_bd_id = 24 : i32}
