//===- basic.mlir ----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memtile-row-stores %s | FileCheck %s

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

// CHECK-LABEL: module {
// CHECK: aie.device(xcve2302) {
// CHECK: %[[MEM:.*]] = aie.tile(2, 1)
// CHECK: %[[CORE:.*]] = aie.tile(2, 2)
// CHECK-NOT: aie.memtile_row_store
// CHECK-DAG: %[[SRC:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_src"} : memref<32x96xbf16>
// CHECK-DAG: %[[DST:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_dst"} : memref<32x96xbf16>
// CHECK-DAG: %[[ROW:.*]] = aie.buffer(%[[MEM]]) {sym_name = "row_store0_row"} : memref<12288xbf16>
// CHECK-DAG: %[[SRC_EMPTY:.*]] = aie.lock(%[[CORE]]) {init = 1 : i32, sym_name = "row_store0_src_empty"}
// CHECK-DAG: %[[SRC_FULL:.*]] = aie.lock(%[[CORE]]) {init = 0 : i32, sym_name = "row_store0_src_full"}
// CHECK-DAG: %[[DST_EMPTY:.*]] = aie.lock(%[[CORE]]) {init = 1 : i32, sym_name = "row_store0_dst_empty"}
// CHECK-DAG: %[[DST_FULL:.*]] = aie.lock(%[[CORE]]) {init = 0 : i32, sym_name = "row_store0_dst_full"}
// CHECK-DAG: %[[ROW_EMPTY:.*]] = aie.lock(%[[MEM]]) {init = 1 : i32, sym_name = "row_store0_row_empty"}
// CHECK-DAG: %[[ROW_FULL:.*]] = aie.lock(%[[MEM]]) {init = 0 : i32, sym_name = "row_store0_row_full"}
// CHECK: aie.flow(%[[CORE]], DMA : 0, %[[MEM]], DMA : 0)
// CHECK: aie.flow(%[[MEM]], DMA : 1, %[[CORE]], DMA : 0)
// CHECK: aie.core(%[[CORE]]) {
// CHECK: aie.use_lock(%[[SRC_EMPTY]], AcquireGreaterEqual, 1)
// CHECK-NEXT: aie.use_lock(%[[SRC_FULL]], Release, 1)
// CHECK-NEXT: aie.use_lock(%[[DST_FULL]], AcquireGreaterEqual, 1)
// CHECK-NEXT: aie.use_lock(%[[DST_EMPTY]], Release, 1)
// CHECK-NEXT: aie.end
// CHECK: aie.mem(%[[CORE]]) {
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%[[SRC_FULL]], AcquireGreaterEqual, 1)
// CHECK: aie.dma_bd(%[[SRC]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.use_lock(%[[SRC_EMPTY]], Release, 1)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.use_lock(%[[DST_EMPTY]], AcquireGreaterEqual, 1)
// CHECK: aie.dma_bd(%[[DST]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.use_lock(%[[DST_FULL]], Release, 1)
// CHECK: aie.memtile_dma(%[[MEM]]) {
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.use_lock(%[[ROW_EMPTY]], AcquireGreaterEqual, 1)
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 3072, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 6144, 3072)
// CHECK: aie.dma_bd(%[[ROW]] : memref<12288xbf16>, 9216, 3072)
// CHECK: aie.use_lock(%[[ROW_FULL]], Release, 1)
// CHECK: aie.dma_start(MM2S, 1
// CHECK: aie.use_lock(%[[ROW_FULL]], AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%[[ROW_EMPTY]], Release, 1)
