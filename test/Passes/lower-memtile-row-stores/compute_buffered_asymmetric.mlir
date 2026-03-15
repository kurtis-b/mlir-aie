//===- compute_buffered_asymmetric.mlir ------------------------*- MLIR -*-===//
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

    aie.memtile_row_store @row_store0(%core, %mem) {
      part_count = 2 : i32,
      buffer_count = 2 : i32,
      compute_buffer_count = 1 : i32,
      compute_produce_buffer_count = 2 : i32
    } : memref<32x96xbf16>

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
// CHECK-DAG: %[[SRC0:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_src_0"} : memref<32x96xbf16>
// CHECK-DAG: %[[SRC1:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_src_1"} : memref<32x96xbf16>
// CHECK-DAG: %[[DST:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_dst"} : memref<32x96xbf16>
// CHECK-DAG: %[[IDX:.*]] = aie.buffer(%[[CORE]]) {sym_name = "row_store0_next_index"} : memref<2xi32>
// CHECK-DAG: %[[SRC0_EMPTY:.*]] = aie.lock(%[[CORE]]) {init = 1 : i32, sym_name = "row_store0_src_0_empty"}
// CHECK-DAG: %[[SRC1_EMPTY:.*]] = aie.lock(%[[CORE]]) {init = 1 : i32, sym_name = "row_store0_src_1_empty"}
// CHECK-DAG: %[[DST_EMPTY:.*]] = aie.lock(%[[CORE]]) {init = 1 : i32, sym_name = "row_store0_dst_empty"}
// CHECK-DAG: %[[DST_FULL:.*]] = aie.lock(%[[CORE]]) {init = 0 : i32, sym_name = "row_store0_dst_full"}
// CHECK: aie.core(%[[CORE]]) {
// CHECK-DAG: memref.store {{.*}}, %[[IDX]]{{\[}}%{{.*}}{{\]}} : memref<2xi32>
// CHECK-DAG: memref.store {{.*}}, %[[IDX]]{{\[}}%{{.*}}{{\]}} : memref<2xi32>
// CHECK: scf.index_switch
// CHECK: aie.use_lock(%[[SRC0_EMPTY]], AcquireGreaterEqual, 1)
// CHECK: scf.yield %[[SRC0]] : memref<32x96xbf16>
// CHECK: aie.use_lock(%[[SRC1_EMPTY]], AcquireGreaterEqual, 1)
// CHECK: scf.yield %[[SRC1]] : memref<32x96xbf16>
// CHECK: scf.index_switch
// CHECK: aie.use_lock(%{{.*}}row_store0_src_0_full{{.*}}, Release, 1)
// CHECK: memref.store {{.*}}, %[[IDX]]{{\[}}%{{.*}}{{\]}} : memref<2xi32>
// CHECK: aie.use_lock(%[[DST_FULL]], AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%[[DST_EMPTY]], Release, 1)
// CHECK: aie.mem(%[[CORE]]) {
// CHECK: aie.dma_bd(%[[SRC0]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[SRC1]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[DST]] : memref<32x96xbf16>, 0, 3072)
