//===- memtile_row_store.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

aie.device(xcve2302) {
  %mem = aie.tile(2, 1)
  %core = aie.tile(2, 2)

  aie.memtile_row_store @row_store0(%core, %mem) {part_count = 4 : i32, buffer_count = 2 : i32, compute_buffer_count = 4 : i32} : memref<32x96xbf16>
  aie.memtile_row_store @row_store1(%core, %mem) {
    part_count = 4 : i32,
    buffer_count = 2 : i32,
    compute_buffer_count = 1 : i32,
    compute_produce_buffer_count = 2 : i32,
    compute_consume_buffer_count = 3 : i32
  } : memref<32x96xbf16>

  aie.core(%core) {
    %src = aie.memtile_row_store.acquire @row_store0(Produce) : memref<32x96xbf16>
    %dst = aie.memtile_row_store.acquire @row_store0(Consume) : memref<32x96xbf16>
    %src1 = aie.memtile_row_store.acquire @row_store1(Produce) : memref<32x96xbf16>
    %dst1 = aie.memtile_row_store.acquire @row_store1(Consume) : memref<32x96xbf16>
    aie.memtile_row_store.release @row_store1(Consume)
    aie.memtile_row_store.release @row_store1(Produce)
    aie.memtile_row_store.release @row_store0(Consume)
    aie.memtile_row_store.release @row_store0(Produce)
    aie.end
  }
}

// CHECK: aie.memtile_row_store @row_store0(%[[CORE:.*]], %[[MEM:.*]]) {buffer_count = 2 : i32, compute_buffer_count = 4 : i32, part_count = 4 : i32} : memref<32x96xbf16>
// CHECK: aie.memtile_row_store @row_store1(%[[CORE]], %[[MEM]]) {buffer_count = 2 : i32, compute_consume_buffer_count = 3 : i32, compute_produce_buffer_count = 2 : i32, part_count = 4 : i32} : memref<32x96xbf16>
// CHECK: %[[SRC:.*]] = aie.memtile_row_store.acquire @row_store0(Produce) : memref<32x96xbf16>
// CHECK: %[[DST:.*]] = aie.memtile_row_store.acquire @row_store0(Consume) : memref<32x96xbf16>
// CHECK: %[[SRC1:.*]] = aie.memtile_row_store.acquire @row_store1(Produce) : memref<32x96xbf16>
// CHECK: %[[DST1:.*]] = aie.memtile_row_store.acquire @row_store1(Consume) : memref<32x96xbf16>
// CHECK: aie.memtile_row_store.release @row_store1(Consume)
// CHECK: aie.memtile_row_store.release @row_store1(Produce)
// CHECK: aie.memtile_row_store.release @row_store0(Consume)
// CHECK: aie.memtile_row_store.release @row_store0(Produce)
