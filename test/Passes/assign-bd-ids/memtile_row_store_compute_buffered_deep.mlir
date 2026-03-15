//===- memtile_row_store_compute_buffered_deep.mlir ------------*- MLIR -*-===//
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

    aie.memtile_row_store @row_store0(%core, %mem) {
      part_count = 2 : i32,
      buffer_count = 2 : i32,
      compute_buffer_count = 4 : i32
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

// CHECK-LABEL: aie.device(xcve2302) {
// CHECK: aie.dma_bd(%{{.*}}row_store0_src_0{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_src_1{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 1 : i32, next_bd_id = 2 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_src_2{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 2 : i32, next_bd_id = 3 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_src_3{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 3 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_dst_0{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 4 : i32, next_bd_id = 5 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_dst_1{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 5 : i32, next_bd_id = 6 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_dst_2{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 6 : i32, next_bd_id = 7 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_dst_3{{.*}} : memref<32x96xbf16>, 0, 3072) {bd_id = 7 : i32, next_bd_id = 4 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_row_0{{.*}} : memref<6144xbf16>, 0, 6144) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_row_1{{.*}} : memref<6144xbf16>, 0, 6144) {bd_id = 1 : i32, next_bd_id = 0 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_row_0{{.*}} : memref<6144xbf16>, 0, 6144) {bd_id = 24 : i32, next_bd_id = 25 : i32}
// CHECK: aie.dma_bd(%{{.*}}row_store0_row_1{{.*}} : memref<6144xbf16>, 0, 6144) {bd_id = 25 : i32, next_bd_id = 24 : i32}
