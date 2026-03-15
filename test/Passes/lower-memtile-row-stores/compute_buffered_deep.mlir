//===- compute_buffered_deep.mlir ------------------------------*- MLIR -*-===//
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
// CHECK-DAG: %[[SRC0:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_src_0"} : memref<32x96xbf16>
// CHECK-DAG: %[[SRC1:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_src_1"} : memref<32x96xbf16>
// CHECK-DAG: %[[SRC2:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_src_2"} : memref<32x96xbf16>
// CHECK-DAG: %[[SRC3:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_src_3"} : memref<32x96xbf16>
// CHECK-DAG: %[[DST0:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_dst_0"} : memref<32x96xbf16>
// CHECK-DAG: %[[DST1:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_dst_1"} : memref<32x96xbf16>
// CHECK-DAG: %[[DST2:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_dst_2"} : memref<32x96xbf16>
// CHECK-DAG: %[[DST3:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_dst_3"} : memref<32x96xbf16>
// CHECK-DAG: %[[IDX:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_next_index"} : memref<2xi32>
// CHECK: scf.index_switch
// CHECK: aie.use_lock(%{{.*}}src_0_empty{{.*}}, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}src_1_empty{{.*}}, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}src_2_empty{{.*}}, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}src_3_empty{{.*}}, AcquireGreaterEqual, 1)
// CHECK: scf.index_switch
// CHECK: aie.use_lock(%{{.*}}dst_0_full{{.*}}, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}dst_1_full{{.*}}, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}dst_2_full{{.*}}, AcquireGreaterEqual, 1)
// CHECK: aie.use_lock(%{{.*}}dst_3_full{{.*}}, AcquireGreaterEqual, 1)
// CHECK: aie.mem(%{{.*}}) {
// CHECK: aie.dma_bd(%[[SRC0]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[SRC1]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[SRC2]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[SRC3]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[DST0]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[DST1]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[DST2]] : memref<32x96xbf16>, 0, 3072)
// CHECK: aie.dma_bd(%[[DST3]] : memref<32x96xbf16>, 0, 3072)
