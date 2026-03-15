//===- row_store_group.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memtile-row-stores --aie-create-pathfinder-flows --aie-find-flows %s | aie-translate --aie-design-to-json | FileCheck %s

// CHECK: "groups": [
// CHECK-DAG: "id": "row_store:row_store0"
// CHECK-DAG: "kind": "row_store"
// CHECK-DAG: "name": "row_store0"
// CHECK-DAG: "buffers": [
// CHECK-DAG: "row_store0_src"
// CHECK-DAG: "row_store0_dst"
// CHECK-DAG: "row_store0_row"
// CHECK-DAG: "locks": [
// CHECK-DAG: "row_store0_src_empty"
// CHECK-DAG: "row_store0_src_full"
// CHECK-DAG: "row_store0_dst_empty"
// CHECK-DAG: "row_store0_dst_full"
// CHECK-DAG: "row_store0_row_empty"
// CHECK-DAG: "row_store0_row_full"
// CHECK-DAG: "dmas": [
// CHECK-DAG: "mem_2_2"
// CHECK-DAG: "memtile_2_1"
// CHECK-DAG: "streams": [
// CHECK-DAG: "route0"
// CHECK-DAG: "route1"
// CHECK-DAG: "tiles": [
// CHECK-DAG: "col": 2
// CHECK-DAG: "row": 1
// CHECK-DAG: "col": 2
// CHECK-DAG: "row": 2
// CHECK: "streams": [
// CHECK-DAG: "id": "route0"
// CHECK-DAG: "kind": "circuit"
// CHECK-DAG: "id": "route1"
// CHECK-DAG: "kind": "circuit"
// CHECK-DAG: "group_candidates": [
// CHECK-DAG: "row_store:row_store0"

module {
  aie.device(xcve2302) {
    %mem = aie.tile(2, 1)
    %core = aie.tile(2, 2)

    aie.memtile_row_store @row_store0(%core, %mem) {part_count = 4 : i32} : memref<32x96xbf16>

    aie.core(%core) {
      %src = aie.memtile_row_store.acquire @row_store0(Produce) : memref<32x96xbf16>
      %dst = aie.memtile_row_store.acquire @row_store0(Consume) : memref<32x96xbf16>
      memref.copy %src, %dst : memref<32x96xbf16> to memref<32x96xbf16>
      aie.memtile_row_store.release @row_store0(Produce)
      aie.memtile_row_store.release @row_store0(Consume)
      aie.end
    }
  }
}
