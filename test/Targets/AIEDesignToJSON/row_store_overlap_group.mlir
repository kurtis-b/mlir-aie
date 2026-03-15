//===- row_store_overlap_group.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memtile-row-stores --aie-create-pathfinder-flows --aie-find-flows %s | aie-translate --aie-design-to-json | FileCheck %s

// CHECK: "groups": [
// CHECK-DAG: "id": "row_store:row_store0"
// CHECK-DAG: "id": "row_store:row_store1"

// CHECK: "group_candidates": [
// CHECK-NEXT: "row_store:row_store0"
// CHECK: "group_matches": [
// CHECK-DAG: "id": "row_store:row_store0"
// CHECK-DAG: "score": 3
// CHECK-DAG: "source_dma_match": true
// CHECK: "id": "route0"
// CHECK: "provenance": {
// CHECK-DAG: "status": "resolved"
// CHECK-DAG: "method": "dma_channel"
// CHECK-DAG: "selected_group_count": 1
// CHECK-DAG: "considered_group_count": 2
// CHECK-DAG: "best_score": 3
// CHECK-DAG: "used_fallback_flow_group": false

// CHECK: "group_candidates": [
// CHECK-NEXT: "row_store:row_store1"
// CHECK: "group_matches": [
// CHECK-DAG: "id": "row_store:row_store1"
// CHECK-DAG: "score": 3
// CHECK-DAG: "source_dma_match": true
// CHECK: "id": "route1"

// CHECK: "group_candidates": [
// CHECK-NEXT: "row_store:row_store0"
// CHECK: "group_matches": [
// CHECK-DAG: "id": "row_store:row_store0"
// CHECK-DAG: "score": 3
// CHECK-DAG: "source_dma_match": true
// CHECK: "id": "route2"

// CHECK: "group_candidates": [
// CHECK-NEXT: "row_store:row_store1"
// CHECK: "group_matches": [
// CHECK-DAG: "id": "row_store:row_store1"
// CHECK-DAG: "score": 3
// CHECK-DAG: "source_dma_match": true
// CHECK: "id": "route3"

module {
  aie.device(xcve2302) {
    %mem = aie.tile(2, 1)
    %core = aie.tile(2, 2)

    aie.memtile_row_store @row_store0(%core, %mem) {
      part_count = 4 : i32,
      compute_mm2s_channel = 0 : i32,
      compute_s2mm_channel = 0 : i32,
      memtile_ingress_channel = 0 : i32,
      memtile_egress_channel = 1 : i32
    } : memref<32x96xbf16>

    aie.memtile_row_store @row_store1(%core, %mem) {
      part_count = 4 : i32,
      compute_mm2s_channel = 1 : i32,
      compute_s2mm_channel = 1 : i32,
      memtile_ingress_channel = 2 : i32,
      memtile_egress_channel = 3 : i32
    } : memref<32x96xbf16>

    aie.core(%core) {
      %src0 = aie.memtile_row_store.acquire @row_store0(Produce) : memref<32x96xbf16>
      %dst0 = aie.memtile_row_store.acquire @row_store0(Consume) : memref<32x96xbf16>
      %src1 = aie.memtile_row_store.acquire @row_store1(Produce) : memref<32x96xbf16>
      %dst1 = aie.memtile_row_store.acquire @row_store1(Consume) : memref<32x96xbf16>
      memref.copy %src0, %dst0 : memref<32x96xbf16> to memref<32x96xbf16>
      memref.copy %src1, %dst1 : memref<32x96xbf16> to memref<32x96xbf16>
      aie.memtile_row_store.release @row_store0(Produce)
      aie.memtile_row_store.release @row_store0(Consume)
      aie.memtile_row_store.release @row_store1(Produce)
      aie.memtile_row_store.release @row_store1(Consume)
      aie.end
    }
  }
}
