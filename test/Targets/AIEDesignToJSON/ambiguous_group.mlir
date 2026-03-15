//===- ambiguous_group.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | aie-translate --aie-design-to-json | FileCheck %s

// CHECK: "groups": [
// CHECK-DAG: "id": "row_store:ambig0"
// CHECK-DAG: "id": "row_store:ambig1"
// CHECK: "streams": [
// CHECK-DAG: "id": "route0"
// CHECK-DAG: "group_candidates": [
// CHECK-DAG: "row_store:ambig0"
// CHECK-DAG: "row_store:ambig1"
// CHECK-DAG: "group_matches": [
// CHECK-DAG: "id": "row_store:ambig0"
// CHECK-DAG: "score": 0
// CHECK-DAG: "id": "row_store:ambig1"
// CHECK-DAG: "provenance": {
// CHECK-DAG: "status": "ambiguous"
// CHECK-DAG: "method": "tile_membership"
// CHECK-DAG: "selected_group_count": 2
// CHECK-DAG: "considered_group_count": 2
// CHECK-DAG: "best_score": 0
// CHECK-DAG: "used_fallback_flow_group": false

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    %ambig0_src_0 = aie.buffer(%tile_0_2) {sym_name = "ambig0_src_0"} : memref<4xi32>
    %ambig0_dst_0 = aie.buffer(%tile_0_3) {sym_name = "ambig0_dst_0"} : memref<4xi32>
    %ambig1_src_0 = aie.buffer(%tile_0_2) {sym_name = "ambig1_src_0"} : memref<4xi32>
    %ambig1_dst_0 = aie.buffer(%tile_0_3) {sym_name = "ambig1_dst_0"} : memref<4xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    }

    aie.flow(%tile_0_2, Core : 0, %tile_0_3, Core : 0)
  }
}
