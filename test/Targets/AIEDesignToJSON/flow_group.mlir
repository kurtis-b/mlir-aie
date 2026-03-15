//===- flow_group.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | aie-translate --aie-design-to-json | FileCheck %s

// CHECK: "groups": [
// CHECK-DAG: "id": "flow:flow_2_0_DMA_0"
// CHECK-DAG: "kind": "flow"
// CHECK-DAG: "name": "flow_2_0_DMA_0"
// CHECK-DAG: "streams": [
// CHECK-DAG: "route0"
// CHECK-DAG: "tiles": [
// CHECK-DAG: "col": 2
// CHECK-DAG: "row": 0
// CHECK-DAG: "col": 1
// CHECK-DAG: "row": 3
// CHECK-DAG: "col": 3
// CHECK-DAG: "row": 1
// CHECK: "metadata": {
// CHECK-DAG: "flow_count": 1
// CHECK-DAG: "group_count": 1
// CHECK: "streams": [
// CHECK-DAG: "id": "route0"
// CHECK-DAG: "kind": "circuit"
// CHECK-DAG: "group_candidates": [
// CHECK-DAG: "flow:flow_2_0_DMA_0"
// CHECK-DAG: "source": {
// CHECK-DAG: "col": 2
// CHECK-DAG: "row": 0
// CHECK-DAG: "bundle": "DMA"
// CHECK-DAG: "channel": 0
// CHECK-DAG: "destinations": [
// CHECK-DAG: "col": 1
// CHECK-DAG: "row": 3
// CHECK-DAG: "bundle": "DMA"
// CHECK-DAG: "channel": 0
// CHECK-DAG: "col": 3
// CHECK-DAG: "row": 1
// CHECK-DAG: "bundle": "DMA"
// CHECK-DAG: "channel": 0

module {
  aie.device(xcvc1902) {
    %t13 = aie.tile(1, 3)
    %t11 = aie.tile(1, 1)
    %t10 = aie.tile(1, 0)
    %t20 = aie.tile(2, 0)
    %t30 = aie.tile(3, 0)
    %t31 = aie.tile(3, 1)

    aie.flow(%t20, DMA : 0, %t13, DMA : 0)
    aie.flow(%t20, DMA : 0, %t31, DMA : 0)
  }
}
