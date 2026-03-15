//===- packet_flow.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-translate --aie-design-to-json | FileCheck %s

// CHECK: "device": {
// CHECK-DAG: "kind": "xcvc1902"
// CHECK: "groups": [
// CHECK: "id": "packet_flow:packet_16_0"
// CHECK-DAG: "kind": "packet_flow"
// CHECK-DAG: "name": "packet_16_0"
// CHECK-DAG: "streams": [
// CHECK-DAG: "route0"
// CHECK-DAG: "packet_flows": [
// CHECK-DAG: "packet_16_0"
// CHECK: "packet_flows": [
// CHECK: "destinations": [
// CHECK: {
// CHECK-DAG: "col": 1
// CHECK-DAG: "row": 2
// CHECK-DAG: "bundle": "Core"
// CHECK-DAG: "channel": 0
// CHECK: {
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 2
// CHECK-DAG: "bundle": "DMA"
// CHECK-DAG: "channel": 1
// CHECK: "id": "packet_16_0"
// CHECK: "packet_id": 16
// CHECK: "source": {
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 1
// CHECK-DAG: "bundle": "Core"
// CHECK-DAG: "channel": 0
// CHECK: "schema_version": 4
// CHECK: "streams": [
// CHECK-DAG: "id": "route0"
// CHECK-DAG: "kind": "packet"
// CHECK-DAG: "packet_flow_id": "packet_16_0"
// CHECK-DAG: "group_candidates": [
// CHECK-DAG: "packet_flow:packet_16_0"

module {
  aie.device(xcvc1902) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_0_2 = aie.tile(0, 2)

    aie.flow(%tile_0_1, DMA : 0, %tile_1_2, Core : 1)

    aie.packet_flow(0x10) {
      aie.packet_source<%tile_0_1, Core : 0>
      aie.packet_dest<%tile_1_2, Core : 0>
      aie.packet_dest<%tile_0_2, DMA : 1>
    }
  }
}
