//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | aie-translate --aie-design-to-json | FileCheck %s

// CHECK: "buffers": [
// CHECK-DAG: "name": "tile_buffer"
// CHECK-DAG: "type": "memref<16xi32>"
// CHECK-DAG: "allocation_bytes": 64
// CHECK-DAG: "address": 4096
// CHECK: "cores": [
// CHECK-DAG: "id": "core_0_2"
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 2
// CHECK-DAG: "stack_size": 1024
// CHECK-DAG: "is_empty": true
// CHECK-DAG: "operation_count": 1
// CHECK-DAG: "operations": [
// CHECK-DAG: "aie.end"
// CHECK: "device": {
// CHECK-DAG: "kind": "npu1"
// CHECK-DAG: "columns": 4
// CHECK-DAG: "rows": 6
// CHECK: "dmas": [
// CHECK-DAG: "kind": "mem"
// CHECK-DAG: "direction": "S2MM"
// CHECK-DAG: "channel_index": 0
// CHECK-DAG: "dest": "block1"
// CHECK-DAG: "chain": "block2"
// CHECK-DAG: "buffer": "tile_buffer"
// CHECK-DAG: "offset": 4
// CHECK-DAG: "length": 8
// CHECK-DAG: "bd_id": 7
// CHECK: "groups": [
// CHECK-DAG: "id": "flow:flow_0_0_DMA_0"
// CHECK-DAG: "kind": "flow"
// CHECK-DAG: "name": "flow_0_0_DMA_0"
// CHECK-DAG: "streams": [
// CHECK-DAG: "route0"
// CHECK-DAG: "tiles": [
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 0
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 2
// CHECK: "locks": [
// CHECK-DAG: "name": "tile_lock"
// CHECK-DAG: "lock_id": 1
// CHECK-DAG: "init": 0
// CHECK: "metadata": {
// CHECK-DAG: "flow_count": 1
// CHECK-DAG: "group_count": 1
// CHECK-DAG: "switchbox_count": 3
// CHECK: "schema_version": 4
// CHECK: "streams": [
// CHECK-DAG: "id": "route0"
// CHECK-DAG: "kind": "circuit"
// CHECK-DAG: "group_candidates": [
// CHECK-DAG: "flow:flow_0_0_DMA_0"
// CHECK-DAG: "provenance": {
// CHECK-DAG: "status": "fallback"
// CHECK-DAG: "method": "flow_group"
// CHECK-DAG: "selected_group_count": 1
// CHECK-DAG: "considered_group_count": 0
// CHECK-DAG: "used_fallback_flow_group": true
// CHECK-DAG: "source": {
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 0
// CHECK-DAG: "bundle": "DMA"
// CHECK-DAG: "channel": 0
// CHECK-DAG: "destinations": [
// CHECK-DAG: "dest": {
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 2
// CHECK-DAG: "bundle": "DMA"
// CHECK-DAG: "channel": 0
// CHECK-DAG: "route": [
// CHECK-DAG: "id": "switchbox00"
// CHECK-DAG: "id": "switchbox01"
// CHECK-DAG: "id": "switchbox02"
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 0
// CHECK-DAG: "kind": "shim_noc"
// CHECK-DAG: "shim_mux_present": true
// CHECK-DAG: "row": 1
// CHECK-DAG: "kind": "mem"
// CHECK-DAG: "switchbox_present": true
// CHECK-DAG: "row": 2
// CHECK-DAG: "kind": "core"
// CHECK-DAG: "core_present": true
// CHECK-DAG: "core_ids": [
// CHECK-DAG: "core_0_2"
// CHECK-DAG: "mem_present": true

module @aie_module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_buffer = aie.buffer(%tile_0_2) {sym_name = "tile_buffer", address = 0x1000 : i32} : memref<16xi32>
    %tile_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "tile_lock"}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %dma_0 = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%tile_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%tile_buffer : memref<16xi32>, 4, 8) {bd_id = 7 : i32}
      aie.use_lock(%tile_lock, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
  }
}
