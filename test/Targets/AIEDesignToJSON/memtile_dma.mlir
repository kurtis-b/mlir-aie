//===- memtile_dma.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-design-to-json %s | FileCheck %s

// CHECK: "buffers": [
// CHECK-DAG: "name": "memtile_buf"
// CHECK-DAG: "address": 512
// CHECK: "dmas": [
// CHECK-DAG: "kind": "memtile"
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 1
// CHECK-DAG: "direction": "S2MM"
// CHECK-DAG: "channel_index": 0
// CHECK-DAG: "buffer": "memtile_buf"
// CHECK-DAG: "length": 8
// CHECK-DAG: "bd_id": 3
// CHECK: "locks": [
// CHECK-DAG: "name": "memtile_prod"
// CHECK-DAG: "name": "memtile_cons"
// CHECK-DAG: "lock_id": 0
// CHECK-DAG: "lock_id": 1
// CHECK-DAG: "memtile_dma_present": true

module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_buf = aie.buffer(%tile_0_1) {sym_name = "memtile_buf", address = 0x200 : i32} : memref<8xi32>
    %memtile_prod = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "memtile_prod"}
    %memtile_cons = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "memtile_cons"}

    %memtile_dma = aie.memtile_dma(%tile_0_1) {
      %s2mm = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%memtile_prod, AcquireGreaterEqual, 1)
      aie.dma_bd(%memtile_buf : memref<8xi32>, 0, 8) {bd_id = 3 : i32}
      aie.use_lock(%memtile_cons, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
