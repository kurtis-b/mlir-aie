//===- memtile_dma_op.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-design-to-json %s | FileCheck %s

// CHECK: "dmas": [
// CHECK-DAG: "kind": "memtile"
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 1
// CHECK-DAG: "style": "dma"
// CHECK-DAG: "direction": "MM2S"
// CHECK-DAG: "channel_index": 1
// CHECK-DAG: "repeat_count": 3
// CHECK-DAG: "symbol_name": "sender"
// CHECK-DAG: "buffer": "memtile_dma_buf"
// CHECK-DAG: "bd_id": 5
// CHECK-DAG: "memtile_dma_present": true

module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_buf = aie.buffer(%tile_0_1) {sym_name = "memtile_dma_buf"} : memref<8xi32>
    %memtile_ready = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "memtile_ready"}
    %memtile_done = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "memtile_done"}

    %memtile_dma = aie.memtile_dma(%tile_0_1) {
      %sender = aie.dma(MM2S, 1) {loop = false, repeat_count = 3 : i32, sym_name = "sender"} [{
        aie.use_lock(%memtile_done, AcquireGreaterEqual, 1)
        aie.dma_bd(%memtile_dma_buf : memref<8xi32>, 0, 8) {bd_id = 5 : i32}
        aie.use_lock(%memtile_ready, Release, 1)
      }]
      aie.end
    }
  }
}
