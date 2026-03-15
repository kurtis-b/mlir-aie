//===- mem_dma_op.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-design-to-json %s | FileCheck %s

// CHECK: "dmas": [
// CHECK-DAG: "kind": "mem"
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 2
// CHECK-DAG: "style": "dma"
// CHECK-DAG: "direction": "S2MM"
// CHECK-DAG: "channel_index": 0
// CHECK-DAG: "loop": false
// CHECK-DAG: "repeat_count": 7
// CHECK-DAG: "symbol_name": "player"
// CHECK-DAG: "buffer": "core_dma_buf"
// CHECK-DAG: "bd_id": 2
// CHECK-DAG: "block_ids": [
// CHECK-DAG: "dma_S2MM_0_bd0"

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %core_dma_buf = aie.buffer(%tile_0_2) {sym_name = "core_dma_buf"} : memref<4xi32>
    %core_dma_prod = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "core_dma_prod"}
    %core_dma_cons = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "core_dma_cons"}

    %mem = aie.mem(%tile_0_2) {
      %player = aie.dma(S2MM, 0) {loop = false, repeat_count = 7 : i32, sym_name = "player"} [{
        aie.use_lock(%core_dma_prod, AcquireGreaterEqual, 1)
        aie.dma_bd(%core_dma_buf : memref<4xi32>, 0, 4) {bd_id = 2 : i32}
        aie.use_lock(%core_dma_cons, Release, 1)
      }]
      aie.end
    }
  }
}
