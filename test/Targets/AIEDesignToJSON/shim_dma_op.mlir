//===- shim_dma_op.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-design-to-json %s | FileCheck %s

// CHECK: "dmas": [
// CHECK-DAG: "kind": "shim"
// CHECK-DAG: "col": 0
// CHECK-DAG: "row": 0
// CHECK-DAG: "style": "dma"
// CHECK-DAG: "direction": "MM2S"
// CHECK-DAG: "channel_index": 0
// CHECK-DAG: "symbol_name": "shim_sender"
// CHECK-DAG: "buffer": "shim_dma_buffer"
// CHECK-DAG: "bd_id": 9
// CHECK-DAG: "shim_dma_present": true

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %shim_dma_buffer = aie.external_buffer {sym_name = "shim_dma_buffer"} : memref<16xi32>
    %shim_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "shim_lock"}

    %shim_dma = aie.shim_dma(%tile_0_0) {
      %shim_sender = aie.dma(MM2S, 0) {sym_name = "shim_sender"} [{
        aie.use_lock(%shim_lock, Acquire, 1)
        aie.dma_bd(%shim_dma_buffer : memref<16xi32>, 0, 16) {bd_id = 9 : i32}
        aie.use_lock(%shim_lock, Release, 1)
      }]
      aie.end
    }
  }
}
