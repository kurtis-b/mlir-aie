//===- shim_dma.mlir -------------------------------------------*- MLIR -*-===//
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
// CHECK-DAG: "direction": "MM2S"
// CHECK-DAG: "channel_index": 0
// CHECK-DAG: "buffer": "shim_buffer"
// CHECK-DAG: "length": 16
// CHECK-DAG: "bd_id": 5
// CHECK: "locks": [
// CHECK-DAG: "name": "shim_lock"
// CHECK-DAG: "lock_id": 0
// CHECK-DAG: "shim_dma_present": true

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %shim_buffer = aie.external_buffer {sym_name = "shim_buffer"} : memref<16xi32>
    %shim_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "shim_lock"}

    %shim_dma = aie.shim_dma(%tile_0_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%shim_lock, Acquire, 1)
      aie.dma_bd(%shim_buffer : memref<16xi32>, 0, 16) {bd_id = 5 : i32}
      aie.use_lock(%shim_lock, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
