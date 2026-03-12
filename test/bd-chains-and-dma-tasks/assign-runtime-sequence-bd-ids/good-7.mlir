//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 AMD Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// This test ensures that runtime-sequence BD assignment respects the DMA
// channel of each task on memtiles, whose BD IDs are partitioned by channel
// parity.

module {
  aie.device(npu1) {
    %mem_tile_0_1 = aie.tile(0, 1)

    aie.runtime_sequence(%arg0: memref<8xi16>) {
      %t0 = aiex.dma_configure_task(%mem_tile_0_1, MM2S, 0) {
        // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t1 = aiex.dma_configure_task(%mem_tile_0_1, MM2S, 1) {
        // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 24 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      %t2 = aiex.dma_configure_task(%mem_tile_0_1, S2MM, 2) {
        // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
    }
  }
}
