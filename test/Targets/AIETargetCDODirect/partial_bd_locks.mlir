//===- partial_bd_locks.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-cdo %s --cdo-debug=true | FileCheck %s

// Regression test for direct CDO generation when a BD block carries only one
// side of the lock pair. This is the shape produced by the memtile row-store
// lowering for the first and last BD in a chained memtile DMA transfer.

// CHECK: (BlockWrite-DMAWriteCmd): Start Address:
// CHECK: (Write64): Address:
// CHECK: (MaskWrite64): Address:
// CHECK: (BlockWrite-DMAWriteCmd): Start Address:
// CHECK: (Write64): Address:
// CHECK: (MaskWrite64): Address:

module {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %row = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "row"} : memref<8xi32>
    %row_empty = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "row_empty"}
    %row_full = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "row_full"}

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      aie.dma_start(S2MM, 0, ^ingress_bd, ^egress_dma)
    ^egress_dma:
      aie.dma_start(MM2S, 1, ^egress_bd, ^end)
    ^ingress_bd:
      aie.use_lock(%row_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%row : memref<8xi32>, 0, 4) {bd_id = 0 : i32}
      aie.next_bd ^end
    ^egress_bd:
      aie.dma_bd(%row : memref<8xi32>, 4, 4) {bd_id = 24 : i32}
      aie.use_lock(%row_full, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
