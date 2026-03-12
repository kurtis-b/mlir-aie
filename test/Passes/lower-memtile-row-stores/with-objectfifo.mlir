//===- with-objectfifo.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memtile-row-stores --aie-objectFifo-stateful-transform %s | FileCheck %s

module {
  aie.device(xcve2302) {
    %mem = aie.tile(2, 1)
    %producer = aie.tile(2, 2)
    %consumer = aie.tile(3, 2)

    aie.memtile_row_store @row_store0(%producer, %mem) {part_count = 4 : i32} : memref<32x96xbf16>
    aie.objectfifo @of0(%producer, {%consumer}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<32x96xbf16>>

    aie.core(%producer) {
      %src = aie.memtile_row_store.acquire @row_store0(Produce) : memref<32x96xbf16>
      aie.memtile_row_store.release @row_store0(Produce)
      aie.end
    }

    aie.core(%consumer) {
      aie.end
    }
  }
}

// CHECK-LABEL: aie.device(xcve2302) {
// CHECK: %[[PRODUCER:.*]] = aie.tile(2, 2)
// CHECK: %[[CONSUMER:.*]] = aie.tile(3, 2)
// CHECK: aie.mem(%[[PRODUCER]]) {
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(MM2S, 1
// CHECK: aie.mem(%[[CONSUMER]]) {
// CHECK: aie.dma_start(S2MM, 0
