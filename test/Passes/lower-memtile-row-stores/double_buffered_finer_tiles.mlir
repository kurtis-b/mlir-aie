//===- double_buffered_finer_tiles.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memtile-row-stores %s | FileCheck %s

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 5)

    aie.memtile_row_store @row_store0(%core, %mem) {part_count = 16 : i32, buffer_count = 2 : i32} : memref<24x64xbf16>

    aie.core(%core) {
      %src = aie.memtile_row_store.acquire @row_store0(Produce) : memref<24x64xbf16>
      aie.memtile_row_store.release @row_store0(Produce)
      %dst = aie.memtile_row_store.acquire @row_store0(Consume) : memref<24x64xbf16>
      aie.memtile_row_store.release @row_store0(Consume)
      aie.end
    }
  }
}

// CHECK-LABEL: aie.device(npu2) {
// CHECK: %[[ROW0:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_row_0"} : memref<24576xbf16>
// CHECK: %[[ROW1:.*]] = aie.buffer(%{{.*}}) {sym_name = "row_store0_row_1"} : memref<24576xbf16>
// CHECK: aie.memtile_dma(%{{.*}}) {
// CHECK: aie.dma_bd(%[[ROW0]] : memref<24576xbf16>, 0, 24576)
// CHECK: aie.dma_bd(%[[ROW1]] : memref<24576xbf16>, 0, 24576)
// CHECK: aie.dma_bd(%[[ROW0]] : memref<24576xbf16>, 0, 24576)
// CHECK: aie.dma_bd(%[[ROW1]] : memref<24576xbf16>, 0, 24576)
