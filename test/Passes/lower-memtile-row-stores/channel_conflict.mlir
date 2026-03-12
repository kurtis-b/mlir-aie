//===- channel_conflict.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-memtile-row-stores --verify-diagnostics %s

module {
  aie.device(xcve2302) {
    %mem = aie.tile(2, 1)
    %core = aie.tile(2, 2)
    %buffer = aie.buffer(%core) : memref<32x96xbf16>
    %lock = aie.lock(%core) {init = 1 : i32}
    %lock2 = aie.lock(%core) {init = 0 : i32}
    %m = aie.mem(%core) {
      aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer : memref<32x96xbf16>, 0, 3072)
      aie.use_lock(%lock2, Release, 1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    // expected-error@+1 {{'aie.memtile_row_store' op compute_mm2s_channel conflicts with an existing DMA start on tile (2, 2), direction MM2S, channel 0}}
    aie.memtile_row_store @row_store0(%core, %mem) {part_count = 4 : i32} : memref<32x96xbf16>
  }
}
