//===- memtile_channel_bd_aware_assignment.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids %s | FileCheck %s

// This reduced repro matches the encoder-style pressure pattern on a memtile.
// The deep replay path must not monopolize one BD bank, or the tail drain
// fails later during aie-assign-bd-ids. The objectfifo lowering now chooses
// channels with enough provisional BD headroom on the matching even/odd bank.

// CHECK-LABEL: aie.device(npu2)
// CHECK: %memtile_dma_7_1 = aie.memtile_dma(%{{.*}}tile_7_1)
// CHECK: aie.dma_start(MM2S, 0,
// CHECK: aie.dma_bd(%ffnROut_cons_buff_0{{.*}}) {bd_id = 0 : i32, next_bd_id = 1 : i32}
// CHECK: aie.dma_start(S2MM, 1,
// CHECK: aie.dma_bd(%ffnROut_cons_buff_0{{.*}}) {bd_id = 24 : i32, next_bd_id = 25 : i32}
// CHECK: aie.dma_start(S2MM, 0,
// CHECK: aie.dma_bd(%inR_cons_buff_0{{.*}}) {bd_id = 12 : i32, next_bd_id = 13 : i32}
// CHECK: aie.dma_start(MM2S, 1,
// CHECK: aie.dma_bd(%inR_cons_buff_0{{.*}}) {bd_id = 36 : i32, next_bd_id = 37 : i32}
// CHECK: aie.dma_start(MM2S, 2,
// CHECK: aie.dma_bd(%outLN2_cons_buff_0{{.*}}) {bd_id = 14 : i32, next_bd_id = 15 : i32}
// CHECK: aie.dma_start(S2MM, 3,
// CHECK: aie.dma_bd(%outLN2_cons_buff_0{{.*}}) {bd_id = 38 : i32, next_bd_id = 39 : i32}

module {
  aie.device(npu2) {
    %tile_4_2 = aie.tile(4, 2)
    %tile_7_5 = aie.tile(7, 5)
    %mem_tile_7_1 = aie.tile(7, 1)
    %shim_noc_tile_7_0 = aie.tile(7, 0)

    aie.objectfifo @ffnRIn(%mem_tile_7_1, {%tile_7_5}, 12 : i32)
      : !aie.objectfifo<memref<32x64xbf16>>
    aie.objectfifo @ffnROut(%tile_4_2, {%mem_tile_7_1}, 12 : i32)
      : !aie.objectfifo<memref<32x64xbf16>>
    aie.objectfifo.link [@ffnROut] -> [@ffnRIn]([] [0])

    aie.objectfifo @inR(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32)
      : !aie.objectfifo<memref<32x64xbf16>>
    aie.objectfifo @memR(%mem_tile_7_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_4_2}, 2 : i32)
      : !aie.objectfifo<memref<32x64xbf16>>
    aie.objectfifo.link [@inR] -> [@memR]([] [0])

    aie.objectfifo @memLN2(%mem_tile_7_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32)
      : !aie.objectfifo<memref<32x64xbf16>>
    aie.objectfifo @outLN2(%tile_7_5, {%mem_tile_7_1}, 2 : i32)
      : !aie.objectfifo<memref<32x64xbf16>>
    aie.objectfifo.link [@outLN2] -> [@memLN2]([] [0])
  }
}
