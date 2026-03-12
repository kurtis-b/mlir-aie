//===- memtile_row_store_bad.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.memtile_row_store' op computeTile must be a core tile

aie.device(xcve2302) {
  %mem = aie.tile(2, 1)
  aie.memtile_row_store @row_store0(%mem, %mem) {part_count = 4 : i32} : memref<32x96xbf16>
}

// -----

// CHECK: 'aie.memtile_row_store' op memTile must be a memtile

aie.device(xcve2302) {
  %mem = aie.tile(2, 1)
  %core = aie.tile(2, 2)
  aie.memtile_row_store @row_store0(%core, %core) {part_count = 4 : i32} : memref<32x96xbf16>
}

// -----

// CHECK: 'aie.memtile_row_store' op part_count must be >= 1

aie.device(xcve2302) {
  %mem = aie.tile(2, 1)
  %core = aie.tile(2, 2)
  aie.memtile_row_store @row_store0(%core, %mem) {part_count = 0 : i32} : memref<32x96xbf16>
}

// -----

// CHECK: 'aie.memtile_row_store.acquire' op must be called from inside a CoreOp

aie.device(xcve2302) {
  %mem = aie.tile(2, 1)
  %core = aie.tile(2, 2)
  aie.memtile_row_store @row_store0(%core, %mem) {part_count = 4 : i32} : memref<32x96xbf16>
  %src = aie.memtile_row_store.acquire @row_store0(Produce) : memref<32x96xbf16>
}

// -----

// CHECK: 'aie.memtile_row_store.acquire' op memtile row store element and acquire result type must match.

aie.device(xcve2302) {
  %mem = aie.tile(2, 1)
  %core = aie.tile(2, 2)
  aie.memtile_row_store @row_store0(%core, %mem) {part_count = 4 : i32} : memref<32x96xbf16>
  aie.core(%core) {
    %src = aie.memtile_row_store.acquire @row_store0(Produce) : memref<64x48xbf16>
    aie.end
  }
}

// -----

// CHECK: 'aie.core' op memtile row store accessed by core running on non-compute tile

aie.device(xcve2302) {
  %mem = aie.tile(2, 1)
  %core = aie.tile(2, 2)
  %other = aie.tile(2, 3)
  aie.memtile_row_store @row_store0(%core, %mem) {part_count = 4 : i32} : memref<32x96xbf16>
  aie.core(%other) {
    aie.memtile_row_store.release @row_store0(Consume)
    aie.end
  }
}
