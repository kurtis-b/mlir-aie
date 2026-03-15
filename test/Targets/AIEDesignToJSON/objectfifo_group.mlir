//===- objectfifo_group.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | aie-translate --aie-design-to-json | FileCheck %s

// CHECK: "groups": [
// CHECK: "id": "objectfifo:objfifo"
// CHECK-DAG: "kind": "objectfifo"
// CHECK-DAG: "name": "objfifo"
// CHECK-DAG: "buffers": [
// CHECK-DAG: "objfifo_buff_0"
// CHECK-DAG: "objfifo_buff_1"
// CHECK-DAG: "locks": [
// CHECK-DAG: "objfifo_lock_0"
// CHECK-DAG: "objfifo_lock_1"
// CHECK-DAG: "tiles": [
// CHECK-DAG: "col": 1
// CHECK-DAG: "row": 2

module {
  aie.device(xcvc1902) {
    %producer = aie.tile(1, 2)
    %consumer = aie.tile(1, 3)

    aie.objectfifo @objfifo(%producer, {%consumer}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%producer) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %subview = aie.objectfifo.acquire @objfifo(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      memref.store %c0_i32, %elem[%c0] : memref<16xi32>
      aie.objectfifo.release @objfifo(Produce, 1)
      aie.end
    }

    aie.core(%consumer) {
      %c0 = arith.constant 0 : index
      %subview = aie.objectfifo.acquire @objfifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %v = memref.load %elem[%c0] : memref<16xi32>
      aie.objectfifo.release @objfifo(Consume, 1)
      aie.end
    }
  }
}
