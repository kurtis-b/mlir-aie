#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence

from aie.iron import str_to_dtype


microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-H", type=int, default=12)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        maybe_taps = my_mha(
            args.dev,
            args.M,
            args.K,
            args.N,
            args.H,
            args.n_aie_cols,
            args.dtype_in,
            args.dtype_out,
            args.b_col_maj,
            args.emulate_bf16_mmul_with_bfp16,
            args.trace_size,
            args.generate_taps,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)

    if args.generate_taps:
        return maybe_taps


def ceildiv(a, b):
    return (a + b - 1) // b


def my_mha(
    dev,
    M,
    K,
    N,
    H,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    generate_taps=False,
):
    """
    Generates a design for Multi-Head Attention. Due to the number of L3 channels
    needed in this design, 4 columns of the AIE array are required. This means
    a full encoder can't be run on the Phoenix, i.e. the FFN and layer norm
    layers can't be run on the array. 
    """
    head_dim = N // H

    X_STR = "x_proj"
    WQ_STR = "q_proj"
    WK_STR = "k_proj"
    WV_STR = "v_proj"
    WO_STR = "o_proj"
    Q_STR = "q"
    K_STR = "k"
    V_STR = "v"
    ATTN_SCORE_STR = "attn_score"
    SOFTMAX_STR = "softmax"
    ATTN_SCORE_V_STR = "attns_score_v"
    ACCUM_STR = "accum"
    OUTPUT_STR = "output"
    L3_POS_STR = "l3_col"
    L2_POS_STR = "l2_col"
    L1_POS_STR = "l1_row_col"
    COL_IDX = 1
    ROW_IDX = 0
    left_mtx_in = {
        X_STR: {
            L1_POS_STR: [(0, 2), (1, 2)],
            L2_POS_STR: 0,
            L3_POS_STR: 0,
        },
        Q_STR: {
            L1_POS_STR: [(0, 1), (0, 3)],
            L2_POS_STR: 1,
            L3_POS_STR: 1,
        }
    }
    right_mtx_in = {
        WQ_STR: {
            L1_POS_STR: [(0, 0), (1, 0)],
            L2_POS_STR: 0,
            L3_POS_STR: 0,
        },
        WK_STR: {
            L1_POS_STR: [(2, 0), (3, 0)],
            L2_POS_STR: 2,
            L3_POS_STR: 2,
        },
        WV_STR: {
            L1_POS_STR: [(0, 2), (1, 2)],
            L2_POS_STR: 2,
            L3_POS_STR: 2,
        },
        K_STR: {
            L1_POS_STR: [(0, 1), (0, 3)],
            L2_POS_STR: 3,
            L3_POS_STR: 3,
        },
        V_STR: {
            L1_POS_STR: [(2, 1), (2, 3)],
            L2_POS_STR: 3,
            L3_POS_STR: 3,
        },
        WO_STR: {
            L1_POS_STR: [(3, 1), (3, 3)],
            L2_POS_STR: 2,
            L3_POS_STR: 1,
        }
    }
    l1_fuse_mtx_in = {
        ATTN_SCORE_STR: {
            L1_POS_STR: [(1, 1), (1, 3)],
            L2_POS_STR: 1,
        },
        SOFTMAX_STR: {
            L1_POS_STR: [(2, 1), (2, 3)],
            L2_POS_STR: 1,
        },
        ATTN_SCORE_V_STR: {
            L1_POS_STR: [(3, 1), (3, 3)],
        },
        ACCUM_STR: {
            L1_POS_STR: [(3, 2)],
        }
    }
    l3_fuse_mtx_out = {
        Q_STR: {
            L2_POS_STR: 0,
            L3_POS_STR: 0,
        },
        K_STR: {
            L2_POS_STR: 0,
            L3_POS_STR: 0,
        },
        V_STR: {
            L2_POS_STR: 3,
            L3_POS_STR: 3,
        },
        OUTPUT_STR: {
            L2_POS_STR: 3,
            L3_POS_STR: 3,
        }
    }

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    for key, val in left_mtx_in.items():
        l1_list = val.get(L1_POS_STR, [])
        assert len(l1_list) == len(set(l1_list)), f"Multiple left matrix tiles found in left_mtx_in[{key}][{L1_POS_STR}]"
    for key, val in right_mtx_in.items():
        l1_list = val.get(L1_POS_STR, [])
        assert len(l1_list) == len(set(l1_list)), f"Multiple right matrix tiles found in right_mtx_in[{key}][{L1_POS_STR}]"
    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    v_proj_dims = (64, 64, 64)
    proj_dims = [v_proj_dims]
    
    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    # npu is a 4 row x 4 col array
    if dev == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    # npu2 is a 4 row x 8 col array
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    assert (
        N % H == 0
    ), """Embedding dimension must be divisible by number of heads"""
    for dims in proj_dims:
        assert (
            M % dims[0]  == 0
        ), """A must be tileable into (m, dims[1])-sized blocks"""
        assert K % dims[1] == 0
        assert (
            dims[0] * dims[1] * dims[2] > 0
        ), f"Matmul dims {dims} must be non-zero"
        assert (
            dims[0] % r == 0
        ), f"Matmul dim {dims[0]} must be divisible by {r}"
        assert (
            dims[1] % s == 0
        ), f"Matmul dim {dims[1]} must be divisible by {s}"
        assert (
            dims[2] % t == 0
        ), f"Matmul dim {dims[2]} must be divisible by {t}"
    for data_path in right_mtx_in.values():
        for dims in proj_dims:
            assert (
                N % (dims[2] * len(data_path[L1_POS_STR])) == 0
            ), """B must be tileable into (k, dims[2] * n_aie_rows_projs)-sized blocks"""

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    if dev == "npu":
        dev_ty = AIEDevice.npu1
    else:
        dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        # L2 tiles for Q, K, V projections
        X_l2_ty = np.ndarray[(v_proj_dims[0] * v_proj_dims[1],), np.dtype[dtype_in]]
        Wv_l2_ty = np.ndarray[(v_proj_dims[1] * v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        v_l2_ty = np.ndarray[(v_proj_dims[0] * v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR]),), np.dtype[dtype_out]]

        # L1 tiles for Q, K, V projections
        X_l1_ty = np.ndarray[(v_proj_dims[0] * v_proj_dims[1],), np.dtype[dtype_in]]
        Wv_l1_ty = np.ndarray[(v_proj_dims[1] * v_proj_dims[2],), np.dtype[dtype_in]]
        v_l1_ty_out = np.ndarray[(v_proj_dims[0] * v_proj_dims[2],), np.dtype[dtype_out]]

        # AIE Core Function declarations
        # Last part of the name is whether the right matrix is row major
        matmul_vectorized_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
        row_major = 1
        col_major = 0
        zero_projs = external_func(
            f"zero_{dtype_out_str}_{v_proj_dims[0]}_{v_proj_dims[1]}_{v_proj_dims[2]}_{row_major}",
            inputs=[v_l1_ty_out]
        )
        matmul_projs = external_func(
            matmul_vectorized_func_name + f"_{v_proj_dims[0]}_{v_proj_dims[1]}_{v_proj_dims[2]}_{row_major}",
            inputs=[X_l1_ty, Wv_l1_ty, v_l1_ty_out]
        )

        if dev == "npu":
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 4th columns
        else:
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 4th columns
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # AIE-array data movement for Q, K, V projections
        # L3 to L2 data movement
        X_l3l2_fifos = object_fifo(
            f"X_L3L2", 
            shim_tiles[left_mtx_in[X_STR][L3_POS_STR]], 
            mem_tiles[left_mtx_in[X_STR][L2_POS_STR]], 
            fifo_depth,
            X_l2_ty
        )
        Wv_l3l2_fifos = object_fifo(
            f"Wv_L3L2", 
            shim_tiles[right_mtx_in[WV_STR][L3_POS_STR]], 
            mem_tiles[right_mtx_in[WV_STR][L2_POS_STR]], 
            fifo_depth,
            Wv_l2_ty
        )
        # L2 to L1 data movement
        X_l2l1_fifos = object_fifo(
            f"X_L2L1", 
            mem_tiles[left_mtx_in[X_STR][L2_POS_STR]], 
            [core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]] for l1_pos in left_mtx_in[X_STR][L1_POS_STR]], 
            fifo_depth,
            X_l1_ty,
            [
                (v_proj_dims[0] // r, r * v_proj_dims[1]),
                (v_proj_dims[1] // s, s),
                (r, v_proj_dims[1]),
                (s, 1),
            ],
        )
        Wv_l2l1_fifos = [object_fifo(
            f"Wv_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
            mem_tiles[right_mtx_in[WV_STR][L2_POS_STR]], 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            fifo_depth,
            Wv_l1_ty,
            [
                (v_proj_dims[1] // s, s * v_proj_dims[2]),
                (v_proj_dims[2] // t, t),
                (s, v_proj_dims[2]),
                (t, 1),
            ],
        ) for l1_pos in right_mtx_in[WV_STR][L1_POS_STR]]
        # L3 to L1 links
        object_fifo_link(X_l3l2_fifos, X_l2l1_fifos)
        object_fifo_link(Wv_l3l2_fifos, Wv_l2l1_fifos, [], [v_proj_dims[1] * v_proj_dims[2] * i for i in range(len(right_mtx_in[WV_STR][L1_POS_STR]))])
        # L1 to L2 data movement
        v_l1l2_fifos = [object_fifo(
            f"v_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            mem_tiles[l3_fuse_mtx_out[V_STR][L2_POS_STR]], 
            fifo_depth, 
            v_l1_ty_out,
        ) for l1_pos in right_mtx_in[WV_STR][L1_POS_STR]]
        # L2 to L3 data movement
        v_l2l3_fifos = object_fifo(
            f"v_L2L3", 
            mem_tiles[l3_fuse_mtx_out[V_STR][L2_POS_STR]], 
            shim_tiles[l3_fuse_mtx_out[V_STR][L3_POS_STR]], 
            fifo_depth, 
            v_l2_ty,
            [
                (v_proj_dims[0] // r, r * v_proj_dims[2]),
                (r, t),
                (v_proj_dims[2] // t, r * t),
                (t, 1),
            ],
        )
        # L1 to L3 links
        object_fifo_link(v_l1l2_fifos, v_l2l3_fifos, [v_proj_dims[1] * v_proj_dims[2] * i for i in range(len(right_mtx_in[WV_STR][L1_POS_STR]))], [])

        # Compute for Q, K, V projections
        for row, l1_pos in enumerate(right_mtx_in[WV_STR][L1_POS_STR]):
            @core(
                core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]],
                f"mha_mm_{v_proj_dims[0]}x{v_proj_dims[1]}x{v_proj_dims[2]}_row_major.o",
                stack_size=0xF00
            )
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(N // v_proj_dims[2] // len(right_mtx_in[WV_STR][L1_POS_STR])):
                        elem_v = v_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                        zero_projs(elem_v)
                        for _ in range_(K // v_proj_dims[1]):
                            elem_in_xv = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wv = Wv_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                            matmul_projs(elem_in_xv, elem_in_wv, elem_v)
                            Wv_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                            X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        v_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[((M * K) + (K * N),), np.dtype[dtype_in]], # Order: X
            np.ndarray[(M * N,), np.dtype[dtype_in]], # Order: W_V
            np.ndarray[(M * N,), np.dtype[dtype_out]], # V
        )
        def sequence(A, B, C):
            # Send the data for calculating Q, K, V projections
            for row_offset in range(0, M, v_proj_dims[0]):
                npu_dma_memcpy_nd(
                    metadata=X_l3l2_fifos,
                    bd_id=0,
                    mem=A,
                    offsets=[0, 0, 0, row_offset * K],
                    sizes=[N // v_proj_dims[2] // len(right_mtx_in[WV_STR][L1_POS_STR]), K // v_proj_dims[1], v_proj_dims[0], v_proj_dims[1]],
                    strides=[0, v_proj_dims[1], K, 1],
                )

                for col_offset_v in range(0, N, v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR])):
                    npu_dma_memcpy_nd(
                        metadata=Wv_l3l2_fifos,
                        bd_id=3,
                        mem=B,
                        offsets=[0, 0, 0, col_offset_v],
                        sizes=[K // v_proj_dims[1], len(right_mtx_in[WV_STR][L1_POS_STR]), v_proj_dims[1], v_proj_dims[2]],
                        strides=[v_proj_dims[1] * N, v_proj_dims[2], N, 1],
                    )

                    npu_dma_memcpy_nd(
                        metadata=v_l2l3_fifos,
                        bd_id=6,
                        mem=C,
                        offsets=[0, 0, 0, row_offset * N + col_offset_v],
                        sizes=[1, len(right_mtx_in[WV_STR][L1_POS_STR]), v_proj_dims[0], v_proj_dims[2]],
                        strides=[0, v_proj_dims[2], N, 1],
                    )

                    dma_wait(v_l2l3_fifos)

if __name__ == "__main__":
    main()
