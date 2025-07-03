#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np

from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}

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
            False: (4, 8, 4),
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
    Generates a matrix multiplication design for the Multi-Head Attention, 
    up to the attention score calculation. Due to the number of L3 channels
    needed in this design, 4 columns of the AIE array are required. This means
    a full encoder can't be run on the Phoenix, i.e. the FFN and layer norm
    layers can't be run on the array. 
    """
    X_STR = "x_proj"
    WQ_STR = "q_proj"
    WK_STR = "k_proj"
    WV_STR = "v_proj"
    Q_STR = "q"
    K_STR = "k"
    V_STR = "v"
    ATTN_SCORE_STR = "attn_score"
    L3_POS_STR = "l3_col"
    L2_POS_STR = "l2_col"
    L1_POS_STR = "l1_row_col"
    COL_IDX = 1
    ROW_IDX = 0

    left_mtx_in = {
        X_STR: {
            L2_POS_STR: 0,
            L3_POS_STR: 0,
            L1_POS_STR: [(0, 0), (1, 0), (2, 0), (3, 0), (0, 2), (1, 2)],
        },
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
    }
    mtx_out = {
        Q_STR: {
            L2_POS_STR: 0,
            L3_POS_STR: 0,
        },
        K_STR: {
            L2_POS_STR: 0,
            L3_POS_STR: 0,
        },
        V_STR: {
            L2_POS_STR: 2,
            L3_POS_STR: 2,
        },
        ATTN_SCORE_STR: {
            L2_POS_STR: 3,
            L3_POS_STR: 3,  
        }
    }
    head_dim = N // H

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    q_proj_dims = (64, 64, 64)
    k_proj_dims = (64, 64, 64)
    v_proj_dims = (64, 64, 64)
    proj_dims = [q_proj_dims, k_proj_dims, v_proj_dims]
    attn_score_mm_dims = (16, 32, 256)
    
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
    assert (
        head_dim % attn_score_mm_dims[1] == 0
    ), """Head dimension must be divisible by left mtx col dim"""
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
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu1_4col
    else:
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        # L2 tiles for Q, K, V projections
        X_l2_ty = np.ndarray[(q_proj_dims[0] * q_proj_dims[1],), np.dtype[dtype_in]]
        Wq_l2_ty = np.ndarray[(q_proj_dims[1] * q_proj_dims[2] * len(right_mtx_in[WQ_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        Wk_l2_ty = np.ndarray[(k_proj_dims[1] * k_proj_dims[2] * len(right_mtx_in[WK_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        Wv_l2_ty = np.ndarray[(v_proj_dims[1] * v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        q_l2_ty = np.ndarray[(q_proj_dims[0] * q_proj_dims[2] * len(right_mtx_in[WQ_STR][L1_POS_STR]),), np.dtype[dtype_out]]
        k_l2_ty = np.ndarray[(k_proj_dims[0] * k_proj_dims[2] * len(right_mtx_in[WK_STR][L1_POS_STR]),), np.dtype[dtype_out]]
        v_l2_ty = np.ndarray[(v_proj_dims[0] * v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR]),), np.dtype[dtype_out]]

        # L1 tiles for Q, K, V projections
        X_l1_ty = np.ndarray[(q_proj_dims[0] * q_proj_dims[1],), np.dtype[dtype_in]]
        Wq_l1_ty = np.ndarray[(q_proj_dims[1] * q_proj_dims[2],), np.dtype[dtype_in]]
        Wk_l1_ty = np.ndarray[(k_proj_dims[1] * k_proj_dims[2],), np.dtype[dtype_in]]
        Wv_l1_ty = np.ndarray[(v_proj_dims[1] * v_proj_dims[2],), np.dtype[dtype_in]]
        q_l1_ty_out = np.ndarray[(q_proj_dims[0] * q_proj_dims[2],), np.dtype[dtype_out]]
        k_l1_ty_out = np.ndarray[(k_proj_dims[0] * k_proj_dims[2],), np.dtype[dtype_out]]
        v_l1_ty_out = np.ndarray[(v_proj_dims[0] * v_proj_dims[2],), np.dtype[dtype_out]]

        # L2/L1 tiles for attention score calculation
        q_l1_ty_in = np.ndarray[(attn_score_mm_dims[0] * attn_score_mm_dims[1],), np.dtype[dtype_out]]
        k_l1_ty_in = np.ndarray[(attn_score_mm_dims[2] * attn_score_mm_dims[1],), np.dtype[dtype_out]]
        attn_score_l1_ty = np.ndarray[(attn_score_mm_dims[0] * attn_score_mm_dims[2],), np.dtype[dtype_out]]

        # AIE Core Function declarations
        # Last part of the name is whether the right matrix is row major
        matmul_vectorized_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
        zero_projs = external_func(
            f"zero_{dtype_out_str}_{q_proj_dims[0]}_{q_proj_dims[1]}_{q_proj_dims[2]}_1",
            inputs=[q_l1_ty_out]
        )
        matmul_projs = external_func(
            matmul_vectorized_func_name + f"_{q_proj_dims[0]}_{q_proj_dims[1]}_{q_proj_dims[2]}_1",
            inputs=[X_l1_ty, Wq_l1_ty, q_l1_ty_out]
        )
        zero_attn_score = external_func(
            f"zero_{dtype_out_str}_{attn_score_mm_dims[0]}_{attn_score_mm_dims[1]}_{attn_score_mm_dims[2]}_0",
            inputs=[attn_score_l1_ty]
        )
        matmul_attn_score = external_func(
            matmul_vectorized_func_name + f"_{attn_score_mm_dims[0]}_{attn_score_mm_dims[1]}_{attn_score_mm_dims[2]}_0",
            inputs=[q_l1_ty_in, k_l1_ty_in, attn_score_l1_ty]
        )
        div_projs = external_func(
            f"div_2d_bf16", 
            inputs=[attn_score_l1_ty, attn_score_l1_ty]
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
            f"X_L3L2", shim_tiles[left_mtx_in[X_STR][L3_POS_STR]], mem_tiles[left_mtx_in[X_STR][L2_POS_STR]], fifo_depth, X_l2_ty
        )
        Wq_l3l2_fifos = object_fifo(
            f"Wq_L3L2", shim_tiles[right_mtx_in[WQ_STR][L3_POS_STR]], mem_tiles[right_mtx_in[WQ_STR][L2_POS_STR]], fifo_depth, Wq_l2_ty
        )
        Wk_l3l2_fifos = object_fifo(
            f"Wk_L3L2", shim_tiles[right_mtx_in[WK_STR][L3_POS_STR]], mem_tiles[right_mtx_in[WK_STR][L2_POS_STR]], fifo_depth, Wk_l2_ty
        )
        Wv_l3l2_fifos = object_fifo(
            f"Wv_L3L2", shim_tiles[right_mtx_in[WV_STR][L3_POS_STR]], mem_tiles[right_mtx_in[WV_STR][L2_POS_STR]], fifo_depth, Wv_l2_ty
        )
        # L2 to L1 data movement
        X_l2l1_fifos = object_fifo(
            f"X_L2L1", mem_tiles[left_mtx_in[X_STR][L2_POS_STR]], [core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]] for l1_pos in left_mtx_in[X_STR][L1_POS_STR]], fifo_depth, X_l1_ty,
            [
                (q_proj_dims[0] // r, r * q_proj_dims[1]),
                (q_proj_dims[1] // s, s),
                (r, q_proj_dims[1]),
                (s, 1),
            ],
        )
        Wq_l2l1_fifos = [object_fifo(
            f"Wq_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", mem_tiles[right_mtx_in[WQ_STR][L2_POS_STR]], core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], fifo_depth, Wq_l1_ty,
            [
                (q_proj_dims[1] // s, s * q_proj_dims[2]),
                (q_proj_dims[2] // t, t),
                (s, q_proj_dims[2]),
                (t, 1),
            ],
        ) for l1_pos in right_mtx_in[WQ_STR][L1_POS_STR]]
        Wk_l2l1_fifos = [object_fifo(
            f"Wk_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", mem_tiles[right_mtx_in[WK_STR][L2_POS_STR]], core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], fifo_depth, Wk_l1_ty,
            [
                (k_proj_dims[1] // s, s * k_proj_dims[2]),
                (k_proj_dims[2] // t, t),
                (s, k_proj_dims[2]),
                (t, 1),
            ],
        ) for l1_pos in right_mtx_in[WK_STR][L1_POS_STR]]
        Wv_l2l1_fifos = [object_fifo(
            f"Wv_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", mem_tiles[right_mtx_in[WV_STR][L2_POS_STR]], core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], fifo_depth, Wv_l1_ty,
            [
                (v_proj_dims[1] // s, s * v_proj_dims[2]),
                (v_proj_dims[2] // t, t),
                (s, v_proj_dims[2]),
                (t, 1),
            ],
        ) for l1_pos in right_mtx_in[WV_STR][L1_POS_STR]]
        # L3 to L1 links
        object_fifo_link(X_l3l2_fifos, X_l2l1_fifos)
        object_fifo_link(Wq_l3l2_fifos, Wq_l2l1_fifos, [], [q_proj_dims[1] * q_proj_dims[2] * i for i in range(len(right_mtx_in[WQ_STR][L1_POS_STR]))])
        object_fifo_link(Wk_l3l2_fifos, Wk_l2l1_fifos, [], [k_proj_dims[1] * k_proj_dims[2] * i for i in range(len(right_mtx_in[WK_STR][L1_POS_STR]))])
        object_fifo_link(Wv_l3l2_fifos, Wv_l2l1_fifos, [], [v_proj_dims[1] * v_proj_dims[2] * i for i in range(len(right_mtx_in[WV_STR][L1_POS_STR]))])
        # L1 to L2 data movement
        q_l1l2_fifos = [object_fifo(
            f"q_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], mem_tiles[mtx_out[Q_STR][L2_POS_STR]], fifo_depth, q_l1_ty_out,
        ) for l1_pos in right_mtx_in[WQ_STR][L1_POS_STR]]
        k_l1l2_fifos = [object_fifo(
            f"k_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], mem_tiles[mtx_out[K_STR][L2_POS_STR]], fifo_depth, k_l1_ty_out,
        ) for l1_pos in right_mtx_in[WK_STR][L1_POS_STR]]
        v_l1l2_fifos = [object_fifo(
            f"v_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], mem_tiles[mtx_out[V_STR][L2_POS_STR]], fifo_depth, v_l1_ty_out,
        ) for l1_pos in right_mtx_in[WV_STR][L1_POS_STR]]
        # L2 to L3 data movement
        q_l2l3_fifos = object_fifo(
            f"q_L2L3", mem_tiles[mtx_out[Q_STR][L2_POS_STR]], shim_tiles[mtx_out[Q_STR][L3_POS_STR]], fifo_depth, q_l2_ty,
            [
                (q_proj_dims[0] // r, r * q_proj_dims[2]),
                (r, t),
                (q_proj_dims[2] // t, r * t),
                (t, 1),
            ],
        )
        k_l2l3_fifos = object_fifo(
            f"k_L2L3", mem_tiles[mtx_out[K_STR][L2_POS_STR]], shim_tiles[mtx_out[K_STR][L3_POS_STR]], fifo_depth, k_l2_ty,
            [
                (k_proj_dims[0] // r, r * k_proj_dims[2]),
                (r, t),
                (k_proj_dims[2] // t, r * t),
                (t, 1),
            ],
        )
        v_l2l3_fifos = object_fifo(
            f"v_L2L3", mem_tiles[mtx_out[V_STR][L2_POS_STR]], shim_tiles[mtx_out[V_STR][L3_POS_STR]], fifo_depth, v_l2_ty,
            [
                (v_proj_dims[0] // r, r * v_proj_dims[2]),
                (r, t),
                (v_proj_dims[2] // t, r * t),
                (t, 1),
            ],
        )
        # L1 to L3 links
        object_fifo_link(q_l1l2_fifos, q_l2l3_fifos, [q_proj_dims[0] * q_proj_dims[2] * i for i in range(len(right_mtx_in[WQ_STR][L1_POS_STR]))], [])
        object_fifo_link(k_l1l2_fifos, k_l2l3_fifos, [k_proj_dims[0] * k_proj_dims[2] * i for i in range(len(right_mtx_in[WK_STR][L1_POS_STR]))], [])
        object_fifo_link(v_l1l2_fifos, v_l2l3_fifos, [v_proj_dims[1] * v_proj_dims[2] * i for i in range(len(right_mtx_in[WV_STR][L1_POS_STR]))], [])

        # AIE-array data movement for attention score calculation
        # L3 to L2 data movement
        q_l3l2_fifos = object_fifo(f"q_L3L2", shim_tiles[1], mem_tiles[1], fifo_depth, q_l1_ty_in,)
        k_l3l2_fifos = object_fifo(f"k_L3L2", shim_tiles[3], mem_tiles[3], fifo_depth, k_l1_ty_in,)
        # L2 to L1 data movement
        q_l2l1_fifos = object_fifo(f"q_L2L1", mem_tiles[1], core_tiles[0][1], fifo_depth, q_l1_ty_in,
                                    [
                                        (attn_score_mm_dims[0] // r, r * attn_score_mm_dims[1]),
                                        (attn_score_mm_dims[1] // s, s),
                                        (r, attn_score_mm_dims[1]),
                                        (s, 1),
                                    ],)
        k_l2l1_fifos = object_fifo(f"k_L2L1", mem_tiles[3], core_tiles[0][1], fifo_depth, k_l1_ty_in,
                                    [
                                        (attn_score_mm_dims[2] // t, t * attn_score_mm_dims[1]),
                                        (attn_score_mm_dims[1] // s, s),
                                        (t, attn_score_mm_dims[1]),
                                        (s, 1),
                                    ],)
        # L3 to L1 links
        object_fifo_link(q_l3l2_fifos, q_l2l1_fifos)
        object_fifo_link(k_l3l2_fifos, k_l2l1_fifos)
        # L1 to L2 data movement
        attn_score_l1l2_fifos = object_fifo(f"attn_score_L1L2", core_tiles[0][1], mem_tiles[3], fifo_depth, attn_score_l1_ty)
        # L2 to L3 data movement
        attn_score_l2l3_fifos = object_fifo(f"attn_score_L2L3", mem_tiles[3], shim_tiles[3], fifo_depth, attn_score_l1_ty,
                                    [
                                        (attn_score_mm_dims[0] // r, r * attn_score_mm_dims[2]),
                                        (r, t),
                                        (attn_score_mm_dims[2] // t, r * t),
                                        (t, 1),
                                    ],)
        # L1 to L3 links
        object_fifo_link(attn_score_l1l2_fifos, attn_score_l2l3_fifos)

        # Compute for Q, K, V projections
        for row, l1_pos in enumerate(right_mtx_in[WQ_STR][L1_POS_STR]):
            @core(
                core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]],
                f"mha_mm_{q_proj_dims[0]}x{q_proj_dims[1]}x{q_proj_dims[2]}_row_major.o",
                stack_size=0xD00
            )
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(N // q_proj_dims[2] // len(right_mtx_in[WQ_STR][L1_POS_STR])):
                        elem_q = q_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                        zero_projs(elem_q)
                        for _ in range_(K // q_proj_dims[1]):
                            elem_in_X = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wq = Wq_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                            matmul_projs(elem_in_X, elem_in_wq, elem_q)
                            Wq_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                            X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        q_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        for row, l1_pos in enumerate(right_mtx_in[WK_STR][L1_POS_STR]):
            @core(
                core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]],
                f"mha_mm_{k_proj_dims[0]}x{k_proj_dims[1]}x{k_proj_dims[2]}_row_major.o",
                stack_size=0xD00
            )
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(N // k_proj_dims[2] // len(right_mtx_in[WK_STR][L1_POS_STR])):
                        elem_k = k_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
                        zero_projs(elem_k)
                        for _ in range_(K // k_proj_dims[1]):
                            elem_in_xk = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wk = Wk_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                            matmul_projs(elem_in_xk, elem_in_wk, elem_k)
                            Wk_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                            X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        k_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        for row, l1_pos in enumerate(right_mtx_in[WV_STR][L1_POS_STR]):
            @core(
                core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]],
                f"mha_mm_{v_proj_dims[0]}x{v_proj_dims[1]}x{v_proj_dims[2]}_row_major.o",
                stack_size=0xD00
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

        # Compute for attention score
        @core(core_tiles[0][1], f"mha_mm_{attn_score_mm_dims[0]}x{attn_score_mm_dims[1]}x{attn_score_mm_dims[2]}_col_major.o", stack_size=0xD00)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                for _ in range_(H):
                    elem_attn_score = attn_score_l1l2_fifos.acquire(ObjectFifoPort.Produce, 1)
                    zero_attn_score(elem_attn_score)
                    for _ in range_(head_dim // attn_score_mm_dims[1]):
                        elem_in_q = q_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_k = k_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
                        matmul_attn_score(elem_in_q, elem_in_k, elem_attn_score)
                        q_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                        k_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
                    div_projs(elem_attn_score, elem_attn_score)
                    attn_score_l1l2_fifos.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[((M * K) + (2 * K * N),), np.dtype[dtype_in]],       # Order: X, W_Q, W_K
            np.ndarray[(K * N,), np.dtype[dtype_in]],                       # Order: W_V
            np.ndarray[((3 * M * N) + (H * M * M),), np.dtype[dtype_out]],  # Order: Q, K, V, attn_score
        )
        def sequence(A, B, C):
                # Send the data for calculating Q, K, V projections
                for row_offset in range(0, M, q_proj_dims[0]):
                    npu_dma_memcpy_nd(
                        metadata=X_l3l2_fifos,
                        bd_id=0,
                        mem=A,
                        offsets=[0, 0, 0, row_offset * K],
                        sizes=[N // q_proj_dims[2] // len(right_mtx_in[WQ_STR][L1_POS_STR]), K // q_proj_dims[1], q_proj_dims[0], q_proj_dims[1]],
                        strides=[0, q_proj_dims[1], K, 1],
                    )

                    for col_offset in range(0, N, q_proj_dims[2] * len(right_mtx_in[WQ_STR][L1_POS_STR])):
                        npu_dma_memcpy_nd(
                            metadata=Wq_l3l2_fifos,
                            bd_id=1,
                            mem=A,
                            offsets=[0, 0, 0, M * K + col_offset],
                            sizes=[K // q_proj_dims[1], len(right_mtx_in[WQ_STR][L1_POS_STR]), q_proj_dims[1], q_proj_dims[2]],
                            strides=[q_proj_dims[1] * N, q_proj_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=Wk_l3l2_fifos,
                            bd_id=2,
                            mem=A,
                            offsets=[0, 0, 0, M * K + K * N + col_offset],
                            sizes=[K // k_proj_dims[1], len(right_mtx_in[WK_STR][L1_POS_STR]), k_proj_dims[1], k_proj_dims[2]],
                            strides=[k_proj_dims[1] * N, k_proj_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=Wv_l3l2_fifos,
                            bd_id=3,
                            mem=B,
                            offsets=[0, 0, 0, col_offset],
                            sizes=[K // v_proj_dims[1], len(right_mtx_in[WV_STR][L1_POS_STR]), v_proj_dims[1], v_proj_dims[2]],
                            strides=[v_proj_dims[1] * N, v_proj_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=q_l2l3_fifos,
                            bd_id=4,
                            mem=C,
                            offsets=[0, 0, 0, row_offset * N + col_offset],
                            sizes=[1, len(right_mtx_in[WQ_STR][L1_POS_STR]), q_proj_dims[0], q_proj_dims[2]],
                            strides=[0, q_proj_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=k_l2l3_fifos,
                            bd_id=5,
                            mem=C,
                            offsets=[0, 0, 0, M * N + row_offset * N + col_offset],
                            sizes=[1, len(right_mtx_in[WK_STR][L1_POS_STR]), k_proj_dims[0], k_proj_dims[2]],
                            strides=[0, k_proj_dims[2], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=v_l2l3_fifos,
                            bd_id=6,
                            mem=C,
                            offsets=[0, 0, 0, 2 * M * N + row_offset * N + col_offset],
                            sizes=[1, len(right_mtx_in[WV_STR][L1_POS_STR]), v_proj_dims[0], v_proj_dims[2]],
                            strides=[0, v_proj_dims[2], N, 1],
                        )

                        dma_wait(q_l2l3_fifos, k_l2l3_fifos, v_l2l3_fifos)
                
                # Send the data for calculating attention scores
                for head in range(H):
                    for row_offset in range(0, M, attn_score_mm_dims[0]):
                        npu_dma_memcpy_nd(
                            metadata=q_l3l2_fifos,
                            bd_id=0,
                            mem=C,
                            offsets=[0, 0, 0, row_offset * N + head * head_dim],
                            sizes=[M // attn_score_mm_dims[2], head_dim // attn_score_mm_dims[1], attn_score_mm_dims[0], attn_score_mm_dims[1]],
                            strides=[0, attn_score_mm_dims[1], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=k_l3l2_fifos,
                            bd_id=1,
                            mem=C,
                            offsets=[0, 0, 0, M * N + head * head_dim],
                            sizes=[M // attn_score_mm_dims[2], head_dim // attn_score_mm_dims[1], attn_score_mm_dims[2], attn_score_mm_dims[1]],
                            strides=[attn_score_mm_dims[2] * N, attn_score_mm_dims[1], N, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=attn_score_l2l3_fifos,
                            bd_id=2,
                            mem=C,
                            offsets=[0, 0, 0, 3 * M * N + head * M * M + row_offset * M],
                            sizes=[1, M // attn_score_mm_dims[2], attn_score_mm_dims[0], attn_score_mm_dims[2]],
                            strides=[0, attn_score_mm_dims[2], M, 1],
                        )

                        dma_wait(attn_score_l2l3_fifos)


if __name__ == "__main__":
    main()
