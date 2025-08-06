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
    This design isolates a step in the Attention part of MHA. This is done by
    only running the zero function on subsequent steps, which 
    means most of the run time will be the load, compute, and store at the 
    relevant tile. The step isolated here will be the QK^T step. The subsequent
    steps will have objectfifos of size =
    16 (256 bits for 1 store per cycle / 16 bits per element),
    which when double buffered means the run time of the isolated step will 
    make up most of the run time.
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
            L1_POS_STR: [(0, 0), (1, 0), (2, 0), (3, 0), (0, 2), (1, 2)],
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

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

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

    q_proj_dims = (64, 64, 64)
    k_proj_dims = (64, 64, 64)
    v_proj_dims = (64, 64, 64)
    proj_dims = [q_proj_dims, k_proj_dims, v_proj_dims]

    attn_score_mm_dims = (16, 32, 256)
    softmax_dims = (attn_score_mm_dims[0], attn_score_mm_dims[2])
    attn_score_v_mm_dims = (softmax_dims[0], softmax_dims[1], 16)
    output_mm_dims = (attn_score_v_mm_dims[0], attn_score_v_mm_dims[2], 256)
    mha_dims = [attn_score_mm_dims, attn_score_v_mm_dims, output_mm_dims]
    if dev == "npu2" and dtype_in_str == "bf16":
        min_elems = 32
    elif dev == "npu" and dtype_in_str == "bf16":
        min_elems = 16
    else:
        raise AssertionError("Unsupported device and dtype combination for minimum elements")

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
    for dims in proj_dims + mha_dims:
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
    for dims in mha_dims:
        assert (
            N % (dims[2]) == 0
        ), """B must be tileable into (k, dims[2])-sized blocks"""
    assert (
        head_dim % attn_score_mm_dims[1] == 0
    ), """Head dimension must be divisible by left mtx col dim"""
    assert (M / attn_score_mm_dims[2] == 1), """Softmax tile col size must be the whole sequence length"""

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    if dev == "npu":
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu1
    else:
        if n_aie_cols == 4:
            dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        # # L2 tiles for Q, K, V projections
        # X_l2_ty = np.ndarray[(q_proj_dims[0] * q_proj_dims[1],), np.dtype[dtype_in]]
        # Wq_l2_ty = np.ndarray[(q_proj_dims[1] * q_proj_dims[2] * len(right_mtx_in[WQ_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        # Wk_l2_ty = np.ndarray[(k_proj_dims[1] * k_proj_dims[2] * len(right_mtx_in[WK_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        # Wv_l2_ty = np.ndarray[(v_proj_dims[1] * v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        # q_l2_ty = np.ndarray[(q_proj_dims[0] * q_proj_dims[2] * len(right_mtx_in[WQ_STR][L1_POS_STR]),), np.dtype[dtype_out]]
        # k_l2_ty = np.ndarray[(k_proj_dims[0] * k_proj_dims[2] * len(right_mtx_in[WK_STR][L1_POS_STR]),), np.dtype[dtype_out]]
        # v_l2_ty = np.ndarray[(v_proj_dims[0] * v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR]),), np.dtype[dtype_out]]

        # # L1 tiles for Q, K, V projections
        # X_l1_ty = np.ndarray[(q_proj_dims[0] * q_proj_dims[1],), np.dtype[dtype_in]]
        # Wq_l1_ty = np.ndarray[(q_proj_dims[1] * q_proj_dims[2],), np.dtype[dtype_in]]
        # Wk_l1_ty = np.ndarray[(k_proj_dims[1] * k_proj_dims[2],), np.dtype[dtype_in]]
        # Wv_l1_ty = np.ndarray[(v_proj_dims[1] * v_proj_dims[2],), np.dtype[dtype_in]]
        # q_l1_ty_out = np.ndarray[(q_proj_dims[0] * q_proj_dims[2],), np.dtype[dtype_out]]
        # k_l1_ty_out = np.ndarray[(k_proj_dims[0] * k_proj_dims[2],), np.dtype[dtype_out]]
        # v_l1_ty_out = np.ndarray[(v_proj_dims[0] * v_proj_dims[2],), np.dtype[dtype_out]]

        # L2/L1 tiles for multi-head attention
        q_l2_ty_in = np.ndarray[(attn_score_mm_dims[0] * attn_score_mm_dims[1] * len(left_mtx_in[Q_STR][L1_POS_STR]),), np.dtype[dtype_out]]
        k_l2_ty_in = np.ndarray[(attn_score_mm_dims[2] * attn_score_mm_dims[1] * len(right_mtx_in[K_STR][L1_POS_STR]),), np.dtype[dtype_out]]
        v_l2_ty_in = np.ndarray[(attn_score_v_mm_dims[1] * attn_score_v_mm_dims[2] * len(right_mtx_in[V_STR][L1_POS_STR]),), np.dtype[dtype_out]]
        wo_l2_ty_in = np.ndarray[(output_mm_dims[1] * output_mm_dims[2] * len(right_mtx_in[WO_STR][L1_POS_STR]),), np.dtype[dtype_in]]
        q_l1_ty_in = np.ndarray[(attn_score_mm_dims[0] * attn_score_mm_dims[1],), np.dtype[dtype_out]]
        k_l1_ty_in = np.ndarray[(attn_score_mm_dims[2] * attn_score_mm_dims[1],), np.dtype[dtype_out]]
        v_l1_ty_in = np.ndarray[(attn_score_v_mm_dims[1] * attn_score_v_mm_dims[2],), np.dtype[dtype_out]]
        Wo_l1_ty_in = np.ndarray[(output_mm_dims[1] * output_mm_dims[2],), np.dtype[dtype_in]]
        attn_score_l1_ty = np.ndarray[(attn_score_mm_dims[0] * attn_score_mm_dims[2],), np.dtype[dtype_out]]
        softmax_l1_ty = np.ndarray[(softmax_dims[0] * softmax_dims[1],), np.dtype[dtype_out]]
        attn_score_v_l1_ty = np.ndarray[(attn_score_v_mm_dims[0] * attn_score_v_mm_dims[2],), np.dtype[dtype_out]]
        output_l1_ty = np.ndarray[(output_mm_dims[0] * output_mm_dims[2],), np.dtype[dtype_out]]
        min_l1_ty = np.ndarray[(min_elems,), np.dtype[dtype_out]] # Used for the subsequent steps to minimize time added from irrelevant tiles

        # AIE Core Function declarations
        # Last part of the name is whether the right matrix is row major
        matmul_vectorized_func_name = f"matmul_{dtype_in_str}_{dtype_out_str}"
        row_major = 1
        col_major = 0
        # zero_projs = external_func(
        #     f"zero_{dtype_out_str}_{q_proj_dims[0]}_{q_proj_dims[1]}_{q_proj_dims[2]}_{row_major}",
        #     inputs=[q_l1_ty_out]
        # )
        # matmul_projs = external_func(
        #     matmul_vectorized_func_name + f"_{q_proj_dims[0]}_{q_proj_dims[1]}_{q_proj_dims[2]}_{row_major}",
        #     inputs=[X_l1_ty, Wq_l1_ty, q_l1_ty_out]
        # )
        zero_attn_score = external_func(
            f"zero_{dtype_out_str}_{attn_score_mm_dims[0]}_{attn_score_mm_dims[2]}",
            inputs=[attn_score_l1_ty]
        )
        # matmul_attn_score = external_func(
        #     matmul_vectorized_func_name + f"_{attn_score_mm_dims[0]}_{attn_score_mm_dims[1]}_{attn_score_mm_dims[2]}_{col_major}",
        #     inputs=[q_l1_ty_in, k_l1_ty_in, attn_score_l1_ty]
        # )
        # div_projs = external_func(
        #     f"div_2d_bf16", 
        #     inputs=[attn_score_l1_ty, attn_score_l1_ty]
        # )
        # zero_softmax_attn = external_func(
        #     f"zero_{dtype_out_str}_{softmax_dims[0]}_{softmax_dims[1]}",
        #     inputs=[softmax_l1_ty]
        # )
        # softmax_attn_score = external_func(
        #     f"softmax_{dtype_in_str}",
        #     inputs=[softmax_l1_ty, softmax_l1_ty],
        # )
        zero_attn_score_v = external_func(
            f"zero_{dtype_out_str}_{attn_score_v_mm_dims[0]}_{attn_score_v_mm_dims[2]}",
            inputs=[attn_score_v_l1_ty]
        )
        matmul_attn_score_v = external_func(
            matmul_vectorized_func_name + f"_{attn_score_v_mm_dims[0]}_{attn_score_v_mm_dims[1]}_{attn_score_v_mm_dims[2]}_{row_major}",
            inputs=[softmax_l1_ty, v_l1_ty_in, attn_score_v_l1_ty],
        )
        # zero_output = external_func(
        #     f"zero_{dtype_out_str}_{output_mm_dims[0]}_{output_mm_dims[2]}",
        #     inputs=[output_l1_ty]
        # )
        # matmul_output = external_func(
        #     matmul_vectorized_func_name + f"_{output_mm_dims[0]}_{output_mm_dims[1]}_{output_mm_dims[2]}_{row_major}",
        #     inputs=[attn_score_v_l1_ty, Wo_l1_ty_in, output_l1_ty],
        # )
        # zero_at_add = external_func(
        #     f"zero_{dtype_out_str}_{output_mm_dims[0]}_{output_mm_dims[2]}",
        #     inputs=[output_l1_ty]
        # )
        # add_output = external_func(
        #     f"eltwise_add_{dtype_out_str}_vector",
        #     inputs=[output_l1_ty, output_l1_ty, output_l1_ty],
        # )
        zero_min = external_func(
            f"zero_{dtype_out_str}_{min_elems}_1",
            inputs=[min_l1_ty]
        )

        if dev == "npu":
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 4th columns
        else:
            tiles = [[tile(col + 0, row) for col in range(0, n_aie_cols)] for row in range(0, 6)] # 1st to 4th columns
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # # AIE-array data movement for Q, K, V projections
        # # L3 to L2 data movement
        # X_l3l2_fifos = object_fifo(
        #     f"X_L3L2", 
        #     shim_tiles[left_mtx_in[X_STR][L3_POS_STR]], 
        #     mem_tiles[left_mtx_in[X_STR][L2_POS_STR]], 
        #     fifo_depth,
        #     X_l2_ty
        # )
        # Wq_l3l2_fifos = object_fifo(
        #     f"Wq_L3L2", 
        #     shim_tiles[right_mtx_in[WQ_STR][L3_POS_STR]], 
        #     mem_tiles[right_mtx_in[WQ_STR][L2_POS_STR]], 
        #     fifo_depth,
        #     Wq_l2_ty
        # )
        # Wk_l3l2_fifos = object_fifo(
        #     f"Wk_L3L2", 
        #     shim_tiles[right_mtx_in[WK_STR][L3_POS_STR]], 
        #     mem_tiles[right_mtx_in[WK_STR][L2_POS_STR]], 
        #     fifo_depth,
        #     Wk_l2_ty
        # )
        # Wv_l3l2_fifos = object_fifo(
        #     f"Wv_L3L2", 
        #     shim_tiles[right_mtx_in[WV_STR][L3_POS_STR]], 
        #     mem_tiles[right_mtx_in[WV_STR][L2_POS_STR]], 
        #     fifo_depth,
        #     Wv_l2_ty
        # )
        # # L2 to L1 data movement
        # X_l2l1_fifos = object_fifo(
        #     f"X_L2L1", 
        #     mem_tiles[left_mtx_in[X_STR][L2_POS_STR]], 
        #     [core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]] for l1_pos in left_mtx_in[X_STR][L1_POS_STR]], 
        #     fifo_depth,
        #     X_l1_ty,
        #     [
        #         (q_proj_dims[0] // r, r * q_proj_dims[1]),
        #         (q_proj_dims[1] // s, s),
        #         (r, q_proj_dims[1]),
        #         (s, 1),
        #     ],
        # )
        # Wq_l2l1_fifos = [object_fifo(
        #     f"Wq_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
        #     mem_tiles[right_mtx_in[WQ_STR][L2_POS_STR]], 
        #     core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
        #     fifo_depth,
        #     Wq_l1_ty,
        #     [
        #         (q_proj_dims[1] // s, s * q_proj_dims[2]),
        #         (q_proj_dims[2] // t, t),
        #         (s, q_proj_dims[2]),
        #         (t, 1),
        #     ],
        # ) for l1_pos in right_mtx_in[WQ_STR][L1_POS_STR]]
        # Wk_l2l1_fifos = [object_fifo(
        #     f"Wk_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
        #     mem_tiles[right_mtx_in[WK_STR][L2_POS_STR]], 
        #     core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
        #     fifo_depth,
        #     Wk_l1_ty,
        #     [
        #         (k_proj_dims[1] // s, s * k_proj_dims[2]),
        #         (k_proj_dims[2] // t, t),
        #         (s, k_proj_dims[2]),
        #         (t, 1),
        #     ],
        # ) for l1_pos in right_mtx_in[WK_STR][L1_POS_STR]]
        # Wv_l2l1_fifos = [object_fifo(
        #     f"Wv_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
        #     mem_tiles[right_mtx_in[WV_STR][L2_POS_STR]], 
        #     core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
        #     fifo_depth,
        #     Wv_l1_ty,
        #     [
        #         (v_proj_dims[1] // s, s * v_proj_dims[2]),
        #         (v_proj_dims[2] // t, t),
        #         (s, v_proj_dims[2]),
        #         (t, 1),
        #     ],
        # ) for l1_pos in right_mtx_in[WV_STR][L1_POS_STR]]
        # # L3 to L1 links
        # object_fifo_link(X_l3l2_fifos, X_l2l1_fifos)
        # object_fifo_link(Wq_l3l2_fifos, Wq_l2l1_fifos, [], [q_proj_dims[1] * q_proj_dims[2] * i for i in range(len(right_mtx_in[WQ_STR][L1_POS_STR]))])
        # object_fifo_link(Wk_l3l2_fifos, Wk_l2l1_fifos, [], [k_proj_dims[1] * k_proj_dims[2] * i for i in range(len(right_mtx_in[WK_STR][L1_POS_STR]))])
        # object_fifo_link(Wv_l3l2_fifos, Wv_l2l1_fifos, [], [v_proj_dims[1] * v_proj_dims[2] * i for i in range(len(right_mtx_in[WV_STR][L1_POS_STR]))])
        # # L1 to L2 data movement
        # q_l1l2_fifos = [object_fifo(
        #     f"q_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
        #     core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
        #     mem_tiles[l3_fuse_mtx_out[Q_STR][L2_POS_STR]], 
        #     fifo_depth, 
        #     q_l1_ty_out,
        # ) for l1_pos in right_mtx_in[WQ_STR][L1_POS_STR]]
        # k_l1l2_fifos = [object_fifo(
        #     f"k_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
        #     core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
        #     mem_tiles[l3_fuse_mtx_out[K_STR][L2_POS_STR]], 
        #     fifo_depth, 
        #     k_l1_ty_out,
        # ) for l1_pos in right_mtx_in[WK_STR][L1_POS_STR]]
        # v_l1l2_fifos = [object_fifo(
        #     f"v_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
        #     core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
        #     mem_tiles[l3_fuse_mtx_out[V_STR][L2_POS_STR]], 
        #     fifo_depth, 
        #     v_l1_ty_out,
        # ) for l1_pos in right_mtx_in[WV_STR][L1_POS_STR]]
        # # L2 to L3 data movement
        # q_l2l3_fifos = object_fifo(
        #     f"q_L2L3", 
        #     mem_tiles[l3_fuse_mtx_out[Q_STR][L2_POS_STR]], 
        #     shim_tiles[l3_fuse_mtx_out[Q_STR][L3_POS_STR]], 
        #     fifo_depth, 
        #     q_l2_ty,
        #     [
        #         (q_proj_dims[0] // r, r * q_proj_dims[2]),
        #         (r, t),
        #         (q_proj_dims[2] // t, r * t),
        #         (t, 1),
        #     ],
        # )
        # k_l2l3_fifos = object_fifo(
        #     f"k_L2L3", 
        #     mem_tiles[l3_fuse_mtx_out[K_STR][L2_POS_STR]], 
        #     shim_tiles[l3_fuse_mtx_out[K_STR][L3_POS_STR]], 
        #     fifo_depth, 
        #     k_l2_ty,
        #     [
        #         (k_proj_dims[0] // r, r * k_proj_dims[2]),
        #         (r, t),
        #         (k_proj_dims[2] // t, r * t),
        #         (t, 1),
        #     ],
        # )
        # v_l2l3_fifos = object_fifo(
        #     f"v_L2L3", 
        #     mem_tiles[l3_fuse_mtx_out[V_STR][L2_POS_STR]], 
        #     shim_tiles[l3_fuse_mtx_out[V_STR][L3_POS_STR]], 
        #     fifo_depth, 
        #     v_l2_ty,
        #     [
        #         (v_proj_dims[0] // r, r * v_proj_dims[2]),
        #         (r, t),
        #         (v_proj_dims[2] // t, r * t),
        #         (t, 1),
        #     ],
        # )
        # # L1 to L3 links
        # object_fifo_link(q_l1l2_fifos, q_l2l3_fifos, [q_proj_dims[0] * q_proj_dims[2] * i for i in range(len(right_mtx_in[WQ_STR][L1_POS_STR]))], [])
        # object_fifo_link(k_l1l2_fifos, k_l2l3_fifos, [k_proj_dims[0] * k_proj_dims[2] * i for i in range(len(right_mtx_in[WK_STR][L1_POS_STR]))], [])
        # object_fifo_link(v_l1l2_fifos, v_l2l3_fifos, [v_proj_dims[1] * v_proj_dims[2] * i for i in range(len(right_mtx_in[WV_STR][L1_POS_STR]))], [])

        # AIE-array data movement for attention score calculation
        # L3 to L2 data movement
        q_l3l2_fifos = object_fifo(
            f"q_L3L2", 
            shim_tiles[left_mtx_in[Q_STR][L3_POS_STR]], 
            mem_tiles[left_mtx_in[Q_STR][L2_POS_STR]], 
            fifo_depth, 
            q_l2_ty_in,
        )
        k_l3l2_fifos = object_fifo(
            f"k_L3L2", 
            shim_tiles[right_mtx_in[K_STR][L3_POS_STR]], 
            mem_tiles[right_mtx_in[K_STR][L2_POS_STR]], 
            fifo_depth, 
            k_l2_ty_in,
        )
        v_l3l2_fifos = object_fifo(
            f"v_L3L2", 
            shim_tiles[right_mtx_in[V_STR][L3_POS_STR]], 
            mem_tiles[right_mtx_in[V_STR][L2_POS_STR]], 
            fifo_depth, 
            v_l2_ty_in,
        )
        # Wo_l3l2_fifos = object_fifo(
        #     f"Wo_L3L2", 
        #     shim_tiles[right_mtx_in[WO_STR][L3_POS_STR]], 
        #     mem_tiles[right_mtx_in[WO_STR][L2_POS_STR]], 
        #     fifo_depth,
        #     wo_l2_ty_in,
        # )
        # L2 to L1 data movement
        q_l2l1_fifos = [object_fifo(
            f"q_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
            mem_tiles[left_mtx_in[Q_STR][L2_POS_STR]], 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            fifo_depth, 
            q_l1_ty_in,
            [
                (attn_score_mm_dims[0] // r, r * attn_score_mm_dims[1]),
                (attn_score_mm_dims[1] // s, s),
                (r, attn_score_mm_dims[1]),
                (s, 1),
            ],
        ) for l1_pos in left_mtx_in[Q_STR][L1_POS_STR]]
        k_l2l1_fifos = [object_fifo(
            f"k_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
            mem_tiles[right_mtx_in[K_STR][L2_POS_STR]], 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            fifo_depth, 
            k_l1_ty_in,
            [
                (attn_score_mm_dims[2] // t, t * attn_score_mm_dims[1]),
                (attn_score_mm_dims[1] // s, s),
                (t, attn_score_mm_dims[1]),
                (s, 1),
            ],
        ) for l1_pos in right_mtx_in[K_STR][L1_POS_STR]]
        v_l2l1_fifos = [object_fifo(
            f"v_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}",
            mem_tiles[right_mtx_in[V_STR][L2_POS_STR]], 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            fifo_depth, 
            v_l1_ty_in,
            [
                (attn_score_v_mm_dims[1] // t, t * attn_score_v_mm_dims[2]),
                (attn_score_v_mm_dims[2] // s, s),
                (t, attn_score_v_mm_dims[2]),
                (s, 1),
            ],
        ) for l1_pos in right_mtx_in[V_STR][L1_POS_STR]]
        # Wo_l2l1_fifos = [object_fifo(
        #     f"Wo_L2L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
        #     mem_tiles[right_mtx_in[WO_STR][L2_POS_STR]], 
        #     core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
        #     fifo_depth, 
        #     Wo_l1_ty_in,
        #     [
        #         (output_mm_dims[1] // s, s * output_mm_dims[2]),
        #         (output_mm_dims[2] // t, t),
        #         (s, output_mm_dims[2]),
        #         (t, 1),
        #     ],
        # ) for l1_pos in right_mtx_in[WO_STR][L1_POS_STR]]
        # L3 to L1 links
        object_fifo_link(q_l3l2_fifos, q_l2l1_fifos, [], [attn_score_mm_dims[0] * attn_score_mm_dims[1] * i for i in range(len(left_mtx_in[Q_STR][L1_POS_STR]))])
        object_fifo_link(k_l3l2_fifos, k_l2l1_fifos, [], [attn_score_mm_dims[1] * attn_score_mm_dims[2] * i for i in range(len(right_mtx_in[K_STR][L1_POS_STR]))])
        object_fifo_link(v_l3l2_fifos, v_l2l1_fifos, [], [attn_score_v_mm_dims[1] * attn_score_v_mm_dims[2] * i for i in range(len(right_mtx_in[V_STR][L1_POS_STR]))])
        # object_fifo_link(Wo_l3l2_fifos, Wo_l2l1_fifos, [], [output_mm_dims[1] * output_mm_dims[2] * i for i in range(len(right_mtx_in[WO_STR][L1_POS_STR]))])
        # L1 to L2 to L1 data movement, need to send tile back to L2 since we need 4-D transformation
        attn_score_l1l2_fifos = [object_fifo(
            f"attn_score_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}_{l1_fuse_mtx_in[ATTN_SCORE_STR][L2_POS_STR]}", 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            mem_tiles[l1_fuse_mtx_in[ATTN_SCORE_STR][L2_POS_STR]], 
            fifo_depth, 
            attn_score_l1_ty,
        ) for l1_pos in left_mtx_in[Q_STR][L1_POS_STR]]
        attn_score_l2l1_fifos = [object_fifo(
            f"attn_score_L2L1_{l1_fuse_mtx_in[ATTN_SCORE_STR][L2_POS_STR]}_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
            mem_tiles[l1_fuse_mtx_in[ATTN_SCORE_STR][L2_POS_STR]], 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            fifo_depth, 
            attn_score_l1_ty,
            [
                (softmax_dims[0] // r, r * softmax_dims[1]),
                (r, t),
                (softmax_dims[1] // t, r * t),
                (t, 1),
            ],
        ) for l1_pos in l1_fuse_mtx_in[ATTN_SCORE_STR][L1_POS_STR]]
        for l1_src, l1_dst in zip(attn_score_l1l2_fifos, attn_score_l2l1_fifos):
            object_fifo_link(l1_src, l1_dst)
        # L1 to L2 to L1 data movement, need to send tile back to L2 since we need 4-D transformation
        softmax_l1l2_fifos = [object_fifo(
            f"softmax_L1L2_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}_{l1_fuse_mtx_in[SOFTMAX_STR][L2_POS_STR]}", 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            mem_tiles[l1_fuse_mtx_in[SOFTMAX_STR][L2_POS_STR]], 
            fifo_depth, 
            softmax_l1_ty,
        ) for l1_pos in l1_fuse_mtx_in[ATTN_SCORE_STR][L1_POS_STR]]
        softmax_l2l1_fifos = [object_fifo(
            f"softmax_L2L1_{l1_fuse_mtx_in[ATTN_SCORE_STR][L2_POS_STR]}_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}", 
            mem_tiles[l1_fuse_mtx_in[ATTN_SCORE_STR][L2_POS_STR]], 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            fifo_depth, 
            softmax_l1_ty,
            [
                (attn_score_v_mm_dims[0] // r, r * attn_score_v_mm_dims[1]),
                (attn_score_v_mm_dims[1] // s, s),
                (r, attn_score_v_mm_dims[1]),
                (s, 1),
            ],
        ) for l1_pos in l1_fuse_mtx_in[SOFTMAX_STR][L1_POS_STR]]
        for l1_src, l1_dst in zip(softmax_l1l2_fifos, softmax_l2l1_fifos):
            object_fifo_link(l1_src, l1_dst)
        # L1 to L1 data movement 
        attn_score_v_l1l1_fifos = [object_fifo(
            f"attn_score_v_L1L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}_{l1_dst[ROW_IDX]}_{l1_dst[COL_IDX]}", 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            core_tiles[l1_dst[ROW_IDX]][l1_dst[COL_IDX]], 
            fifo_depth, 
            attn_score_v_l1_ty,
        ) for l1_pos, l1_dst in zip(l1_fuse_mtx_in[SOFTMAX_STR][L1_POS_STR], l1_fuse_mtx_in[ATTN_SCORE_V_STR][L1_POS_STR])]
        output_l1l1_fifos = [object_fifo(
            f"output_L1L1_{l1_pos[ROW_IDX]}_{l1_pos[COL_IDX]}_{l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][ROW_IDX]}_{l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][COL_IDX]}", 
            core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], 
            core_tiles[l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][ROW_IDX]][l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][COL_IDX]], 
            fifo_depth, 
            min_l1_ty,
        ) for l1_pos in l1_fuse_mtx_in[ATTN_SCORE_V_STR][L1_POS_STR]]
        # L1 to L2 data movement
        output_l1l2_fifos = object_fifo(
            f"output_L1L2",
            core_tiles[l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][ROW_IDX]][l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][COL_IDX]],
            mem_tiles[l3_fuse_mtx_out[OUTPUT_STR][L2_POS_STR]],
            fifo_depth,
            min_l1_ty,
        )
        # L2 to L3 data movement
        output_l2l3_fifos = object_fifo(
            f"output_L2L3", 
            mem_tiles[l3_fuse_mtx_out[OUTPUT_STR][L2_POS_STR]], 
            shim_tiles[l3_fuse_mtx_out[OUTPUT_STR][L3_POS_STR]], 
            fifo_depth, 
            min_l1_ty,
            # [
            #     (output_mm_dims[0] // r, r * output_mm_dims[2]),
            #     (r, t),
            #     (output_mm_dims[2] // t, r * t),
            #     (t, 1),
            # ],
        )
        # L1 to L3 links
        object_fifo_link(output_l1l2_fifos, output_l2l3_fifos)

        # Compute for Q, K, V projections
        # for row, l1_pos in enumerate(right_mtx_in[WQ_STR][L1_POS_STR]):
        #     @core(
        #         core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]],
        #         f"mha_mm_{q_proj_dims[0]}x{q_proj_dims[1]}x{q_proj_dims[2]}_row_major.o",
        #         stack_size=0xD00
        #     )
        #     def core_body():
        #         for _ in range_(0xFFFFFFFF):
        #             for _ in range_(N // q_proj_dims[2] // len(right_mtx_in[WQ_STR][L1_POS_STR])):
        #                 elem_q = q_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
        #                 zero_projs(elem_q)
        #                 for _ in range_(K // q_proj_dims[1]):
        #                     elem_in_X = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
        #                     elem_in_wq = Wq_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
        #                     matmul_projs(elem_in_X, elem_in_wq, elem_q)
        #                     Wq_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
        #                     X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
        #                 q_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        # for row, l1_pos in enumerate(right_mtx_in[WK_STR][L1_POS_STR]):
        #     @core(
        #         core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]],
        #         f"mha_mm_{k_proj_dims[0]}x{k_proj_dims[1]}x{k_proj_dims[2]}_row_major.o",
        #         stack_size=0xD00
        #     )
        #     def core_body():
        #         for _ in range_(0xFFFFFFFF):
        #             for _ in range_(N // k_proj_dims[2] // len(right_mtx_in[WK_STR][L1_POS_STR])):
        #                 elem_k = k_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
        #                 zero_projs(elem_k)
        #                 for _ in range_(K // k_proj_dims[1]):
        #                     elem_in_xk = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
        #                     elem_in_wk = Wk_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
        #                     matmul_projs(elem_in_xk, elem_in_wk, elem_k)
        #                     Wk_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
        #                     X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
        #                 k_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        # for row, l1_pos in enumerate(right_mtx_in[WV_STR][L1_POS_STR]):
        #     @core(
        #         core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]],
        #         f"mha_mm_{v_proj_dims[0]}x{v_proj_dims[1]}x{v_proj_dims[2]}_row_major.o",
        #         stack_size=0xD00
        #     )
        #     def core_body():
        #         for _ in range_(0xFFFFFFFF):
        #             for _ in range_(N // v_proj_dims[2] // len(right_mtx_in[WV_STR][L1_POS_STR])):
        #                 elem_v = v_l1l2_fifos[row].acquire(ObjectFifoPort.Produce, 1)
        #                 zero_projs(elem_v)
        #                 for _ in range_(K // v_proj_dims[1]):
        #                     elem_in_xv = X_l2l1_fifos.acquire(ObjectFifoPort.Consume, 1)
        #                     elem_in_wv = Wv_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
        #                     matmul_projs(elem_in_xv, elem_in_wv, elem_v)
        #                     Wv_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
        #                     X_l2l1_fifos.release(ObjectFifoPort.Consume, 1)
        #                 v_l1l2_fifos[row].release(ObjectFifoPort.Produce, 1)

        # Compute for attention score
        for head, l1_pos in enumerate(left_mtx_in[Q_STR][L1_POS_STR]):
            @core(core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], f"mha_mm_{attn_score_mm_dims[0]}x{attn_score_mm_dims[1]}x{attn_score_mm_dims[2]}_col_major.o", stack_size=0xD00)
            def core_body():
                # Produce the data right away since we don't want the execution here to affect the run time
                # Doing it this way since we still need to consume the data from shim
                for _ in range_(0xFFFFFFFF):
                    elem_attn_score = attn_score_l1l2_fifos[head].acquire(ObjectFifoPort.Produce, 1)
                    zero_attn_score(elem_attn_score)
                    attn_score_l1l2_fifos[head].release(ObjectFifoPort.Produce, 1)
                    for _ in range_(H // len(left_mtx_in[Q_STR][L1_POS_STR])):
                        # elem_attn_score = attn_score_l1l2_fifos[head].acquire(ObjectFifoPort.Produce, 1)
                        # zero_attn_score(elem_attn_score)
                        for _ in range_(head_dim // attn_score_mm_dims[1]):
                            elem_in_q = q_l2l1_fifos[head].acquire(ObjectFifoPort.Consume, 1)
                            elem_in_k = k_l2l1_fifos[head].acquire(ObjectFifoPort.Consume, 1)
                            # matmul_attn_score(elem_in_q, elem_in_k, elem_attn_score)
                            q_l2l1_fifos[head].release(ObjectFifoPort.Consume, 1)
                            k_l2l1_fifos[head].release(ObjectFifoPort.Consume, 1)
                        # div_projs(elem_attn_score, elem_attn_score)
                        # attn_score_l1l2_fifos[head].release(ObjectFifoPort.Produce, 1)

        # Apply softmax to attention scores
        for head, l1_pos in enumerate(l1_fuse_mtx_in[ATTN_SCORE_STR][L1_POS_STR]):
            @core(core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], f"mha_softmax_{softmax_dims[0]}x{softmax_dims[1]}.o", stack_size=0xD00) # Make sure to use the bundled obj file, not the softmax-only obj file
            def core_body():
                # Produce the data right away since we don't want the execution here to affect the run time
                for _ in range_(0xFFFFFFFF):
                    # for _ in range_(H // len(l1_fuse_mtx_in[ATTN_SCORE_STR][L1_POS_STR])):
                    for _ in range_(1):
                        elem_attn_score = softmax_l1l2_fifos[head].acquire(ObjectFifoPort.Produce, 1)
                        zero_attn_score(elem_attn_score)
                        elem_o1 = attn_score_l2l1_fifos[head].acquire(ObjectFifoPort.Consume, 1)
                        # softmax_attn_score(elem_o1, elem_attn_score)
                        attn_score_l2l1_fifos[head].release(ObjectFifoPort.Consume, 1)
                        softmax_l1l2_fifos[head].release(ObjectFifoPort.Produce, 1)

        # Calculate attention score * V        
        for head, l1_pos in enumerate(l1_fuse_mtx_in[SOFTMAX_STR][L1_POS_STR]):
            @core(core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], f"mha_mm_{attn_score_v_mm_dims[0]}x{attn_score_v_mm_dims[1]}x{attn_score_v_mm_dims[2]}_row_major.o", stack_size=0xD00)
            def core_body():
                # This core is what we're measuring the runtime of. We consume the previous step's output only once
                # so that the data movement within the previous core doesn't affect the measured runtime.
                for _ in range_(0xFFFFFFFF):
                    elem_in_softmax = softmax_l2l1_fifos[head].acquire(ObjectFifoPort.Consume, 1) 
                    for _ in range_(H // len(l1_fuse_mtx_in[SOFTMAX_STR][L1_POS_STR])):
                        # elem_in_softmax = softmax_l2l1_fifos[head].acquire(ObjectFifoPort.Consume, 1)
                        for _ in range_(head_dim // attn_score_v_mm_dims[2]):
                            elem_attn_score_v = attn_score_v_l1l1_fifos[head].acquire(ObjectFifoPort.Produce, 1)
                            zero_attn_score_v(elem_attn_score_v)
                            elem_in_v = v_l2l1_fifos[head].acquire(ObjectFifoPort.Consume, 1)
                            matmul_attn_score_v(elem_in_softmax, elem_in_v, elem_attn_score_v)
                            v_l2l1_fifos[head].release(ObjectFifoPort.Consume, 1)
                            attn_score_v_l1l1_fifos[head].release(ObjectFifoPort.Produce, 1)
                        # softmax_l2l1_fifos[head].release(ObjectFifoPort.Consume, 1)
                    softmax_l2l1_fifos[head].release(ObjectFifoPort.Consume, 1)

        for head, l1_pos in enumerate(l1_fuse_mtx_in[ATTN_SCORE_V_STR][L1_POS_STR]):
            @core(core_tiles[l1_pos[ROW_IDX]][l1_pos[COL_IDX]], f"mha_mm_{output_mm_dims[0]}x{output_mm_dims[1]}x{output_mm_dims[2]}_row_major.o", stack_size=0xD00)
            def core_body():
                # We want this core to wait for each tile to be ready so that the measured runtime reflects
                # the time it takes for the previous step to finish
                # Since there's no compute done here, it shouldn't affect the runtime
                for _ in range_(0xFFFFFFFF):
                    elem_output = output_l1l1_fifos[head].acquire(ObjectFifoPort.Produce, 1)
                    zero_min(elem_output)
                    for _ in range_(K // output_mm_dims[1] // len(l1_fuse_mtx_in[ATTN_SCORE_V_STR][L1_POS_STR])):
                        elem_in_o3 = attn_score_v_l1l1_fifos[head].acquire(ObjectFifoPort.Consume, 1)
                    #     elem_in_wo = Wo_l2l1_fifos[head].acquire(ObjectFifoPort.Consume, 1)
                    #     matmul_output(elem_in_o3, elem_in_wo, elem_output)
                    #     Wo_l2l1_fifos[head].release(ObjectFifoPort.Consume, 1)
                        attn_score_v_l1l1_fifos[head].release(ObjectFifoPort.Consume, 1)
                    output_l1l1_fifos[head].release(ObjectFifoPort.Produce, 1)

        @core(core_tiles[l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][ROW_IDX]][l1_fuse_mtx_in[ACCUM_STR][L1_POS_STR][0][COL_IDX]], f"mha_add_{output_mm_dims[0]}x{output_mm_dims[2]}.o", stack_size=0xD00)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                elem_output = output_l1l2_fifos.acquire(ObjectFifoPort.Produce, 1)
                zero_min(elem_output)
                elem_in_1 = output_l1l1_fifos[0].acquire(ObjectFifoPort.Consume, 1)
                elem_in_2 = output_l1l1_fifos[1].acquire(ObjectFifoPort.Consume, 1)
                # add_output(elem_in_1, elem_in_2, elem_output)
                output_l1l1_fifos[0].release(ObjectFifoPort.Consume, 1)
                output_l1l1_fifos[1].release(ObjectFifoPort.Consume, 1)
                output_l1l2_fifos.release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[((M * K) + (2 * K * N),), np.dtype[dtype_in]],       # Order: X, W_Q, W_K
            np.ndarray[(2 * K * N,), np.dtype[dtype_in]],                   # Order: W_V, W_O
            np.ndarray[((4 * M * N),), np.dtype[dtype_out]],                # Order: Q, K, V, attn_score_v
        )
        def sequence(A, B, C):
            # Send the data for calculating Q, K, V projections
            # for row_offset in range(0, M, q_proj_dims[0]):
            #     npu_dma_memcpy_nd(
            #         metadata=X_l3l2_fifos,
            #         bd_id=0,
            #         mem=A,
            #         offsets=[0, 0, 0, row_offset * K],
            #         sizes=[N // q_proj_dims[2] // len(right_mtx_in[WQ_STR][L1_POS_STR]), K // q_proj_dims[1], q_proj_dims[0], q_proj_dims[1]],
            #         strides=[0, q_proj_dims[1], K, 1],
            #     )

            #     for col_offset_q, col_offset_k, col_offset_v in zip(
            #         range(0, N, q_proj_dims[2] * len(right_mtx_in[WQ_STR][L1_POS_STR])),
            #         range(0, N, k_proj_dims[2] * len(right_mtx_in[WK_STR][L1_POS_STR])),
            #         range(0, N, v_proj_dims[2] * len(right_mtx_in[WV_STR][L1_POS_STR]))
            #     ):
            #         npu_dma_memcpy_nd(
            #             metadata=Wq_l3l2_fifos,
            #             bd_id=1,
            #             mem=A,
            #             offsets=[0, 0, 0, M * K + col_offset_q],
            #             sizes=[K // q_proj_dims[1], len(right_mtx_in[WQ_STR][L1_POS_STR]), q_proj_dims[1], q_proj_dims[2]],
            #             strides=[q_proj_dims[1] * N, q_proj_dims[2], N, 1],
            #         )

            #         npu_dma_memcpy_nd(
            #             metadata=Wk_l3l2_fifos,
            #             bd_id=2,
            #             mem=A,
            #             offsets=[0, 0, 0, M * K + K * N + col_offset_k],
            #             sizes=[K // k_proj_dims[1], len(right_mtx_in[WK_STR][L1_POS_STR]), k_proj_dims[1], k_proj_dims[2]],
            #             strides=[k_proj_dims[1] * N, k_proj_dims[2], N, 1],
            #         )

            #         npu_dma_memcpy_nd(
            #             metadata=Wv_l3l2_fifos,
            #             bd_id=3,
            #             mem=B,
            #             offsets=[0, 0, 0, col_offset_v],
            #             sizes=[K // v_proj_dims[1], len(right_mtx_in[WV_STR][L1_POS_STR]), v_proj_dims[1], v_proj_dims[2]],
            #             strides=[v_proj_dims[1] * N, v_proj_dims[2], N, 1],
            #         )

            #         npu_dma_memcpy_nd(
            #             metadata=q_l2l3_fifos,
            #             bd_id=4,
            #             mem=C,
            #             offsets=[0, 0, 0, row_offset * N + col_offset_q],
            #             sizes=[1, len(right_mtx_in[WQ_STR][L1_POS_STR]), q_proj_dims[0], q_proj_dims[2]],
            #             strides=[0, q_proj_dims[2], N, 1],
            #         )

            #         npu_dma_memcpy_nd(
            #             metadata=k_l2l3_fifos,
            #             bd_id=5,
            #             mem=C,
            #             offsets=[0, 0, 0, M * N + row_offset * N + col_offset_k],
            #             sizes=[1, len(right_mtx_in[WK_STR][L1_POS_STR]), k_proj_dims[0], k_proj_dims[2]],
            #             strides=[0, k_proj_dims[2], N, 1],
            #         )

            #         npu_dma_memcpy_nd(
            #             metadata=v_l2l3_fifos,
            #             bd_id=6,
            #             mem=C,
            #             offsets=[0, 0, 0, 2 * M * N + row_offset * N + col_offset_v],
            #             sizes=[1, len(right_mtx_in[WV_STR][L1_POS_STR]), v_proj_dims[0], v_proj_dims[2]],
            #             strides=[0, v_proj_dims[2], N, 1],
            #         )

            #         dma_wait(q_l2l3_fifos, k_l2l3_fifos, v_l2l3_fifos)
            
            # Send the data for calculating attention scores
            for row_offset in range(0, M, output_mm_dims[0]):
                for col_offset in range(0, N, output_mm_dims[2]):
                    for head_offset in range(0, N, head_dim * len(left_mtx_in[Q_STR][L1_POS_STR])):
                        # Since we need the full rows of attention score for softmax, the tiling is done 
                        # such that the full rows are calculated. No need to tile the columns in this case.
                        npu_dma_memcpy_nd(
                            metadata=q_l3l2_fifos,
                            bd_id=0,
                            mem=C,
                            offsets=[0, 0, 0, row_offset * N + head_offset],
                            sizes=[head_dim // attn_score_mm_dims[1], len(left_mtx_in[Q_STR][L1_POS_STR]), attn_score_mm_dims[0], attn_score_mm_dims[1]],
                            strides=[attn_score_mm_dims[1], head_dim, N, 1],
                            issue_token=True,
                        )

                        npu_dma_memcpy_nd(
                            metadata=k_l3l2_fifos,
                            bd_id=1,
                            mem=C,
                            offsets=[0, 0, 0, M * N + head_offset],
                            sizes=[head_dim // attn_score_mm_dims[1], len(right_mtx_in[K_STR][L1_POS_STR]), attn_score_mm_dims[2], attn_score_mm_dims[1]],
                            strides=[attn_score_mm_dims[1], head_dim, N, 1],
                            issue_token=True,
                        )

                        # Since the full row of the attention score is calculated, we don't need to tile the rows for V
                        npu_dma_memcpy_nd(
                            metadata=v_l3l2_fifos,
                            bd_id=2,
                            mem=C,
                            offsets=[0, 0, 0, 2 * M * N + head_offset],
                            sizes=[head_dim // attn_score_v_mm_dims[2], len(right_mtx_in[V_STR][L1_POS_STR]), attn_score_v_mm_dims[1], attn_score_v_mm_dims[2]],
                            strides=[attn_score_v_mm_dims[2], head_dim, N, 1],
                            issue_token=True,
                        )

                        # npu_dma_memcpy_nd(
                        #     metadata=Wo_l3l2_fifos,
                        #     bd_id=3,
                        #     mem=B,
                        #     offsets=[0, 0, 0, K * N + head_offset * N + col_offset],
                        #     sizes=[head_dim // output_mm_dims[1], len(right_mtx_in[WO_STR][L1_POS_STR]), output_mm_dims[1], output_mm_dims[2]],
                        #     strides=[output_mm_dims[1] * N, head_dim * N, N, 1],
                        #     issue_token=True,
                        # )
                        # dma_wait(q_l3l2_fifos, k_l3l2_fifos, v_l3l2_fifos, Wo_l3l2_fifos)
                        dma_wait(q_l3l2_fifos, k_l3l2_fifos, v_l3l2_fifos)

                    # npu_dma_memcpy_nd(
                    #     metadata=output_l2l3_fifos,
                    #     bd_id=4,
                    #     mem=C,
                    #     offsets=[0, 0, 0, 3 * M * N + row_offset * N + col_offset],
                    #     sizes=[1, 1, output_mm_dims[0], output_mm_dims[2]],
                    #     strides=[0, 0, N, 1],
                    # )
                    npu_dma_memcpy_nd(
                        metadata=output_l2l3_fifos,
                        bd_id=4,
                        mem=C,
                        offsets=[0, 0, 0, 3 * M * N + row_offset * N + col_offset],
                        sizes=[1, 1, 1, min_elems],
                        strides=[0, 0, 0, 1],
                    )
                    dma_wait(output_l2l3_fifos)

if __name__ == "__main__":
    main()
