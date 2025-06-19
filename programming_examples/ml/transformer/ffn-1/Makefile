##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=ffn-1
targetname=ffn-1


# M, K, N: final GEMM dimensions
M?=256
K?=768
N?=3072

# m, k, n: single AIE core GEMM dimensions
m?=32
k?=64
n?=64

ifeq ($(devicename),npu2)
n_aie_cols?=2
else 
n_aie_cols?=1
endif
b_col_maj?=0

kernels=ffn-1_mm_${m}x${k}x${n}
aieargs=--dev ${devicename} -M $M -K $K -N $N --dtype_in ${dtype_in} --dtype_out ${dtype_out} -m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs=--b_col_maj ${b_col_maj} -v 2 --warmup 10 --iters 50
target_suffix=ffn-1_${m}x${k}x${n}

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

dtype_in?=bf16
dtype_out?=bf16
trace_size?=0

KERNEL_DEFINES=-D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n}
ifeq (${b_col_maj}, 1)
	KERNEL_DEFINES+=-DB_COL_MAJ
endif

RELU_KERNEL_DEFINES=-DDIM_M=${m} -DDIM_N=${n}

include ${srcdir}/../makefile-common

build/mm_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}

build/relu_${m}x${n}.o: ${kernels_dir}/relu.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${RELU_KERNEL_DEFINES} -c $< -o ${@F}

build/ffn-1_mm_${m}x${k}x${n}.o: build/mm_${m}x${k}x${n}.o build/relu_${m}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N ${runargs}

trace: ${targetname}.exe ${trace_xclbin_target} ${insts_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N ${runargs} -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --input trace.txt --mlir ${trace_mlir_target} --output trace_${target_suffix}.json
