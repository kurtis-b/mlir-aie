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
subdir=add_and_norm
targetname=addandnorm


# M, N: final GEMM dimensions
M?=256
N?=768

# m, k, n: single AIE core GEMM dimensions
# NOTE: Make sure N=n. For now, the implementation requires the width of the 
# tiles to be the width of the workload
m?=4
n?=768

n_aie_cols?=1
b_col_maj?=0

kernels=addandnorm_${m}x${n}
aieargs=--dev ${devicename} -M $M -N $N --dtype_in ${dtype_in} --dtype_out ${dtype_out} -m $m -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs=--b_col_maj ${b_col_maj} -v 2 --warmup 10 --iters 100
target_suffix=addandnorm_${m}x${n}

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

dtype_in?=i16
dtype_out?=bf16
trace_size?=0

include ${srcdir}/../makefile-common

ifeq ($(devicename),npu2)
kernels_dir=${srcdir}/../../../../aie_kernels/aie2p
else 
kernels_dir=${srcdir}/../../../../aie_kernels/aie2
endif

# Quantized layernorm will be the same as non-quantized layer norm, so output will be bfloat or float
ifeq ($(dtype_out),bf16)
	dtype_out_cpp=std::bfloat16_t
	dtype_acc_cpp=float
endif
ifeq ($(dtype_out),i16)
	dtype_out_cpp=std::bfloat16_t
	dtype_acc_cpp=std::bfloat16_t
endif
ifeq ($(dtype_out),i32)
	dtype_out_cpp=float
	dtype_acc_cpp=float
endif
ifeq ($(dtype_out),f32)
	dtype_out_cpp=float
	dtype_acc_cpp=float
endif
ifeq ($(dtype_out),i8)
	dtype_out_cpp=std::bfloat16_t
	dtype_acc_cpp=std::bfloat16_t
endif


build/layernorm_${m}x${n}.o: ${kernels_dir}/layernorm.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_M=${m} -DDIM_N=${n} -I${srcdir}/../../../../aie_runtime_lib/AIE2 -c $< -o ${@F}
else ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DDIM_M=${m} -DDIM_N=${n} -I${srcdir}/../../../../aie_runtime_lib/AIE2 -c $< -o ${@F}
endif

build/add_${m}x${n}.o: ${kernels_dir}/add.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_M=${m} -DDIM_N=${n} -I${srcdir}/../../../../aie_runtime_lib/AIE2 -c $< -o ${@F}
else ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DDIM_M=${m} -DDIM_N=${n} -I${srcdir}/../../../../aie_runtime_lib/AIE2 -c $< -o ${@F}
endif

build/addandnorm_${m}x${n}.o: build/layernorm_${m}x${n}.o build/add_${m}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^

build/%.o: %.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I${srcdir}/../../../../aie_runtime_lib/AIE2 -c $< -o ${@F}
else ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -I${srcdir}/../../../../aie_runtime_lib/AIE2 -c $< -o ${@F}
endif

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -N $N ${runargs}

trace: ${targetname}.exe ${trace_xclbin_target} ${insts_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M-N $N ${runargs} -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir ${trace_mlir_target} --colshift ${col_shift} > trace_${target_suffix}.json
