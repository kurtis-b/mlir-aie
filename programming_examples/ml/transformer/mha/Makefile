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
subdir=mha
targetname=mha


# M, K, N, H: final GEMM dimensions
# NOTE: The maximum sequence length is 256. If the sequence length is larger than 256,
# the sequence should be split into multiple segments of size 256.
M?=256
K?=768
N?=768
H?=12

# m, k, n: single AIE core GEMM dimensions
m?=32
k?=32
n?=32

ifeq ($(devicename),npu2)
n_aie_cols?=3
else 
n_aie_cols?=3
endif
b_col_maj?=0

operations=mm_32x24x256 mm_32x192x32 mm_32x32x256 mm_256x24x16 mm_32x256x16 mm_32x16x192 softmax_32x256
kernels=mha
aieargs=--dev ${devicename} -M $M -K $K -N $N -H $H --dtype_in ${dtype_in} --dtype_out ${dtype_out} -m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs=--b_col_maj ${b_col_maj} -v 2 --warmup 1 --iters 1
target_suffix=mha_${m}x${k}x${n}

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

dtype_in?=i8
dtype_out?=i8
trace_size?=0

include ${srcdir}/../makefile-common

ifeq ($(devicename),npu2)
kernels_dir=${srcdir}/../../../../aie_kernels/aie2p
else 
kernels_dir=${srcdir}/../../../../aie_kernels/aie2
endif



# Add "nm ${@D}/${@F}" at the end of the command to see the symbols in the object file 
aie2_runtime_dir = ${AIEOPT_DIR}/aie_runtime_lib/AIE2
ifeq (${devicename}, npu)
build/mha_mm_b_row_maj_${m}x${k}x${n}.o: ${srcdir}/mm_aie2.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c $< -o ${@F}
else ifeq (${devicename}, npu2)
build/mha_mm_b_row_maj_${m}x${k}x${n}.o: ${srcdir}/mm_aie2p.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c $< -o ${@F}
endif

ifeq (${devicename}, npu)
build/mha_mm_b_col_maj_${m}x${k}x${n}.o: ${srcdir}/mm_b_col_maj_aie2.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c $< -o ${@F}
else ifeq (${devicename}, npu2)
build/mha_mm_b_col_maj_${m}x${k}x${n}.o: ${srcdir}/mm_b_col_maj_aie2p.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c $< -o ${@F}
endif

ifeq ($(devicename),npu)
build/lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

build/softmax_${m}x${n}.o: ${kernels_dir}/softmax.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I${aie2_runtime_dir} -c $< -o ${@F}
else ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -I${aie2_runtime_dir} -c $< -o ${@F}
endif

ifeq (${devicename}, npu)
ifneq (${b_col_maj}, 1)
build/mha.o: build/mha_mm_b_row_maj_${m}x${k}x${n}.o build/softmax_${m}x${n}.o build/lut_based_ops.o
	mkdir -p ${@D}
	ar rvs $@ $^
else
build/mha.o: build/mha_mm_b_col_maj_${m}x${k}x${n}.o build/softmax_${m}x${n}.o build/lut_based_ops.o
	mkdir -p ${@D}
	ar rvs $@ $^
endif
else ifeq (${devicename}, npu2)
ifneq (${b_col_maj}, 1)
build/mha.o: build/mha_mm_b_row_maj_${m}x${k}x${n}.o build/softmax_${m}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^
else
build/mha.o: build/mha_mm_b_col_maj_${m}x${k}x${n}.o build/softmax_${m}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^
endif
endif

build/%.o: %.cc
	mkdir -p ${@D}
ifeq (${devicename}, npu)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq (${devicename}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
endif

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N -H $H ${runargs}

trace: ${targetname}.exe ${trace_xclbin_target} ${insts_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N -H $H ${runargs} -t ${trace_size}
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir ${trace_mlir_target} --colshift ${col_shift} > trace_${target_suffix}.json
