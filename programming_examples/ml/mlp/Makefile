##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

# This file is to be included from one of the subdirectories, e.g. 
# /matrix_vector/, after defining at least the targetname and kernels variable.
#
# The build steps for these matrix and matrix-vector multiplication designs all
# look the same. We need to build:
#  - A common host test code, 
#  - a kernel implemented in C using AIE intrinsics,
#  - the actual design implemented in MLIR.
# There are also targets for signing the resulting xclbin, extracting traces 
# and cleaning everything.
#
# Since the targets are all the same for all designs, they are defined here.
# Subdirectories need only include this makefile-common after defining the 
# following variables:
# - subdir	    -- subdirectory you are including this from
# - targetname  -- resulting test host code will be named targetname.exe
# - kernels	    -- which kernels in kernels folder to compile and link in
#					  (without file extension)
# - M, K, N	    -- final dimensions of matrices, set in each makefile separately 
#				      may be used by design; N=1 for matrix-vector

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include ${current_dir}../../makefile-common
targetname=mlp_design

# M, K, N: final GEMM dimensions
M?=512
K?=4096
N?=4096

# m, k, n: single AIE core GEMM dimensions
m?=64
k?=128
n?=64

n_aie_cols?=4
b_col_maj?=0
n_iterations?=1000

kernels=mm_${m}x${k}x${n}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c
use_alt?=0
use_iron?=0

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

ifeq (${use_alt}, 1)
aie_py_src=${targetname}_alt.py
ifeq (${use_iron}, 1)
$(error Cannot specify both alt design and IRON)
endif
endif

ifeq (${use_iron}, 1)
aie_py_src=${targetname}_iron.py
endif

dtype_in?=i8
dtype_out?=i8

ifeq ($(dtype_in),bf16)
	dtype_in_cpp=std::bfloat16_t
endif
ifeq ($(dtype_out),bf16)
	dtype_out_cpp=std::bfloat16_t
	dtype_acc_cpp=float
endif
ifeq ($(dtype_in),i16)
	dtype_in_cpp=int16_t
endif
ifeq ($(dtype_out),i16)
	dtype_out_cpp=int16_t
	dtype_acc_cpp=int16_t
endif
ifeq ($(dtype_out),i32)
	dtype_out_cpp=int32_t
	dtype_acc_cpp=int32_t
endif
ifeq ($(dtype_out),f32)
	dtype_out_cpp=float
	dtype_acc_cpp=float
endif
ifeq ($(dtype_in),i8)
	dtype_in_cpp=int8_t
endif
ifeq ($(dtype_out),i8)
	dtype_out_cpp=int8_t
	dtype_acc_cpp=int8_t
endif

target_suffix?=${M}x${K}x${N}
mlir_target?=build/aie_${target_suffix}.mlir
xclbin_target?=build/final_${target_suffix}.xclbin
insts_target?=build/insts_${target_suffix}.txt
aie_py_src?=${targetname}.py

runargs?=-v 2 --warmup 0 --iters ${n_iterations} --b_col_maj ${b_col_maj}
aieargs+=-M $M -K $K -N $N --dtype_in ${dtype_in} --dtype_out ${dtype_out} -m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}

# TODO: Can't run with CHESS on my laptop because I don't have an ethernet connection with it
CHESS?=false

# default sequential buffer allocation, if not already defined
buffer_aloc_flag?=basic-sequential

kernels_dir=${srcdir}/../../../aie_kernels/aie2

.PHONY: all
all: ${xclbin_target} 

build/mm_b_row_maj_${m}x${k}x${n}.o: ${kernels_dir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

build/mm_b_col_maj_${m}x${k}x${n}.o: ${kernels_dir}/mm_b_col_maj.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

ifneq (${b_col_maj}, 1)
build/mm_${m}x${k}x${n}.o: build/mm_b_row_maj_${m}x${k}x${n}.o 
	mkdir -p ${@D}
	ar rvs $@ $^
else
build/mm_${m}x${k}x${n}.o: build/mm_b_col_maj_${m}x${k}x${n}.o
	mkdir -p ${@D}
	ar rvs $@ $^
endif

build/%.o: ${kernels_dir}/%.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size 0 > $@

${xclbin_target}: ${mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				$(if $(shell [ $(CHESS) != true ] && echo true), \
					--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}, \
				) \
				--aie-generate-npu --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

xclbin_sign=/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

run_py: ${xclbin_target} 
	${powershell} python3 ${srcdir}/test.py -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N ${runargs}

clean:
	rm -rf build *.elf* *.lst *.bif aie*${target_suffix}.mlir.prj log* *.xclbin sim \
		chess* *.o insts.txt \
		*.log aie_partition.json *.bin BOOT.BIN _x test.exe