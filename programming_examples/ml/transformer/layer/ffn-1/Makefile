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
include ${current_dir}../../../../makefile-common

# M, K, N: final GEMM dimensions
M?=256
K?=512
N?=2048

# m, k, n: single AIE core GEMM dimensions
m?=32
k?=64
n?=32

n_aie_cols?=1
b_col_maj?=0

kernels=mm_${m}x${k}x${n}
aieargs+=-M $M -K $K -N $N --dtype_in ${dtype_in} --dtype_out ${dtype_out} -m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs=--b_col_maj ${b_col_maj} -v 2 --warmup 1 --iters 1
target_suffix=ffn-1

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

dtype_in?=i16
dtype_out?=i16
trace_size?=4096


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

targetname=ffn
mlir_target?=build/aie_${target_suffix}.mlir
trace_mlir_target?=build/aie_trace_${target_suffix}.mlir
xclbin_target?=build/final_${target_suffix}.xclbin
trace_xclbin_target?=build/trace_${target_suffix}.xclbin
insts_target?=build/insts_${target_suffix}.txt
aie_py_src?=${targetname}.py

CHESS?=false

# default sequential buffer allocation, if not already defined
buffer_aloc_flag?=basic-sequential

.PHONY: all
all: ${xclbin_target} ${targetname}.exe

build/mm_b_row_maj_${m}x${k}x${n}.o: ${srcdir}/mm.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o ${@F}

build/mm_b_col_maj_${m}x${k}x${n}.o: ${srcdir}/mm_b_col_maj.cc
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

build/aie_opt.mlir: ${mlir_target}
	mkdir -p ${@D}
	cd ${@D} &&	aie-opt aie_${target_suffix}.mlir -aie-objectFifo-stateful-transform -aie-register-objectFifos 

build/%.o: %.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

${mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size 0 > $@

${trace_mlir_target}: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${aieargs} --trace_size ${trace_size} > $@

${xclbin_target}: ${mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				$(if $(shell [ $(CHESS) != true ] && echo true), \
					--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}, \
				) \
				--aie-generate-npu --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${trace_xclbin_target}: ${trace_mlir_target} ${kernels:%=build/%.o}
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				$(if $(shell [ $(CHESS) != true ] && echo true), \
					--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}, \
				) \
				--aie-generate-npu --npu-insts-name=${insts_target:build/%=%} $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp ${srcdir}/common.h
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -ggdb -DDTYPE_IN=${dtype_in_cpp} -DDTYPE_OUT=${dtype_out_cpp} -DDTYPE_ACC=${dtype_acc_cpp}" \
		cmake ${srcdir} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

xclbin_sign=/opt/xilinx/xrt/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N ${runargs}

trace: ${targetname}.exe ${trace_xclbin_target} ${insts_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N ${runargs} -t ${trace_size}

.PHONY: parse_trace
parse_trace:
	../../../../utils/parse_trace.py --filename trace.txt --mlir ${trace_mlir_target} --colshift 1 > traces/trace_mm_${target_suffix}.json

.PHONY: clean
clean: clean_trace
	rm -rf build _build ${targetname}.exe

.PHONY: clean_trace
clean_trace:
	rm -rf tmpTrace parse*.json traces/trace_mm_${target_suffix}.json trace.txt
