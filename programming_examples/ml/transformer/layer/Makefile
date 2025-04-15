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
include ${current_dir}../../../makefile-common

# M, K, N: final GEMM dimensions
M1?=256
K1?=512
N1?=2048
M2?=256
K2?=2048
N2?=512
b_col_maj?=0

runargs=--b_col_maj ${b_col_maj} -v 2 --warmup 1 --iters 1

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

targetname=transformer
xclbin_target1?=build/final_ffn-1.xclbin
insts_target1?=build/insts_ffn-1.txt
xclbin_target2?=build/final_ffn-2.xclbin
insts_target2?=build/insts_ffn-2.txt

xclbin_target?=${xclbin_target2}

CHESS?=false

# default sequential buffer allocation, if not already defined
buffer_aloc_flag?=basic-sequential

.PHONY: all
all: ${xclbin_target} ${targetname}.exe

${xclbin_target}: 
	mkdir -p ${@D}
	cp ffn-1/build/*.o build/
	cp ffn-2/build/*.o build/
	cd ${@D} && aiecc.py --xclbin-kernel-name=FFNONE --xclbin-kernel-id=0x901 --xclbin-instance-name=FFNONEINST --no-aiesim --alloc-scheme=basic-sequential --aie-generate-cdo --no-compile-host --xclbin-name=final_ffn-1.xclbin --no-xchesscc --no-xbridge --peano /home/cj/mlir-aie/my_install/llvm-aie --aie-generate-npu --npu-insts-name=insts_ffn-1.txt ../ffn-1/build/aie_ffn-1.mlir
	cd ${@D} && aiecc.py --xclbin-kernel-name=FFNTWO --xclbin-kernel-id=0x902 --xclbin-instance-name=FFNTWOINST --no-aiesim  --alloc-scheme=basic-sequential --aie-generate-cdo --no-compile-host --xclbin-input=final_ffn-1.xclbin --xclbin-name=final_ffn-2.xclbin --no-xchesscc --no-xbridge --peano /home/cj/mlir-aie/my_install/llvm-aie --aie-generate-npu --npu-insts-name=insts_ffn-2.txt ../ffn-2/build/aie_ffn-2.mlir


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
	${powershell} ./$< -x ${xclbin_target} --instr1 ${insts_target1} --kernel1 FFNONE --M1 ${M1} --K1 ${K1} --N1 ${N1} --instr2 ${insts_target2} --kernel2 FFNTWO --M2 ${M2} --K2 ${K2} --N2 ${N2} ${runargs}

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe
