##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

# ---

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include ${current_dir}../../../makefile-common

# The following environment variables that point to the Xilinx runtime (XRT)
# should be set up by an environment setup script already.
XILINX_XRT?=/opt/xilinx/xrt
XILINX_VITIS?=/tools/Xilinx/Vitis/2023.2
BOOST_ROOT?=/usr/include/boost
Boost_INCLUDE_DIRS=${BOOST_ROOT}

# ---

XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

TEST_UTILS_INCLUDE?=${current_dir}/../../../../runtime_lib/test_lib

CXX=g++-13

CXXFLAGS+=-std=gnu++23 -ggdb -I${XILINX_XRT_INCLUDE} ${Boost_INCLUDE_DIRS:%=-I%} -I${TEST_UTILS_INCLUDE}
LDFLAGS+=-L${XILINX_XRT_LIB} ${Boost_LIBRARY_DIRS:%=-L%}
LDLIBS+=-lxrt_coreutil -lboost_program_options -lboost_filesystem -luuid
CHESSCCWRAP2_FLAGS = aie2 -I${XILINX_VITIS}/aietools/include 

mlir_target?=build/aie_64x64.mlir
xclbin_target?=build/final.xclbin
insts_target?=build/insts.txt
kernel_target?=build/mv_${m}x${k}.o

M=64
K=64
m=64
k=64
dtype_in=bf16
dtype_out=f32
trace_size = 0 # Set to 0 to disable tracing

aieargs+=-m $m -k $k --dtype_in ${dtype_in} --dtype_out ${dtype_out}

# default sequential buffer allocation, if not already defined
buffer_aloc_flag?=basic-sequential

targetname = gemv
CHESS?=false

.PHONY: all
all: ${xclbin_target} \
	build/insts_64x128.txt \
	build/insts_64x64.txt \

${kernel_target}: kernel.cc zero.cc
	mkdir -p ${@D}
ifeq ($(CHESS), true)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DDIM_M=${m} -DDIM_K=${k} -c ${<:%=../%}-o ${@F} 
else 
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DDIM_M=${m} -DDIM_K=${k} -c ${<:%=../%} -o ${@F}
endif

${xclbin_target}: build/aie_64x128.mlir ${kernel_target} 
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --alloc-scheme=${buffer_aloc_flag} --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				$(if $(shell [ $(CHESS) != true ] && echo true), \
					--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR}, \
				) \
				$(<:%=../%)

build/aie_opt.mlir: build/aie_64x128.mlir
	mkdir -p ${@D}
	cd ${@D} &&	aie-opt aie_64x64.mlir -aie-objectFifo-stateful-transform -aie-register-objectFifos > lowered.mlir

build/aie_64x128.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 64 -K 128 $(aieargs) --trace_size ${trace_size} > $@

build/aie_64x64.mlir: aie2.py
	mkdir -p ${@D}
	python3 $< -M 64 -K 64 $(aieargs) --trace_size ${trace_size} > $@

build/insts_%.txt: build/aie_%.mlir 
	cd ${@D} && aiecc.py -v --aie-only-generate-npu --npu-insts-name=${@:build/%=%} $(<:%=../%)

xclbin_sign=${XILINX_XRT}/amdxdna/setup_xclbin_firmware.sh 
.PHONY: sign
sign: ${xclbin_target}
	${xclbin_sign} -dev Phoenix -xclbin $<

test: test.cpp 
	${CXX} ${CXXFLAGS} ${TEST_UTILS_INCLUDE}/test_utils.cpp -o test_utils.o $^ -o $@  ${LDFLAGS} ${LDLIBS}

.PHONY: run
run: all test
	./test

.PHONY: trace
trace: test ${xclbin_target}
	./test
	${srcdir}/../../../utils/parse_trace.py --filename trace.txt --mlir build/aie_64x64.mlir --colshift 1 > trace_4b.json

.PHONY: clean
clean_trace:
	rm -rf tmpTrace trace.txt trace*json

.PHONY: clean
clean: clean_trace
	rm -rf build 
	rm test
