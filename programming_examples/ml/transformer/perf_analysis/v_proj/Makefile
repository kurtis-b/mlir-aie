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
subdir=perf_analysis/v_proj
targetname=mha


# M, K, N, H: final GEMM dimensions
# NOTE: The maximum sequence length is 256. If the sequence length is larger than 256,
# the sequence should be split into multiple segments of size 256.
M?=256
K?=768
N?=768
H?=12

ifeq ($(devicename),npu2)
n_aie_cols?=3
else 
n_aie_cols?=3
endif
b_col_maj?=0

aieargs=--dev ${devicename} -M $M -K $K -N $N -H $H --dtype_in ${dtype_in} --dtype_out ${dtype_out} --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj}
runargs=--b_col_maj ${b_col_maj} -v 2 --warmup 10 --iters 5
target_suffix=mha

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

SKIP_KERNELS=true

include ${srcdir}/../../makefile-common

ifeq ($(devicename),npu2)
kernels_dir=${srcdir}/../../../../../aie_kernels/aie2p
else 
kernels_dir=${srcdir}/../../../../../aie_kernels/aie2
endif



aie2_runtime_dir = ${AIEOPT_DIR}/aie_runtime_lib/AIE2

# Define matmul and softmax dimension lists
# If these values are changed, make sure to update the corresponding kernel code's
# combos(X) and "#if" macros. 
matmul_dims := 256x32x16
softmax_dims := 32x256

# Add "nm ${@D}/${@F}" at the end of the command to see the symbols in the object file 
# Pattern rule for matmul kernels, select row or col major based on b_col_maj
build/mha_mm_%.o:
	$(eval dims := $(subst x, ,$*))
	$(eval m := $(word 1,${dims}))
	$(eval k := $(word 2,${dims}))
	$(eval n := $(word 3,${dims}))
ifeq ($(b_col_maj),0)
  ifeq ($(devicename),npu)
	@mkdir -p $(@D)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c ${srcdir}/mm_aie2.cc -o $(@F)
  else ifeq ($(devicename),npu2)
	@mkdir -p $(@D)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c ${srcdir}/mm_aie2p.cc -o $(@F)
  endif
else
  ifeq ($(devicename),npu)
	@mkdir -p $(@D)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c ${srcdir}/mm_b_col_maj_aie2.cc -o $(@F)
  else ifeq ($(devicename),npu2)
	@mkdir -p $(@D)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -c ${srcdir}/mm_b_col_maj_aie2p.cc -o $(@F)
  endif
endif

ifeq ($(devicename),npu)
build/mha_lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -c $< -o ${@F}
endif

# Pattern rule for softmax kernels
build/mha_softmax_%.o: 
	$(eval dims := $(subst x, ,$*))
	$(eval m := $(word 1,${dims}))
	$(eval n := $(word 2,${dims}))
	@mkdir -p $(@D)
ifeq ($(devicename),npu)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -I. -I${aie2_runtime_dir} -c ${kernels_dir}/softmax.cc -o $(@F)

else ifeq ($(devicename),npu2)
	cd $(@D) && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c ${kernels_dir}/softmax.cc -o $(@F)
endif

# Bundle rule for softmax kernel and lut_based_ops.o (only for npu)
ifeq ($(devicename),npu)
build/mha_softmax.o: $(addprefix build/mha_softmax_,$(addsuffix .o,$(softmax_dims))) build/mha_lut_based_ops.o
	ar rvs $@ $+
else ifeq ($(devicename),npu2)
build/mha_softmax.o: $(addprefix build/mha_softmax_,$(addsuffix .o,$(softmax_dims)))
	ar rvs $@ $+
endif

# Convenience targets to build all kernels
matmul_objs := $(addprefix build/mha_mm_,$(addsuffix .o,$(matmul_dims))) 

.PHONY: build_kernels
build_kernels: $(matmul_objs) build/mha_softmax.o

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N -H $H ${runargs}

trace: ${targetname}.exe ${trace_xclbin_target} ${insts_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N -H $H ${runargs} -t ${trace_size}
	${srcdir}/../../../../utils/parse_trace.py --filename trace.txt --mlir ${trace_mlir_target} --colshift ${col_shift} > trace_${target_suffix}.json
