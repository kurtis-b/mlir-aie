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
subdir=perf_analysis/softmax
targetname=mha


# M, K, N, H: final GEMM dimensions
# NOTE: The maximum sequence length is 256. If the sequence length is larger than 256,
# the sequence should be split into multiple segments of size 256.
M?=256
K?=768
N?=768
H?=12
H_DIM := $(shell echo $$(($(N)/$(H))))

ifeq ($(devicename),npu2)
n_aie_cols?=4
else 
n_aie_cols?=4
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

include ${srcdir}/../../makefile-common

ifeq ($(devicename),npu2)
kernels_dir=${srcdir}/../../../../../aie_kernels/aie2p
else 
kernels_dir=${srcdir}/../../../../../aie_kernels/aie2
endif
aie2_runtime_dir = ${AIEOPT_DIR}/aie_runtime_lib/AIE2

SKIP_KERNELS=true
KERNEL_DEFINES=-D${dtype_in}_${dtype_out}_ONLY -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -I${kernels_dir} -I${srcdir}/../../../../../aie_kernels 

# Define matmul and softmax dimension lists
# If these values are changed, make sure to update the corresponding kernel code's
# combos(X) and "#if" macros. 
matmul_dims := 64x64x64 16x32x256
softmax_dims := 16x256

# Add "nm ${@D}/${@F}" at the end of the command to see the symbols in the object file 
# Pattern rule for matmul kernels, select row or col major based on b_col_maj

# Extract dimensions from the pattern stem (e.g., 32x64x64)
define extract_dims
$(eval dims := $(subst x, ,$*))
$(eval m := $(word 1,${dims}))
$(eval k := $(word 2,${dims}))
$(eval n := $(word 3,${dims}))
endef

define extract_dims_softmax
$(eval dims := $(subst x, ,$*))
$(eval m := $(word 1,${dims}))
$(eval n := $(word 2,${dims}))
endef

build/div_%.o: ${kernels_dir}/div.cc
	$(call extract_dims)
	mkdir -p $(@D)
	cd $(@D) && ${KERNEL_CC} ${KERNEL_CFLAGS} -DDIM_M=${m} -DDIM_N=${n} -DDIM_HEAD=${H_DIM} -c $< -o $(@F)

build/mm_%_row_major.o: ${srcdir}/mm_aie2.cc
	$(call extract_dims)
	mkdir -p $(@D)
	cd $(@D) && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o $(@F)

build/mha_mm_%_row_major.o: build/mm_%_row_major.o build/div_%.o
	mkdir -p $(@D)
	ar rvs $@ $^

build/mm_%_col_major.o: ${srcdir}/mm_aie2.cc
	$(call extract_dims)
	mkdir -p $(@D)
	cd $(@D) && ${KERNEL_CC} ${KERNEL_CFLAGS} -DB_COL_MAJ ${KERNEL_DEFINES} -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -c $< -o $(@F)

build/mha_mm_%_col_major.o: build/mm_%_col_major.o build/div_%.o
	mkdir -p $(@D)
	ar rvs $@ $^

# Bundle rule for matmul kernels
matmul_objs_row_m := $(addprefix build/mha_mm_,$(addsuffix _row_major.o,$(matmul_dims)))
matmul_objs_col_m := $(addprefix build/mha_mm_,$(addsuffix _col_major.o,$(matmul_dims)))

ifeq ($(devicename),npu)
build/mha_lut_based_ops.o: ${aie2_runtime_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} -I. -c $< -o ${@F}
endif

# Pattern rule for softmax kernels
build/mha_softmax_%.o: ${kernels_dir}/softmax.cc
	$(call extract_dims_softmax)
	@mkdir -p $(@D)
ifeq ($(devicename),npu)
	cd $(@D) && ${KERNEL_CC} ${KERNEL_CFLAGS} -DDIM_M=${m} -DDIM_N=${n} -I. -I${aie2_runtime_dir} -c $< -o $(@F)
else ifeq ($(devicename),npu2)
	cd $(@D) && ${KERNEL_CC} ${KERNEL_CFLAGS} -c $< -o $(@F)
endif

# Bundle rule for softmax kernel and lut_based_ops.o if npu
ifeq ($(devicename),npu)
build/mha_softmax.o: $(addprefix build/mha_softmax_,$(addsuffix .o,$(softmax_dims))) build/mha_lut_based_ops.o
	ar rvs $@ $+
else ifeq ($(devicename),npu2)
build/mha_softmax.o: $(addprefix build/mha_softmax_,$(addsuffix .o,$(softmax_dims)))
	ar rvs $@ $+
endif

# Convenience targets to build all kernels
.PHONY: build_kernels
build_kernels: build/mha_softmax.o $(matmul_objs_row_m) $(matmul_objs_col_m)

.PHONY: run
run: ${targetname}.exe ${xclbin_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N -H $H ${runargs}

trace: ${targetname}.exe ${trace_xclbin_target} ${insts_target}
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< -x ${trace_xclbin_target} -i ${insts_target} -k MLIR_AIE -M $M -K $K -N $N -H $H ${runargs} -t ${trace_size}
	${srcdir}/../../../../utils/parse_trace.py --input trace.txt --mlir ${trace_mlir_target} --output trace_${target_suffix}.json
