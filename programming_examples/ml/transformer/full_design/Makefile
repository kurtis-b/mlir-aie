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
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include ${current_dir}../../../makefile-common
targetname=encoder

dtype_in_cpp=std::bfloat16_t
dtype_out_cpp=std::bfloat16_t
dtype_acc_cpp=float
seq_len_cpp=256
emb_dim_cpp=768
ffn_dim_cpp=3072

b_col_maj?=0
trace_size?=0

runargs=-x ../add_and_norm/build/final_add_and_norm.xclbin --instr_ffn1 ../ffn-1/build/insts_ffn-1.txt --kernel_ffn1 FFNONE --instr_ffn2 ../ffn-2/build/insts_ffn-2.txt --kernel_ffn2 FFNTWO --instr_addnorm ../add_and_norm/build/insts_add_and_norm.txt --kernel_addnorm ADD_AND_NORM --instr_mha ../mha/build/insts_mha.txt --kernel_mha MHA

# default sequential buffer allocation, if not already defined
buffer_aloc_flag?=basic-sequential

# When make is run, the builds are generated in their respective directories
# Thus, the print targets are able to be run even though they're defined in
# makefile-common and it isn't included here
.PHONY: all
all: 
	rm -rf build
	mkdir -p build
	prev_xclbin=""
	counter=0x901; prev_xclbin=""; \
	for dir in ffn-1 ffn-2 mha add_and_norm; do \
		$(MAKE) -C ../$$dir clean; \
		$(MAKE) -C ../$$dir; \
		mlir_target=$$(make -C ../$$dir print-mlir-target --no-print-directory | tail -n 1); \
		xclbin_target=$$(make -C ../$$dir print-xclbin-target --no-print-directory | tail -n 1); \
		insts_target=$$(make -C ../$$dir print-insts-target --no-print-directory | tail -n 1); \
		echo "$$dir mlir_target: $$mlir_target"; \
		echo "$$dir xclbin_target: $$xclbin_target"; \
		echo "$$dir insts_target: $$insts_target"; \
		cp ../$$dir/build/*.o build/; \
		kernel_name=$$(echo $$dir | tr -d '-'); \
		kernel_name=$$(echo $$kernel_name | tr '[:lower:]' '[:upper:]'); \
		kernel_name=$$(echo $$kernel_name | sed -e 's/1/ONE/g' -e 's/2/TWO/g' -e 's/3/THREE/g' -e 's/4/FOUR/g' -e 's/5/FIVE/g' -e 's/6/SIX/g' -e 's/7/SEVEN/g' -e 's/8/EIGHT/g' -e 's/9/NINE/g' -e 's/0/ZERO/g'); \
		aiecc_flags="--xclbin-kernel-name=$$kernel_name --xclbin-kernel-id=$$counter --xclbin-instance-name=$${kernel_name}INST --no-aiesim --alloc-scheme=${buffer_aloc_flag} --aie-generate-xclbin --no-compile-host --xclbin-name=../../$$dir/$$xclbin_target --no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} --aie-generate-npu-insts --npu-insts-name=../../$$dir/$$insts_target"; \
		if [ -n "$$prev_xclbin" ]; then aiecc_flags="$$aiecc_flags --xclbin-input=$$prev_xclbin"; fi; \
		echo "$$dir aiecc_flags: $$aiecc_flags"; \
		cd build && aiecc.py $$aiecc_flags ../../$$dir/$$mlir_target; \
		counter=$$(printf "0x%03x" $$(( 0x$${counter#0x} + 1 )) ); \
		prev_xclbin=../../$$dir/$$xclbin_target; \
		cd ..; \
	done

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake `${getwslpath} ${srcdir}` -DTARGET_NAME=${targetname} -DSEQ_LEN=${seq_len_cpp} -DEMB_DIM=${emb_dim_cpp} -DFFN_DIM=${ffn_dim_cpp}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

.PHONY: run
run: ${targetname}.exe
	export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 && \
	${powershell} ./$< ${runargs}

.PHONY: clean
clean: clean_trace
	rm -rf build _build ${targetname}.exe

.PHONY: clean_trace
clean_trace:
	rm -rf tmpTrace parse*.json trace_${target_suffix}.json trace.txt