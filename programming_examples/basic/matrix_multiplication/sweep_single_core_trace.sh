#!/usr/bin/bash

# run this script from one of the subdirectories to perform a sweep,
# e.g. from within whole_array, run ../sweep.sh.

log_out=my_sweep_trace.log
runargs="--iters 1 --warmup 1"
export runargs=$runargs
iterations=1

Ms=$(awk 'BEGIN { for(i=16; i<=1024; i*=2) print i }')
Ks=$(awk 'BEGIN { for(i=128; i<=1024; i*=2) print i }')
Ns=$(awk 'BEGIN { for(i=128; i<=1024; i*=2) print i }')

ms=(16 32 64 128)
ks=(16 32 64 128)
ns=(16 32 64 128)

dtypes_in=("i8" "bf16")
dtypes_out=("i8" "i16" "bf16" "f32")
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

# Print configuration used to run for reproducibility
env >>$log_out
cat Makefile >>$log_out

for M in $Ms; do
    for K in $Ks; do
        for N in $Ns; do
            for m in ${ms[@]}; do
                for k in ${ks[@]}; do
                    for n in ${ns[@]}; do
                        for dtype_in in ${dtypes_in[@]}; do
                            for dtype_out in ${dtypes_out[@]}; do
                                # trace_size is the size of the trace buffer in bytes. It must be large enough 
                                # to hold all trace data to avoid functional errors in the C matrix output. 
                                # If the buffer is too small, trace data may overwrite the C matrix data, 
                                # possibly due to the buffer address restarting. To verify, run 'make trace' 
                                # and check if the trace.txt file ends with 0x0. If not, increase the buffer 
                                # size until it does, ensuring functional correctness.
                                case $dtype_in in
                                    "i8")
                                        modifier_input=1
                                        ;;
                                    "i16" | "bf16")
                                        modifier_input=2
                                        ;;
                                    *)
                                        modifier_input=4
                                        ;;
                                esac
                                case $dtype_out in
                                    "i8")
                                        modifier_output=1
                                        ;;
                                    "i16" | "bf16")
                                        modifier_output=2
                                        ;;
                                    *)
                                        modifier_output=4
                                        ;;
                                esac
                                trace_size_MN=$((M * N * modifier_output))
                                trace_size_MK=$((M * K * modifier_input))
                                trace_size_KN=$((K * N * modifier_input))
                                trace_size=$(($trace_size_MN > $trace_size_MK ? $trace_size_MN : $trace_size_MK))
                                trace_size=$(($trace_size > $trace_size_KN ? $trace_size : $trace_size_KN))
                                trace_size=$((trace_size * M / 32))
                                if [ $trace_size -lt 4194304 ]; then
                                    trace_size=4194304
                                fi
                                export trace_size=$trace_size
                                if [ $M -lt $m ]; then
                                    m=$M
                                fi
                                export M=$M
                                export K=$K
                                export N=$N
                                export m=$m
                                export k=$k
                                export n=$n
                                if [[ "$dtype_in" == "i8" && "$dtype_out" != "i8" && "$dtype_out" != "i16" ]]; then
                                    continue
                                fi
                                if [[ "$dtype_in" == "i16" && "$dtype_out" != "i16" && "$dtype_out" != "i32" ]]; then
                                    continue
                                fi
                                if [[ "$dtype_in" == "bf16" && "$dtype_out" != "bf16" && "$dtype_out" != "f32" ]]; then
                                    continue
                                fi
                                export dtype_in=$dtype_in
                                export dtype_out=$dtype_out
                                echo ${M}x${K}x${N} ${trace_size} ${m}x${k}x${n} ${dtype_in} ${dtype_out} 1>&2
                                make clean 1>>$log_out 2>&1
                                make all 1>>$log_out 2>&1
                                for i in $(seq 1 $iterations); do
                                    make trace >.tmp_run.log
                                    cat .tmp_run.log $run_output >>$log_out
                                    if [ ! -s trace.txt ]; then
                                        echo "trace.txt is empty, skipping iteration" 1>&2
                                        break
                                    fi
                                    if grep -q "Failed." .tmp_run.log; then
                                        echo "Functional verification failed, skipping trace processing" 1>&2
                                        break
                                    fi
                                    make parse_trace 1>>$log_out 2>&1
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done