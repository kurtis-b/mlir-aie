#!/usr/bin/bash

# run this script from one of the subdirectories to perform a sweep,
# e.g. from within whole_array, run ../sweep.sh.

runargs="--iters 20 --warmup 10"
export runargs=$runargs
iterations=1

Ms=$(awk 'BEGIN { for(i=512; i<=512; i*=2) print i }')
Ks=$(awk 'BEGIN { for(i=4096; i<=4096; i*=2) print i }')
Ns=$(awk 'BEGIN { for(i=4096; i<=4096; i*=2) print i }')

export n_aie_cols=4
dtypes_in=("i8" "bf16")
dtypes_out=("i8" "i16" "bf16" "f32")
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

# Print configuration used to run for reproducibility
log_out=my_sweep.log
env >>$log_out
cat Makefile >>$log_out

csv_out="sweep_one_workload.csv"
printf "M,K,N,m,k,n,dtype_in,dtype_out" >>$csv_out
for i in $(seq 1 $iterations); do
    printf ",It"$i >>$csv_out
done
printf ",avg_cpu_time\n" >>$csv_out

for M in $Ms; do
    for K in $Ks; do
        for N in $Ns; do
            for dtype_in in ${dtypes_in[@]}; do
                for dtype_out in ${dtypes_out[@]}; do
                    if [[ "$dtype_in" == "i8" && "$dtype_out" != "i8" && "$dtype_out" != "i16" ]]; then
                        continue
                    fi
                    if [[ "$dtype_in" == "i16" && "$dtype_out" != "i16" && "$dtype_out" != "i32" ]]; then
                        continue
                    fi
                    if [[ "$dtype_in" == "bf16" && "$dtype_out" != "bf16" && "$dtype_out" != "f32" ]]; then
                        continue
                    fi
                    csv_out="sweep_one_workload.csv"
                    export M=$M
                    export K=$K
                    export N=$N
                    export dtype_in=$dtype_in
                    export dtype_out=$dtype_out
                    export m=$m
                    export k=$k
                    export n=$n
                    echo ${M}x${K}x${N} ${trace_size} ${m}x${k}x${n} ${dtype_in} ${dtype_out} 1>&2
                    make clean 1>>$log_out 2>&1
                    make all 1>>$log_out 2>&1
                    for i in $(seq 1 $iterations); do
                        make run >.tmp_run.log
                        cat .tmp_run.log $run_output >>$log_out
                        if grep -q "error: \"-\":13:17: 'aie.tile' op allocated buffers exceeded available memory" .tmp_run.log; then
                            echo "Memory allocation error detected, skipping the iterations for this configuration and not writing to the CSV."
                            break
                        fi
                        if [ $i -eq 1 ]; then
                            printf "${M},${K},${N},${m},${k},${n},${dtype_in},${dtype_out}" >>$csv_out
                        fi
                        t=$(cat .tmp_run.log | sed -rn 's/^Avg NPU matmul time: ([0-9.]+)us.$/\1/p')
                        printf ",${t}" >>$csv_out
                        c=$(cat .tmp_run.log | sed -rn 's/^Avg CPU matmul time: ([0-9.]+)us.$/\1/p')
                        printf ",${c}" >>$csv_out
                    done
                    printf "\n" >>$csv_out
                done
            done
        done
    done
done
