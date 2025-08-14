#!/usr/bin/bash

orig_dir=$(pwd)
if [ -f "expansion_results.csv" ]; then
    rm "${orig_dir}/expansion_results.csv"
fi
touch "${orig_dir}/expansion_results.csv"
output_file="${orig_dir}/expansion_results.csv"
echo "expansion,avg_us,min_us,max_us,M,K,N,m,k,n" > "$output_file"
rm *.json *.png

cd ".." || continue
for expansion in 2x2 2x4 3x3 4x2; do
    echo "Processing $expansion"
    export mm_src="mm_$expansion.cc"
    if [ "$expansion" = "3x3" ]; then
        export M="264"
        export K="768"
        export N="792"
        export m="24"
        export k="32"
        export n="264"
    else
        export M="256"
        export K="768"
        export N="768"
        export m="16"
        export k="32"
        export n="256"
    fi
    export runargs="--b_col_maj 0 --warmup 0 --iters 1 --verify 1"
    make clean > /dev/null 2>&1
    make trace > trace.log 2>&1
    status_line=$(grep -E "Failed|PASS!" trace.log)
    if [[ $status_line =~ PASS! ]]; then
        mv trace_mm.json "${orig_dir}/${expansion}_expansion.json"
        export runargs="--b_col_maj 0 --warmup 10 --iters 1000 --verify 0"
        make clean > /dev/null 2>&1
        make run > run.log 2>&1
        avg_line=$(grep -E "Avg.*time.*:.*us" run.log)
        min_line=$(grep -E "Min.*time.*:.*us" run.log)
        max_line=$(grep -E "Max.*time.*:.*us" run.log)
        if [[ $avg_line =~ :\ *([0-9.]+)us ]]; then
            avg_value="${BASH_REMATCH[1]}"
        else
            avg_value=""
        fi

        if [[ $min_line =~ :\ *([0-9.]+)us ]]; then
            min_value="${BASH_REMATCH[1]}"
        else
            min_value=""
        fi

        if [[ $max_line =~ :\ *([0-9.]+)us ]]; then
            max_value="${BASH_REMATCH[1]}"
        else
            max_value=""
        fi
        echo "$expansion,$avg_value,$min_value,$max_value,$M,$K,$N,$m,$k,$n" >> "$output_file"
    elif [[ $status_line =~ Failed ]]; then
        echo "Expansion $expansion failed verification"
    fi
done

cd "$orig_dir" || continue
python analyse_traces.py ./ --results_dir ./
echo "Results saved to $output_file"