#!/usr/bin/bash

output_dir="."
if [ $# -ge 1 ]; then
    output_dir="$1"
    mkdir -p "$output_dir"
fi

output_file="$output_dir/run_no_verify_exec_times.csv"
if [ -f "$output_file" ]; then
    rm "$output_file"
fi
touch "$output_file"
echo "design,avg_us,min_us,max_us,M,K,N" > "$output_file"

for dir in mha mha_by_steps/only_attn_steps mha_by_steps/only_proj_steps add_and_norm ffn-1 ffn-2; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        orig_dir=$(pwd)
        cd "$dir" || continue
        make clean >/dev/null 2>&1
        make run_no_verif > run_output.txt 2>&1
        avg_line=$(grep -E "Avg.*time.*:.*us" run_output.txt)
        min_line=$(grep -E "Min.*time.*:.*us" run_output.txt)
        max_line=$(grep -E "Max.*time.*:.*us" run_output.txt)
        matrix_line=$(grep -Eo "Matrix size [0-9]+x[0-9]+(x[0-9]+)?" run_output.txt)

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

        if [[ $matrix_line =~ ([0-9]+)x([0-9]+)x([0-9]+) ]]; then
            M="${BASH_REMATCH[1]}"
            K="${BASH_REMATCH[2]}"
            N="${BASH_REMATCH[3]}"
        elif [[ $matrix_line =~ ([0-9]+)x([0-9]+) ]]; then
            M="${BASH_REMATCH[1]}"
            K="${BASH_REMATCH[2]}"
            N="1"
        else
            M=""
            K=""
            N=""
        fi
        echo "$dir,$avg_value,$min_value,$max_value,$M,$K,$N" >> "$output_file"
        cd "$orig_dir" || exit
    fi
done

echo "Results saved to $output_file"