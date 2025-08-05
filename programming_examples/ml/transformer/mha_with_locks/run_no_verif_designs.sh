#!/usr/bin/bash

if [ "${NPU2}" = "1" ]; then
    devicename="npu2"
else
    devicename="npu"
fi

output_file="${devicename}_run_no_verify_exec_times.csv"
if [ -f "$output_file" ]; then
    rm "$output_file"
fi
touch "$output_file"
echo "step,avg_us,min_us,max_us,M,K,N" > "$output_file"

for step in {0..3}; do
    export step="$step"
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
    echo "$step,$avg_value,$min_value,$max_value,$M,$K,$N" >> "$output_file"
done

echo "Results saved to $output_file"