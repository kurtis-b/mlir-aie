#!/usr/bin/bash

output_file="run_no_verify_exec_times.csv"
echo "design,avg_us,min_us,max_us" > "$output_file"

for dir in mha add_and_norm ffn-1 ffn-2; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        cd "$dir" || continue
        make clean >/dev/null 2>&1
        make run_no_verif > run_output.txt 2>&1
        avg_line=$(grep -E "Avg.*time.*:.*us" run_output.txt)
        min_line=$(grep -E "Min.*time.*:.*us" run_output.txt)
        max_line=$(grep -E "Max.*time.*:.*us" run_output.txt)

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

        echo "$dir,$avg_value,$min_value,$max_value" >> "../$output_file"
        cd ..
    fi
done

echo "Results saved to $output_file"