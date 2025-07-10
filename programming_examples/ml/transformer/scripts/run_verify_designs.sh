#!/usr/bin/bash

output_file="run_verify_results.txt"
> "$output_file"

for dir in add_and_norm ffn-1 ffn-2 mha; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        cd "$dir" || continue
        make clean >/dev/null 2>&1
        make run_verif > run_output.txt 2>&1
        status_line=$(grep -E "Failed|PASS!" run_output.txt)
        if [[ -n $status_line ]]; then
            echo "$dir: $status_line" >> "../$output_file"
        else
            echo "$dir: No PASS!/Failed status found" >> "../$output_file"
        fi
        cd ..
    fi
done

echo "Results saved to $output_file"