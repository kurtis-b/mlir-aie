#!/usr/bin/bash

output_dir="."
if [ $# -ge 1 ]; then
    output_dir="$1"
    mkdir -p "$output_dir"
fi

output_file="$output_dir/run_verify_results.txt"
if [ -f "$output_file" ]; then
    rm "$output_file"
fi
touch "$output_file"

for dir in mha mha_by_steps/only_attn_steps mha_by_steps/only_proj_steps ffn-1 ffn-2 add_and_norm; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        orig_dir=$(pwd)
        cd "$dir" || continue
        make clean >/dev/null 2>&1
        make run_verif > run_output.txt 2>&1
        status_line=$(grep -E "Failed|PASS!" run_output.txt)
        if [[ -n $status_line ]]; then
            echo "$dir: $status_line" >> "$output_file"
        else
            echo "$dir: No PASS!/Failed status found" >> "$output_file"
        fi
        cd "$orig_dir" || exit
    fi
done

echo "Results saved to $output_file"