#!/usr/bin/bash

if [ "${NPU2}" = "1" ]; then
    devicename="npu2"
else
    devicename="npu"
fi

output_file="${devicename}_run_verify_results.txt"
if [ -f "$output_file" ]; then
    rm "$output_file"
fi
touch "$output_file"

for dir in q_proj k_proj v_proj attn_score softmax attn_score_v; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        orig_dir=$(pwd)
        cd "$dir" || continue
        make clean >/dev/null 2>&1
        make run_verif > run_output.txt 2>&1
        status_line=$(grep -E "Failed|PASS!" run_output.txt)
        if [[ -n $status_line ]]; then
            echo "$dir: $status_line" >> "$orig_dir/$output_file"
        else
            echo "$dir: No PASS!/Failed status found" >> "$orig_dir/$output_file"
        fi
        cd "$orig_dir" || exit
    fi
done

echo "Results saved to $output_file"