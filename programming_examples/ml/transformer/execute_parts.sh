#!/bin/bash

output_file="avg_times.txt"
> "$output_file"

for dir in */ ; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        cd "$dir" || continue
        make clean >/dev/null 2>&1
        make run > run_output.txt 2>&1
        avg_line=$(grep -E "Avg.*time.*:.*us" run_output.txt)
        if [[ $avg_line =~ ([Aa]vg.*time)[^0-9]*:\ *([0-9.]+)us ]]; then
            label="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            echo "$dir: $label: $value us" >> "../$output_file"
        else
            echo "$dir: No avg time found" >> "../$output_file"
        fi
        cd ..
    fi
done

echo "Results saved to $output_file"