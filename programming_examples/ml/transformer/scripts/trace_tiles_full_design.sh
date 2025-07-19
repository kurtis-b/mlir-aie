#!/usr/bin/bash

# Sweep through a 4-by-4 array of (row, col) tile values
output_dir="."
design_dir="."
if [ $# -ge 2 ]; then
    output_dir="$1"
    design_dir="$2"
    mkdir -p "$output_dir"
fi

orig_dir="$(pwd)"
cd "$design_dir" || exit

if [ "${NPU2}" = "1" ]; then
    devicename="npu2"
else
    devicename="npu"
fi

traces_dir="traces"
if [ -d "$traces_dir" ]; then
    rm -rf "$traces_dir"
fi
mkdir -p "$traces_dir"

if [ -f "$(pwd)/trace_results.txt" ]; then
    rm "$(pwd)/trace_results.txt"
fi
touch "$(pwd)/trace_results.txt"

if [ -f "run_output.txt" ]; then
    rm "run_output.txt"
fi
touch "run_output.txt"

for row in {0..3}; do
    for col in {0..3}; do
        export trace_tile="($row,$col)"
        echo "Running with trace_tile=$trace_tile"
        make clean
        # Note: trace_no_verif needs to be run in order to build with trace_size > 0
        # Just running make will build with trace_size = 0
        make trace_no_verif >> "$(pwd)/trace_results.txt" 2>&1
        mv trace_mha.json "$traces_dir/trace_mha_${row}_${col}.json"
    done
done

output_file="run_verify_results.txt"
if [ -f "$output_file" ]; then
    rm "$output_file"
fi
touch "$output_file"

# Note that 2,2 doesn't do any computation, so it'll fail verification
# All other runs with tiles should pass verification
for row in {0..3}; do
    for col in {0..3}; do
        export trace_tile="($row,$col)"
        make clean
        make run_verif > run_output.txt 2>&1
        status_line=$(grep -E "Failed|PASS!" run_output.txt)
        if [[ -n $status_line ]]; then
            echo "$row, $col: $status_line" >> "$output_file"
        else
            echo "$row, $col: No PASS!/Failed status found" >> "$output_file"
        fi
    done
done

cd "$orig_dir"
python scripts/analyse_traces.py "$design_dir/$traces_dir" --dev "$devicename" --results_dir "$output_dir"
tar -czf "$output_dir/traces.tar.gz" -C "$design_dir/$traces_dir" .
