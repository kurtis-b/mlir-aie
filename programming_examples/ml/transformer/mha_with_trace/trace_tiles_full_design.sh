#!/usr/bin/bash

# Sweep through a 4-by-4 array of (row, col) tile values
if [ "${NPU2}" = "1" ]; then
    devicename="npu2"
else
    devicename="npu"
fi

trace_dir="$(pwd)/traces"

for row in {0..3}; do
    for col in {0..3}; do
        export trace_tile="($row,$col)"
        echo "Running with trace_tile=$trace_tile"
        make clean
        # Note: trace_no_verif needs to be run in order to build with trace_size > 0
        # Just running make will build with trace_size = 0
        make trace_no_verif >> "$(pwd)/trace_results.txt" 2>&1
        mkdir -p traces
        mv trace_mha.json "traces/trace_mha_${row}_${col}.json"
    done
done

output_file="run_verify_results.txt"
> "$output_file"

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

python analyse_traces.py $trace_dir --dev $devicename
tar -czf traces.tar.gz traces
