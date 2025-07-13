#!/usr/bin/bash

# Sweep through a 4-by-4 array of (row, col) tile values

for row in {0..3}; do
    for col in {0..3}; do
        export trace_tile="($row,$col)"
        echo "Running with trace_tile=$trace_tile"
        make clean
        make trace_no_verif
        mkdir -p traces
        mv trace_mha.json "traces/trace_mha_${row}_${col}.json"
    done
done
