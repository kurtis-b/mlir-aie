#!/usr/bin/bash

if [ "${NPU2}" = "1" ]; then
    devicename="npu2"
else
    devicename="npu"
fi
BASEDIR="$(pwd)"

results_dir="${devicename}_results"
if [ -d "$BASEDIR/$results_dir" ]; then
    rm -rf "$BASEDIR/$results_dir"
fi
mkdir -p "$BASEDIR/$results_dir"

if [ -f "$BASEDIR/$results_dir/run_analysis.txt" ]; then
    rm "$BASEDIR/$results_dir/run_analysis.txt"
fi
touch "$BASEDIR/$results_dir/run_analysis.txt"

echo "Running analysis for device: $devicename"
echo "Starting with design verification..."
"$BASEDIR/scripts/run_verify_designs.sh" "$BASEDIR/$results_dir"
if grep -q "Failed" "$BASEDIR/$results_dir/run_verify_results.txt"; then
    echo "Some designs failed verification."
else
    echo "All designs passed verification."
    echo "Running designs without verification to gather execution times..."
    "$BASEDIR/scripts/run_no_verify_designs.sh" "$BASEDIR/$results_dir"
    "$BASEDIR/scripts/run_fine_grained_profiling.sh" "$BASEDIR/$results_dir"
    # Skipping add_and_norm for now because it fails with the Strix
    # for dir in "$BASEDIR/mha" "$BASEDIR/mha_by_steps/only_attn_steps" "$BASEDIR/mha_by_steps/only_proj_steps" "$BASEDIR/add_and_norm" "$BASEDIR/ffn-1" "$BASEDIR/ffn-2"; do
    for dir in "$BASEDIR/mha" "$BASEDIR/mha_by_steps/only_attn_steps" "$BASEDIR/mha_by_steps/only_proj_steps" "$BASEDIR/ffn-1" "$BASEDIR/ffn-2"; do
        if [ -d "$dir" ]; then
            run_output_dir="$BASEDIR/$results_dir/$(basename "$dir")"
            mkdir -p "$run_output_dir"
            echo "Analysing $dir"
            echo "Saving results to $run_output_dir"
            cd "$dir" || continue
            dirname=$(basename "$dir")
            if [ "$dirname" == "only_attn_steps" ] || [ "$dirname" == "only_proj_steps" ]; then
                dirname="mha"
            fi
            python "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "build/aie_${dirname}.mlir.prj/input_physical.mlir" --task mem-util --output_dir "$run_output_dir" >> "$BASEDIR/$results_dir/run_analysis.txt" 2>&1
            python "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "build/aie_${dirname}.mlir" --task comp-dist --output_dir "$run_output_dir" >> "$BASEDIR/$results_dir/run_analysis.txt" 2>&1
            python "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "build/aie_${dirname}.mlir" --task loop-iters --output_dir "$run_output_dir" >> "$BASEDIR/$results_dir/run_analysis.txt" 2>&1
            cd "$BASEDIR"
        fi
    done
    echo "Generating plot for execution times..."
    python "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "$BASEDIR/$results_dir/run_no_verify_exec_times.csv" --task exec-times --output_dir "$BASEDIR/$results_dir" >> "$BASEDIR/$results_dir/run_analysis.txt" 2>&1
    python "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "$BASEDIR/$results_dir/fine_grained_profiling_times.csv" --task fine-grained --output_dir "$BASEDIR/$results_dir" >> "$BASEDIR/$results_dir/run_analysis.txt" 2>&1

    echo "Generating trace analysis..."
    "$BASEDIR/scripts/trace_tiles_full_design.sh" "$BASEDIR/$results_dir/mha" "$BASEDIR/mha_with_trace"
    echo "Analysis completed."
fi