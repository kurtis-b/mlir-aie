#!/usr/bin/bash

if [ "${NPU2}" = "1" ]; then
    devicename="npu2"
else
    devicename="npu"
fi
BASEDIR="$(pwd)"

if [ -f "$BASEDIR/analysis_results.txt" ]; then
    rm "$BASEDIR/analysis_results.txt"
fi

echo "Running analysis for device: $devicename"
echo "Starting with design verification..."
bash "$BASEDIR/scripts/run_verify_designs.sh"
if grep -q "Failed" run_verify_results.txt; then
    echo "Some designs failed verification."
else
    echo "All designs passed verification."
    echo "Running designs without verification to gather execution times..."
    bash "$BASEDIR/scripts/run_no_verify_designs.sh"
    for dir in "$BASEDIR/mha" "$BASEDIR/mha_by_steps/only_attn_steps" "$BASEDIR/mha_by_steps/only_proj_steps" "$BASEDIR/add_and_norm" "$BASEDIR/ffn-1" "$BASEDIR/ffn-2"; do
        if [ -d "$dir" ]; then
            echo "Analysing $dir"
            cd "$dir" || continue
            make clean >> "$BASEDIR/analysis_results.txt" 2>&1
            mkdir -p results
            dirname=$(basename "$dir")
            python3 "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "build/aie_${dirname}.mlir.prj/input_physical.mlir" --task mem-util >> "$BASEDIR/analysis_results.txt" 2>&1
            python3 "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "build/aie_${dirname}.mlir" --task comp-dist >> "$BASEDIR/analysis_results.txt" 2>&1
            python3 "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "build/aie_${dirname}.mlir" --task loop-iters >> "$BASEDIR/analysis_results.txt" 2>&1
            cd "$BASEDIR"
        fi
    done
    echo "Generating plot for execution times..."
    python3 "$BASEDIR/scripts/run_analysis.py" --dev "$devicename" --input_file "$BASEDIR/run_no_verify_exec_times.csv" --task exec-times --output_dir "$BASEDIR" >> "$BASEDIR/analysis_results.txt" 2>&1
fi

echo "Analysis completed."