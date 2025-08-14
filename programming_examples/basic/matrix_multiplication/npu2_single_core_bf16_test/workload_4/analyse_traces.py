import argparse
import os
import json
import logging
import re
import matplotlib.pyplot as plt

# First number is row, second number is column
THEORETICAL_PEAK_BF16_BF16 = {"npu2": 256, "npu": 128} # MACs per cycle
WORKLOAD_AT_EXPANSION = {
    "2x2": 16*16*256,
    "2x4": 16*16*256,
    "3x3": 24*16*264,
    "4x2": 16*16*256,
}

CLOCK_FREQ = 10**9  # 1 GHz
def analyse_json_file(filepath):
    """
    Analyse the JSON file for performance data.
    Args:
        filepath (str): Path to the JSON file.
    Returns:
        dict: A dictionary containing performance data.
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
        KeyError: If expected keys are missing in the JSON data.
        Exception: For any other unexpected errors.
    """
    with open(filepath, 'r') as f:
        try:
            performance_data = {}
            kernel_time_s_avg = 0
            kernel_time_s_max = 0
            kernel_time_s_min = float('inf')
            num_kernel_traces = 0
            data = json.load(f)
            expansion = None
            workload = None
            start_ts = None
            end_ts = None

            filename = os.path.basename(filepath)
            expansion_match = re.search(r'^(\dx\d)_expansion.json', filename)
            if expansion_match:
                expansion = expansion_match.group(1)
                logging.info(f"Extracted expansion: {expansion}")
            else:
                logging.warning(f"Could not extract expansion from filename {filename}")

            workload = WORKLOAD_AT_EXPANSION.get(expansion)
            if workload:
                logging.info(f"Workload size: {workload}")
            else:
                logging.warning(f"No workload found for expansion {expansion}")

            # Now, iterate through data to find all kernel traces
            i = 0
            while i < len(data):
                trace_data = data[i]
                if (
                    trace_data.get("name") == "INSTR_EVENT_0"
                    and trace_data.get("ph") == "B"
                ):
                    start_ts = trace_data.get("ts")
                    # Search for the next INSTR_EVENT_1 with "E"
                    end_ts = None
                    num_instr_vectors = 0
                    for j in range(i + 1, len(data)):
                        next_item = data[j]
                        if (
                            next_item.get("name") == "INSTR_EVENT_1"
                            and next_item.get("ph") == "E"
                        ):
                            end_ts = next_item.get("ts")
                            if num_kernel_traces < 10:
                                logging.info(f"INSTR_EVENT_0 start ts: {start_ts}, INSTR_EVENT_1 end ts: {end_ts}, duration: {end_ts - start_ts}")
                            i = j  # Move i forward to after this event
                            break
                        if next_item.get("name") == "INSTR_EVENT_0" and next_item.get("ph") == "B":
                            logging.info(f"Replacing start_ts with next INSTR_EVENT_0 ts: {next_item.get('ts')}")
                            start_ts = next_item.get("ts")
                        if next_item.get("name") == "INSTR_VECTOR":
                            num_instr_vectors += 1
                    # 1us is 1 cycle in perfetto
                    kernel_cycles = end_ts - start_ts if start_ts is not None and end_ts is not None else 0
                    if kernel_cycles <= 0:
                        logging.warning(f"Invalid kernel cycles instance found for tile {expansion} at {filepath}")
                    kernel_time_s = kernel_cycles / CLOCK_FREQ if kernel_cycles > 0 else 0
                    if num_instr_vectors != 0:
                        if kernel_time_s > 0:
                            kernel_time_s_avg += kernel_time_s
                            kernel_time_s_max = max(kernel_time_s_max, kernel_time_s) if num_kernel_traces > 0 else kernel_time_s
                            kernel_time_s_min = min(kernel_time_s_min, kernel_time_s) if num_kernel_traces > 0 else kernel_time_s
                            num_kernel_traces += 1
                    else:
                        logging.info(f"Skipping kernel trace with {num_instr_vectors} INSTR_VECTOR events for tile {expansion} at {filepath}")
                i += 1
            kernel_time_s_avg /= num_kernel_traces if num_kernel_traces > 0 else 1
            logging.info(f"Average kernel time (s): {kernel_time_s_avg}")
            logging.info(f"Max kernel time (s): {kernel_time_s_max}")
            logging.info(f"Min kernel time (s): {kernel_time_s_min}")
            # Calculate GFLOPs/sec: (workload size in FLOPs) / (kernel time in seconds) / (10^9 FLOPs per GFLOPs)
            gflops_per_s_avg = (workload * 2 / kernel_time_s_avg / 1e9) if workload and kernel_time_s_avg > 0 else 0
            gflops_per_s_max = (workload * 2 / kernel_time_s_min / 1e9) if workload and kernel_time_s_min > 0 else 0
            gflops_per_s_min = (workload * 2 / kernel_time_s_max / 1e9) if workload and kernel_time_s_max > 0 else 0
            logging.info(f"Average GFLOPs/sec: {gflops_per_s_avg}")
            logging.info(f"Max GFLOPs/sec: {gflops_per_s_max}")
            logging.info(f"Min GFLOPs/sec: {gflops_per_s_min}")
            # Compute utilization: (workload size in MACs) / (kernel time in seconds * CLOCK_FREQ in cycles per second) / (THEORETICAL_PEAK_BF16_BF16 in MACs per cycle)
            compute_utilization = ((workload) / (kernel_time_s_avg * CLOCK_FREQ)) / THEORETICAL_PEAK_BF16_BF16["npu2"] # Fraction of theoretical peak
            logging.info(f"Compute Utilization for {expansion}: {compute_utilization:.2%}")

            # Consistency check: max should not be less than avg, min should not be greater than avg
            if (gflops_per_s_max - gflops_per_s_avg) < -1e-2:
                raise ValueError(f"GFLOPs/sec max ({gflops_per_s_max}) is less than avg ({gflops_per_s_avg}) for tile {expansion} in {filepath}")
            if (gflops_per_s_min - gflops_per_s_avg) > 1e-2:
                raise ValueError(f"GFLOPs/sec min ({gflops_per_s_min}) is greater than avg ({gflops_per_s_avg}) for tile {expansion} in {filepath}")

            performance_data[expansion] = {
                "size": workload if workload else 0,
                "kernel_time_s_avg": kernel_time_s_avg,
                "kernel_time_s_max": kernel_time_s_max,
                "kernel_time_s_min": kernel_time_s_min,
                "gflops_per_s_avg": gflops_per_s_avg,
                "gflops_per_s_max": gflops_per_s_max,
                "gflops_per_s_min": gflops_per_s_min,
                "num_kernel_traces": num_kernel_traces,
                "compute_utilization": compute_utilization,
                "filepath": filepath,
            }
            return performance_data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding {filepath}: {e}")

# Sort by tile col then tile row (extract col and row from tile string)
def tile_sort_key(item):
    # tile string is always at index 0 in the zipped tuple
    expansion = item[0] 
    match = re.match(r"(\d+)x(\d+)", expansion)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return (col, row)
    else:
        return (float('inf'), float('inf'))

def main():
    parser = argparse.ArgumentParser(description="Analyse JSON files in a directory.")
    parser.add_argument("directory", help="Directory containing JSON files")
    parser.add_argument("--results_dir", default="results", help="Directory to save results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not os.path.isdir(args.directory):
        logging.error(f"Error: {args.directory} is not a directory.")
        return

    performance_results = []
    for expansion in ["2x2", "2x4", "3x3", "4x2"]:
        filename = f"{expansion}_expansion.json"
        if filename.endswith('.json'):
            filepath = os.path.join(args.directory, filename)
            if not os.path.isfile(filepath):
                logging.warning(f"File {filepath} does not exist, skipping.")
                continue
            result = analyse_json_file(filepath)
            if result:
                performance_results.append(result)

    # Do something with performance_results
    logging.info(f"Collected performance results: {performance_results}")

    # Flatten results and prepare data for plotting
    expansions = []
    avg_ops = []
    min_ops = []
    max_ops = []
    sizes = []
    for result in performance_results:
        for expansion, data in result.items():
            expansions.append(expansion)
            avg_ops.append(data["gflops_per_s_avg"])
            min_ops.append(data["gflops_per_s_min"])
            max_ops.append(data["gflops_per_s_max"])
            sizes.append(data["size"])

    # Sort the data by tile col then row, keeping all lists aligned
    sorted_data = sorted(zip(expansions, avg_ops, min_ops, max_ops, sizes), key=tile_sort_key)
    expansions, avg_ops, min_ops, max_ops, sizes = map(list, zip(*sorted_data))
    x = range(len(expansions))

    _, ax = plt.subplots(figsize=(20, 6)) 
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    ax.bar(x, avg_ops, label='Avg GFLOPs/sec')
    ax.errorbar(
        x, avg_ops,
        yerr=[
            [avg - minv if abs(avg - minv) > 1e-8 else 0 for avg, minv in zip(avg_ops, min_ops)],
            [maxv - avg if abs(maxv - avg) > 1e-8 else 0 for avg, maxv in zip(avg_ops, max_ops)]
        ],
        fmt='o', color='black', label='Min/Max GFLOPs/sec'
    )

    # Add workload size/type info to x-tick labels
    new_xticklabels = []
    for i, (expansion, size) in enumerate(zip(expansions, sizes)):
        text = f"({size / 1e3:.1f}×10$^3$ MACs)"
        new_xticklabels.append(f"{expansion}\n{text}")

    ax.set_ylabel("Average GFLOPs/sec (log scale)")
    ax.set_xlabel("Expansion\n(Workload Size)")
    ax.set_title(f"Performance Across Expansions")
    ax.set_xticks(range(len(expansions)))
    ax.set_xticklabels(new_xticklabels, ha='center', fontsize=10.5, rotation=45)
    plt.tight_layout()
    output_path = os.path.join(args.results_dir, "performance_per_expansion.png")
    plt.savefig(output_path)
    logging.info(f"Plot saved to {output_path}")
    plt.close()

    # Plot compute utilization across expansions, including workload size in x-axis
    util_expansions = []
    util_values = []
    util_xticklabels = []
    for result in performance_results:
        for expansion, data in result.items():
            util_expansions.append(expansion)
            util_values.append(data["compute_utilization"] * 100)
            # Add workload size/type info
            text = f"({data['size'] / 1e3:.1f}×10$^3$ MACs)"
            util_xticklabels.append(f"{expansion}\n{text}")

    # Sort by tile col then row for consistent plotting
    util_sorted = sorted(zip(util_expansions, util_values, util_xticklabels), key=tile_sort_key)
    util_expansions, util_values, util_xticklabels = map(list, zip(*util_sorted))
    x = range(len(util_expansions))

    fig, ax = plt.subplots(figsize=(20, 6))
    bars = ax.bar(x, util_values)

    ax.set_ylabel("Compute Utilization (/% peak throughput)")
    ax.set_xlabel("Core Location\n(Workload Size)")
    ax.set_title("Compute Utilization Across Expansions with GEMM")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(util_xticklabels, ha='center', fontsize=10.5)
    plt.tight_layout()
    util_output_path = os.path.join(args.results_dir, "compute_utilization_per_expansion.png")
    plt.savefig(util_output_path)
    logging.info(f"Compute utilization plot saved to {util_output_path}")
    plt.close()

    # Plot kernel time (avg, min, max) across tiles, including workload size/type in x-axis
    kernel_avg = []
    kernel_min = []
    kernel_max = []
    kernel_expansions = []
    kernel_xticklabels = []
    for result in performance_results:
        for expansion, data in result.items():
            kernel_expansions.append(expansion)
            kernel_avg.append(data["kernel_time_s_avg"] * 1e6)  # convert to us
            kernel_min.append(data["kernel_time_s_min"] * 1e6)
            kernel_max.append(data["kernel_time_s_max"] * 1e6)
            # Add workload size/type info
            text = f"({data['size'] / 1e3:.1f}×10$^3$ MACs)"
            kernel_xticklabels.append(f"{expansion}\n{text}")

    kernel_sorted = sorted(
        zip(kernel_expansions, kernel_avg, kernel_min, kernel_max, kernel_xticklabels),
        key=tile_sort_key
    )
    kernel_expansions, kernel_avg, kernel_min, kernel_max, kernel_xticklabels = map(list, zip(*kernel_sorted))
    x = range(len(kernel_expansions))

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x, kernel_avg, label="Avg Kernel Time (us)")
    ax.errorbar(
        x, kernel_avg,
        yerr=[
            [avg - minv if abs(avg - minv) > 1e-8 else 0 for avg, minv in zip(kernel_avg, kernel_min)],
            [maxv - avg if abs(maxv - avg) > 1e-8 else 0 for avg, maxv in zip(kernel_avg, kernel_max)]
        ],
        fmt='o', color='black', label='Min/Max Kernel Time (us)'
    )
    ax.set_ylabel("Kernel Time (us)")
    ax.set_xlabel("Core Location\n(Workload Size)")
    ax.set_title("Kernel Time (Avg, Min, Max) Across Expansions")
    ax.set_xticks(x)
    ax.set_xticklabels(kernel_xticklabels, ha='center', fontsize=10.5, rotation=45)
    plt.tight_layout()
    kernel_output_path = os.path.join(args.results_dir, "kernel_time_per_expansion.png")
    plt.savefig(kernel_output_path)
    logging.info(f"Kernel time plot saved to {kernel_output_path}")
    plt.close()


if __name__ == "__main__":
    main()