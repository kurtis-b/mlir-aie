import argparse
import os
import json
import logging
import re
import matplotlib.pyplot as plt

# First number is row, second number is column
THEORETICAL_PEAK_BF16_BF16 = {"npu2": 256, "npu": 128} # MACs per cycle
WORKLOAD_AT_TILE = {
    "tile2,1": {"type": "GEMM", "size": 64*64*64},
    "tile2,2": {"type": "GEMM", "size": 16*32*256},
    "tile2,3": {"type": "GEMM", "size": 64*64*64},
    "tile2,4": {"type": "GEMM", "size": 16*32*256},
    "tile3,1": {"type": "GEMM", "size": 64*64*64},
    "tile3,2": {"type": "Softmax", "size": 16*256},
    "tile3,3": {"type": "GEMM", "size": 64*64*64},
    "tile3,4": {"type": "Softmax", "size": 16*256},
    "tile4,1": {"type": "GEMM", "size": 64*64*64},
    "tile4,2": {"type": "GEMM", "size": 16*256*16},
    "tile4,3": 0,
    "tile4,4": {"type": "GEMM", "size": 16*256*16},
    "tile5,1": {"type": "GEMM", "size": 64*64*64},
    "tile5,2": {"type": "GEMM", "size": 16*16*256},
    "tile5,3": {"type": "Add", "size": 16*256},
    "tile5,4": {"type": "GEMM", "size": 16*16*256},
}

CLOCK_FREQ = 10**9  # 1 GHz
def analyse_json_file(filepath, dev):
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
            tile_str = None
            workload = None
            start_ts = None
            end_ts = None

            # First, find the tile_str from the process_name event
            for trace_data in data:
                if trace_data.get("name") == "process_name":
                    args_value = trace_data.get("args", {})
                    if "name" in args_value:
                        logging.info(f'args["name"]: {args_value["name"]}')
                        tile_str = args_value["name"].split(" for ")[-1]
                        if "core_trace" not in args_value["name"]:
                            logging.info(f"Skipping non-core trace: {args_value['name']}")
                            return {}
                    break  # Only need the first process_name

            workload = WORKLOAD_AT_TILE.get(tile_str)
            if workload:
                logging.info(f"Workload type: {workload['type']}, size: {workload['size']}")
            else:
                logging.warning(f"No workload found for tile {tile_str}")

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
                        logging.warning(f"Invalid kernel cycles instance found for tile {tile_str} at {filepath}")
                    kernel_time_s = kernel_cycles / CLOCK_FREQ if kernel_cycles > 0 else 0
                    if num_instr_vectors != 0:
                        if kernel_time_s > 0:
                            kernel_time_s_avg += kernel_time_s
                            kernel_time_s_max = max(kernel_time_s_max, kernel_time_s) if num_kernel_traces > 0 else kernel_time_s
                            kernel_time_s_min = min(kernel_time_s_min, kernel_time_s) if num_kernel_traces > 0 else kernel_time_s
                            num_kernel_traces += 1
                    else:
                        logging.info(f"Skipping kernel trace with {num_instr_vectors} INSTR_VECTOR events for tile {tile_str} at {filepath}")
                i += 1
            kernel_time_s_avg /= num_kernel_traces if num_kernel_traces > 0 else 1
            logging.info(f"Average kernel time (s): {kernel_time_s_avg}")
            logging.info(f"Max kernel time (s): {kernel_time_s_max}")
            logging.info(f"Min kernel time (s): {kernel_time_s_min}")
            # Calculate GFLOPs/sec: (workload size in FLOPs) / (kernel time in seconds) / (10^9 FLOPs per GFLOPs)
            gflops_per_s_avg = (workload['size'] * 2 / kernel_time_s_avg / 1e9) if workload and kernel_time_s_avg > 0 else 0
            gflops_per_s_max = (workload['size'] * 2 / kernel_time_s_min / 1e9) if workload and kernel_time_s_min > 0 else 0
            gflops_per_s_min = (workload['size'] * 2 / kernel_time_s_max / 1e9) if workload and kernel_time_s_max > 0 else 0
            logging.info(f"Average GFLOPs/sec: {gflops_per_s_avg}")
            logging.info(f"Max GFLOPs/sec: {gflops_per_s_max}")
            logging.info(f"Min GFLOPs/sec: {gflops_per_s_min}")
            if workload and workload['type'] == "GEMM":
                # Compute utilization: (workload size in MACs) / (kernel time in seconds * CLOCK_FREQ in cycles per second) / (THEORETICAL_PEAK_BF16_BF16 in MACs per cycle)
                compute_utilization = ((workload['size']) / (kernel_time_s_avg * CLOCK_FREQ)) / THEORETICAL_PEAK_BF16_BF16[dev] # Fraction of theoretical peak
                logging.info(f"Compute Utilization for {tile_str}: {compute_utilization:.2%}")
            else:
                compute_utilization = None
                logging.info(f"No compute utilization for {tile_str} as it is not a GEMM workload.")

            # Consistency check: max should not be less than avg, min should not be greater than avg
            if gflops_per_s_max < gflops_per_s_avg:
                raise ValueError(f"GFLOPs/sec max ({gflops_per_s_max}) is less than avg ({gflops_per_s_avg}) for tile {tile_str} in {filepath}")
            if gflops_per_s_min > gflops_per_s_avg:
                raise ValueError(f"GFLOPs/sec min ({gflops_per_s_min}) is greater than avg ({gflops_per_s_avg}) for tile {tile_str} in {filepath}")

            performance_data[tile_str] = {
                "type": workload['type'] if workload else "Unknown",
                "size": workload['size'] if workload else 0,
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
    tile_str = item[0] 
    match = re.match(r"\(Row (\-?\d+),Col (\d+)\)", tile_str)
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
    parser.add_argument("--dev", choices=["npu", "npu2"], required=True, help="Device type: 'npu' or 'npu2'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not os.path.isdir(args.directory):
        logging.error(f"Error: {args.directory} is not a directory.")
        return

    performance_results = []
    for filename in os.listdir(args.directory):
        if filename.endswith('.json'):
            filepath = os.path.join(args.directory, filename)
            result = analyse_json_file(filepath, args.dev)
            if result:
                performance_results.append(result)

    # Do something with performance_results
    logging.info(f"Collected performance results: {performance_results}")

    # Flatten results and prepare data for plotting
    tiles = []
    avg_ops = []
    min_ops = []
    max_ops = []
    types = []
    sizes = []
    for result in performance_results:
        for tile, data in result.items():
            # Parse tile string into tuple (e.g., "tile2,1" -> (2, 1))
            match = re.match(r"tile(\d+),(\d+)", tile)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                if args.dev == "npu":
                    tile_str = f"(Row {row-2},Col {col-1})"
                    tiles.append(tile_str)
                else:
                    # Need to check traces generated for Strix. Haven't done so yet.
                    raise ValueError(f"Need to add support for device type: {args.dev}")
            else:
                tiles.append(tile)
            avg_ops.append(data["gflops_per_s_avg"])
            min_ops.append(data["gflops_per_s_min"])
            max_ops.append(data["gflops_per_s_max"])
            types.append(data["type"])
            sizes.append(data["size"])

    # Assign a color to each workload type
    unique_types = list(set(types))
    color_map = plt.get_cmap('tab10')
    type_to_color = {t: color_map(i) for i, t in enumerate(unique_types)}
    bar_colors = [type_to_color[t] for t in types]

    # Sort the data by tile col then row, keeping all lists aligned
    sorted_data = sorted(zip(tiles, avg_ops, min_ops, max_ops, types, sizes, bar_colors), key=tile_sort_key)
    tiles, avg_ops, min_ops, max_ops, types, sizes, bar_colors = map(list, zip(*sorted_data))
    x = range(len(tiles))

    _, ax = plt.subplots(figsize=(20, 6)) 
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    ax.bar(x, avg_ops, color=bar_colors, label='Avg GFLOPs/sec')
    ax.errorbar(
        x, avg_ops,
        yerr=[
            [avg - minv for avg, minv in zip(avg_ops, min_ops)],
            [maxv - avg for avg, maxv in zip(avg_ops, max_ops)]
        ],
        fmt='o', color='black', label='Min/Max GFLOPs/sec'
    )

    # Add workload size/type info to x-tick labels
    new_xticklabels = []
    for i, (tile, type, size) in enumerate(zip(tiles, types, sizes)):
        if type == "GEMM":
            text = f"({size / 1e3:.1f}×10$^3$ MACs)"
        elif type == "Softmax":
            text = f"({size:.1f} Exps)"
        elif type == "Add":
            text = f"({size:.1f} Adds)"
        new_xticklabels.append(f"{tile}\n{text}")

    # Create legend for workload types
    handles = [plt.Rectangle((0,0),1,1, color=type_to_color[t]) for t in unique_types]
    ax.legend(handles, unique_types, title="Workload Type")

    ax.set_ylabel("Average GFLOPs/sec (log scale)")
    ax.set_xlabel("Core Location\n(Workload Size and Type)")
    expected_tiles = [f"(Row {row},Col {col})" for row in range(4) for col in range(4)]
    missing_tiles = [tile for tile in expected_tiles if tile not in tiles]
    missing_str = ", ".join(missing_tiles) if missing_tiles else "None"
    ax.set_title(f"Performance Across Tiles\nMissing traces: {missing_str}")
    ax.set_xticks(range(len(tiles)))
    ax.set_xticklabels(new_xticklabels, ha='center', fontsize=10.5)  
    plt.tight_layout()
    output_path = os.path.join(args.results_dir, "performance_per_tile.png")
    plt.savefig(output_path)
    logging.info(f"Plot saved to {output_path}")
    plt.close()

    # Plot compute utilization across tiles, including workload size/type in x-axis
    util_tiles = []
    util_values = []
    util_colors = []
    util_labels = []
    util_xticklabels = []
    for result in performance_results:
        for tile, data in result.items():
            if data["type"] == "GEMM":
                match = re.match(r"tile(\d+),(\d+)", tile)
                if match:
                    row, col = int(match.group(1)), int(match.group(2))
                    if args.dev == "npu":
                        tile_str = f"(Row {row-2},Col {col-1})"
                    else:
                        # Need to check traces generated for Strix. Haven't done so yet.
                        raise ValueError(f"Need to add support for device type: {args.dev}")
                else:
                    tile_str = tile
                util_tiles.append(tile_str)
                util_values.append(data["compute_utilization"])
                util_colors.append("tab:blue")
                util_labels.append("GEMM")
                # Add workload size/type info
                text = f"({data['size'] / 1e3:.1f}×10$^3$ MACs)"
                util_xticklabels.append(f"{tile_str}\n{text}")

    # Sort by tile col then row for consistent plotting
    util_sorted = sorted(zip(util_tiles, util_values, util_colors, util_labels, util_xticklabels), key=tile_sort_key)
    util_tiles, util_values, util_colors, util_labels, util_xticklabels = map(list, zip(*util_sorted))
    x = range(len(util_tiles))

    fig, ax = plt.subplots(figsize=(20, 6))
    bars = ax.bar(x, util_values, color=util_colors)
    for i, label in enumerate(util_labels):
        if label != "GEMM":
            ax.text(i, 0.01, label, ha='center', va='bottom', fontsize=9, rotation=90, color='gray')

    ax.set_ylabel("Compute Utilization (/% peak throughput)")
    ax.set_xlabel("Core Location\n(Workload Size and Type)")
    ax.set_title("Compute Utilization Across Tiles for Tiles Running GEMM")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(util_xticklabels, ha='center', fontsize=10.5)
    plt.tight_layout()
    util_output_path = os.path.join(args.results_dir, "compute_utilization_per_tile.png")
    plt.savefig(util_output_path)
    logging.info(f"Compute utilization plot saved to {util_output_path}")
    plt.close()

    # Plot kernel time (avg, min, max) across tiles, including workload size/type in x-axis
    kernel_avg = []
    kernel_min = []
    kernel_max = []
    kernel_tiles = []
    kernel_xticklabels = []
    kernel_types = []
    for result in performance_results:
        for tile, data in result.items():
            match = re.match(r"tile(\d+),(\d+)", tile)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                if args.dev == "npu":
                    tile_str = f"(Row {row-2},Col {col-1})"
                else:
                    # Need to check traces generated for Strix. Haven't done so yet.
                    raise ValueError(f"Need to add support for device type: {args.dev}")
            else:
                tile_str = tile
            kernel_tiles.append(tile_str)
            kernel_avg.append(data["kernel_time_s_avg"] * 1e6)  # convert to us
            kernel_min.append(data["kernel_time_s_min"] * 1e6)
            kernel_max.append(data["kernel_time_s_max"] * 1e6)
            kernel_types.append(data["type"])
            # Add workload size/type info
            if data["type"] == "GEMM":
                text = f"({data['size'] / 1e3:.1f}×10$^3$ MACs)"
            elif data["type"] == "Softmax":
                text = f"({data['size']:.1f} Exps)"
            elif data["type"] == "Add":
                text = f"({data['size']:.1f} Adds)"
            else:
                text = ""
            kernel_xticklabels.append(f"{tile_str}\n{text}")

    # Assign a color to each workload type (reuse type_to_color from above)
    kernel_bar_colors = [type_to_color.get(t, "gray") for t in kernel_types]

    kernel_sorted = sorted(
        zip(kernel_tiles, kernel_avg, kernel_min, kernel_max, kernel_xticklabels, kernel_bar_colors, kernel_types),
        key=tile_sort_key
    )
    kernel_tiles, kernel_avg, kernel_min, kernel_max, kernel_xticklabels, kernel_bar_colors, kernel_types = map(list, zip(*kernel_sorted))
    x = range(len(kernel_tiles))

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x, kernel_avg, color=kernel_bar_colors, label="Avg Kernel Time (us)")
    ax.errorbar(
        x, kernel_avg,
        yerr=[
            [avg - minv for avg, minv in zip(kernel_avg, kernel_min)],
            [maxv - avg for avg, maxv in zip(kernel_avg, kernel_max)]
        ],
        fmt='o', color='black', label='Min/Max Kernel Time (us)'
    )
    ax.set_ylabel("Kernel Time (us)")
    ax.set_xlabel("Core Location\n(Workload Size and Type)")
    ax.set_title("Kernel Time (Avg, Min, Max) Across Tiles")
    ax.set_xticks(x)
    ax.set_xticklabels(kernel_xticklabels, ha='center', fontsize=10.5)
    # Create legend for workload types
    kernel_unique_types = list(set(kernel_types))
    kernel_handles = [plt.Rectangle((0,0),1,1, color=type_to_color.get(t, "gray")) for t in kernel_unique_types]
    ax.legend(kernel_handles, kernel_unique_types, title="Workload Type")
    plt.tight_layout()
    kernel_output_path = os.path.join(args.results_dir, "kernel_time_per_tile.png")
    plt.savefig(kernel_output_path)
    logging.info(f"Kernel time plot saved to {kernel_output_path}")
    plt.close()


if __name__ == "__main__":
    main()