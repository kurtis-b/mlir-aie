import argparse
import os
import json
import logging
import re
import matplotlib.pyplot as plt

# First number is row, second number is column
WORKLOAD_AT_TILE = {
    "tile2,1": {"type": "GEMM", "size": 64*64*64*2},
    "tile2,2": {"type": "GEMM", "size": 16*32*256*2},
    "tile2,3": {"type": "GEMM", "size": 64*64*64*2},
    "tile2,4": {"type": "GEMM", "size": 16*32*256*2},
    "tile3,1": {"type": "GEMM", "size": 64*64*64*2},
    "tile3,2": {"type": "Softmax", "size": 16*256},
    "tile3,3": {"type": "GEMM", "size": 64*64*64*2},
    "tile3,4": {"type": "Softmax", "size": 16*256},
    "tile4,1": {"type": "GEMM", "size": 64*64*64*2},
    "tile4,2": {"type": "GEMM", "size": 16*256*16*2},
    "tile4,3": 0,
    "tile4,4": {"type": "GEMM", "size": 16*256*16*2},
    "tile5,1": {"type": "GEMM", "size": 64*64*64*2},
    "tile5,2": {"type": "GEMM", "size": 16*16*256*2},
    "tile5,3": {"type": "Add", "size": 16*256},
    "tile5,4": {"type": "GEMM", "size": 16*16*256*2},
}

CLOCK_FREQ = 1e9  # 1 GHz
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
                    for j in range(i + 1, len(data)):
                        next_item = data[j]
                        if (
                            next_item.get("name") == "INSTR_EVENT_1"
                            and next_item.get("ph") == "E"
                        ):
                            end_ts = next_item.get("ts")
                            logging.info(f"INSTR_EVENT_0 start ts: {start_ts}, INSTR_EVENT_1 end ts: {end_ts}, duration: {end_ts - start_ts}")
                            i = j  # Move i forward to after this event
                            break
                        if next_item.get("name") == "INSTR_EVENT_0" and next_item.get("ph") == "B":
                            logging.info(f"Replacing start_ts with next INSTR_EVENT_0 ts: {next_item.get('ts')}")
                            start_ts = next_item.get("ts")
                    kernel_cycles = end_ts - start_ts if start_ts is not None and end_ts is not None else 0
                    if kernel_cycles <= 0:
                        logging.warning(f"Invalid kernel cycles instance found for tile {tile_str} at {filepath}")
                    kernel_time_s = kernel_cycles / CLOCK_FREQ if kernel_cycles > 0 else 0
                    if kernel_time_s > 0:
                        kernel_time_s_avg += kernel_time_s
                        kernel_time_s_max = max(kernel_time_s_max, kernel_time_s) if num_kernel_traces > 0 else kernel_time_s
                        kernel_time_s_min = min(kernel_time_s_min, kernel_time_s) if num_kernel_traces > 0 else kernel_time_s
                        num_kernel_traces += 1
                i += 1
            
            kernel_time_s_avg /= num_kernel_traces if num_kernel_traces > 0 else 1
            logging.info(f"Average kernel time (s): {kernel_time_s_avg}")
            logging.info(f"Max kernel time (s): {kernel_time_s_max}")
            logging.info(f"Min kernel time (s): {kernel_time_s_min}")
            gflops_per_s_avg = (workload['size'] / kernel_time_s_avg / 1e9) if workload and kernel_time_s_avg > 0 else 0
            gflops_per_s_max = (workload['size'] / kernel_time_s_min / 1e9) if workload and kernel_time_s_min > 0 else 0
            gflops_per_s_min = (workload['size'] / kernel_time_s_max / 1e9) if workload and kernel_time_s_max > 0 else 0
            logging.info(f"Average GFLOPs/sec: {gflops_per_s_avg}")
            logging.info(f"Max GFLOPs/sec: {gflops_per_s_max}")
            logging.info(f"Min GFLOPs/sec: {gflops_per_s_min}")

            # Consistency check: max should not be less than avg, min should not be greater than avg
            if gflops_per_s_max < gflops_per_s_avg:
                raise ValueError(f"GFLOPs/sec max ({gflops_per_s_max}) is less than avg ({gflops_per_s_avg}) for tile {tile_str} in {filepath}")
            if gflops_per_s_min > gflops_per_s_avg:
                raise ValueError(f"GFLOPs/sec min ({gflops_per_s_min}) is greater than avg ({gflops_per_s_avg}) for tile {tile_str} in {filepath}")

            performance_data[tile_str] = {
                "type": workload['type'] if workload else "Unknown",
                "size": workload['size'] if workload else 0,
                "kernel_time_s": kernel_time_s,
                "kernel_time_s_avg": kernel_time_s_avg,
                "kernel_time_s_max": kernel_time_s_max,
                "kernel_time_s_min": kernel_time_s_min,
                "gflops_per_s_avg": gflops_per_s_avg,
                "gflops_per_s_max": gflops_per_s_max,
                "gflops_per_s_min": gflops_per_s_min,
                "num_kernel_traces": num_kernel_traces,
                "filepath": filepath,
            }
            return performance_data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyse JSON files in a directory.")
    parser.add_argument("directory", help="Directory containing JSON files")
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
            result = analyse_json_file(filepath)
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
                    col -= 1
                tile_str = f"(Row {row-2},Col {col})"
                tiles.append(tile_str)
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

    _, ax = plt.subplots(figsize=(20, 6)) 
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')

    # Sort the data by avg_ops (lowest to highest), keeping all lists aligned
    sorted_data = sorted(zip(avg_ops, min_ops, max_ops, tiles, types, sizes, bar_colors))
    avg_ops, min_ops, max_ops, tiles, types, sizes, bar_colors = map(list, zip(*sorted_data))
    x = range(len(tiles))
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "performance_per_tile.png")
    plt.savefig(output_path)
    logging.info(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()