import os
import re
from collections import defaultdict
import json
from collections import OrderedDict
import argparse
import logging
import csv
import matplotlib.pyplot as plt
import numpy as np

def analyse_memory_utilization(dev, input_file, output_dir):
    """
    Analyse memory utilization for MHA in MLIR file.
    This function reads the MLIR file, extracts memory usage per tile,
    and generates plots for L2 and L1 memory utilization.
    It also logs the memory usage details.
    Args:
        dev (str): Device type, either 'npu' or 'npu2'.
        input_file (str): Path to the MLIR file.
        output_dir (str): Directory to save the output plots.
    Returns:
        None
    Raises:
        ValueError: If an unknown device type is provided.
        FileNotFoundError: If the MLIR file does not exist.
        KeyError: If an unknown data type is encountered in the MLIR file.
        RuntimeError: If no tiles are found with row index == 1 or row index > 1.
        Exception: For any other unexpected errors.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Check if the MLIR file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"MLIR file {input_file} does not exist.")

    # Configure logger
    logger = logging.getLogger("analyse_tiling")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Set the file names to write the plots to
    l2_mem_util_file = os.path.join(output_dir, "l2_mem_utilization.png")
    l1_mem_util_file = os.path.join(output_dir, "l1_mem_utilization.png")

    # Delete the output files if they already exist
    if os.path.exists(l2_mem_util_file):
        os.remove(l2_mem_util_file)
    if os.path.exists(l1_mem_util_file):
        os.remove(l1_mem_util_file)

    # Set memory sizes based on device type
    if dev == "npu2":
        l2_mem_size = 512 # KB
        l1_mem_size = 64 # KB
    elif dev == "npu":
        l2_mem_size = 256 # KB
        l1_mem_size = 64 # KB
    # stack_size = 13 # kB, 3328 int32 elements
    # Map bit width to bytes
    type_bytes = {
        'i8': 1,
        'i16': 2,
        'bf16': 2,
        'i32': 4,
        'f32': 4,
    }

    # Regex to match buffer lines
    buffer_re = re.compile(
        r'%\w+\s*=\s*aie\.buffer\((?:%mem_tile_|%tile_)?(\d+)_(\d+)\).*memref<(\d+)(?:\s*x\s*\d+)?\s*x\s*(i\d+|bf\d+|f\d+)>'
    )
    # Regex to match the aie.flow line
    flow_re = re.compile(r'aie\.flow\(%tile_\d+_\d+, DMA : \d+, %tile_\d+_\d+, DMA : \d+\)')

    tile_mem = defaultdict(int)
    tile_buffers = defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            if flow_re.search(line):
                break
            m = buffer_re.search(line)
            if m:
                x, y, num_elem, dtype = m.groups()
                num_elem = int(num_elem)
                bytes_per_elem = type_bytes.get(dtype)
                if bytes_per_elem is None:
                    raise KeyError(f"Unknown data type '{dtype}' in MLIR file.")
                tile = (int(x), int(y))
                tile_mem[tile] += num_elem * bytes_per_elem

                # Extract symbol name if present
                sym_match = re.search(r'sym_name\s*=\s*"([^"]+)"', line)
                sym_name = sym_match.group(1) if sym_match else None
                tile_buffers[tile].append({
                    'sym_name': sym_name,
                    'num_elem': num_elem,
                    'dtype': dtype,
                    'bytes': num_elem * bytes_per_elem
                })
    logger.info("Tile memory usage excluding stack (in bytes and kB):")
    for tile, total_bytes in sorted(tile_mem.items()):
        logger.info(f"Tile {tile}: {total_bytes} bytes ({total_bytes / 1024:.2f} kB)")
        for buf in tile_buffers[tile]:
            logger.info(f"  Buffer: {buf['sym_name']}, {buf['num_elem']} x {buf['dtype']} = {buf['bytes']} bytes")

    # Filter tiles with row index == 1. These are the memory tiles.
    filtered_tiles = [(tile, total_bytes) for tile, total_bytes in tile_mem.items() if tile[1] == 1]
    if filtered_tiles:
        x_vals = [tile[0] for tile, _ in filtered_tiles]
        y_vals = [total_bytes / (l2_mem_size * 1024) * 100 for _, total_bytes in filtered_tiles]

        plt.figure(figsize=(8, 4))
        plt.bar(x_vals, y_vals)
        plt.xlabel("NPU Array Column Index")
        plt.ylabel(f"L2 Memory Usage (% of {l2_mem_size}KB)")
        plt.title("L2 Memory Usage Across Tiles For MHA")
        plt.xticks(x_vals)
        plt.tight_layout()

        plt.savefig(l2_mem_util_file)
        logger.info(f"Saved L2 memory utilization plot to {l2_mem_util_file}")
        plt.close()
    else:
        logger.info("No tiles found with row index == 1.")
    # Filter tiles with row index > 1. These are the compute tiles.
    compute_tiles = sorted(
        # [(tile, total_bytes + stack_size) for tile, total_bytes in tile_mem.items() if tile[1] > 1],
        [(tile, total_bytes) for tile, total_bytes in tile_mem.items() if tile[1] > 1],
        key=lambda x: (x[0][0], x[0][1])
    )
    if compute_tiles:
        x_vals = [tile for tile, _ in compute_tiles]
        y_vals = [total_bytes / (l1_mem_size * 1024) * 100 for _, total_bytes in compute_tiles]

        plt.figure(figsize=(8, 4))
        str_x_vals = [str(x) for x in x_vals]
        plt.bar(str_x_vals, y_vals)
        plt.xlabel("NPU Array Tile Position (col, row)")
        plt.ylabel(f"L1 Memory Usage (% of {l1_mem_size}KB)")
        plt.title("L1 Memory Usage Across Tiles For MHA")
        plt.xticks(str_x_vals)
        plt.tight_layout()

        plt.savefig(l1_mem_util_file)
        logger.info(f"Saved L1 memory utilization plot to {l1_mem_util_file}")
        plt.close()
    else:
        logger.info("No compute tiles found with row index > 1.")

def analyse_computation_distribution(input_file, output_dir):
    """
    Analyze computation distribution for the given MLIR file.
    This function reads the MLIR file, extracts loop upper bounds and matmul information,
    and generates a plot showing the normalized MACs per output tile generated.
    It also logs the details of the computation.
    Args:
        input_file (str): Path to the MLIR file.
        output_dir (str): Directory to save the output plot.
    Returns:
        None
    Raises:
        FileNotFoundError: If the MLIR file does not exist.
        Exception: For any other unexpected errors.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Check if the MLIR file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"MLIR file {input_file} does not exist.")

    # Configure logger
    logger = logging.getLogger("analyse_tiling")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Set the file names to write the plots to
    compute_util_file = os.path.join(output_dir, "comp_distribution.png")

    # Delete the output files if they already exist
    if os.path.exists(compute_util_file):
        os.remove(compute_util_file)

    # Parse loop upper bounds per tile from aie_mha.mlir
    core_loop_bounds = {}
    core_block_re = re.compile(r'%core_(\d+)_(\d+)\s*=\s*aie\.core\(%tile_(\d+)_(\d+)\)\s*{')
    scf_for_re = re.compile(r'scf\.for\s+%arg\d+\s*=\s*%c\d+(?:_\d+)?\s+to\s+%c(\d+)(?:_\d+)?\s+step\s+%c\d+(?:_\d+)?\s*{')

    with open(input_file, 'r') as f:
        core_loop_bounds = {}
        inside_core = False
        tile = None
        for line in f:
            if not inside_core:
                m = core_block_re.match(line.strip())
                if m:
                    logger.debug(f"Found core block: {line.strip()}")
                    tile = f"({int(m.group(3))}, {int(m.group(4))})"
                    inside_core = True
                    scf_for_count = 0
            else:
                # Check for matmul call
                if "matmul" in line:
                    logger.debug(f"Found matmul call in tile {tile}: {line.strip()}")
                    # Example: func.call @matmul_bf16_bf16_64_64_64_1(...)
                    call_match = re.search(r'@matmul_([^(\s]+)\(', line)
                    if call_match:
                        logger.debug(f"Matmul call found in tile {tile}: {call_match.group(0)}")
                        matmul_str = call_match.group(1)  # e.g., 'bf16_bf16_64_64_64_1'
                        parts = matmul_str.split('_')
                        if len(parts) >= 5:
                            dtype = f"{parts[0]}_{parts[1]}"
                            m, k, n = map(int, parts[-4:-1])
                            macs = m * k * n
                            if tile not in core_loop_bounds:
                                core_loop_bounds[tile] = {}
                            logger.debug(f"Tile {tile} matmul: dtype={dtype}, m={m}, k={k}, n={n}, macs_one_iter={macs}")
                            core_loop_bounds[tile]['matmul'] = {
                                'dtype': dtype.split('_')[0],  # e.g., 'bf16'
                                'm': m,
                                'k': k,
                                'n': n,
                                'macs_one_iter': macs
                            }
                m = scf_for_re.search(line)
                if m:
                    logger.debug(f"Found scf.for loop in tile {tile}: {line.strip()}")
                    scf_for_count += 1
                    if scf_for_count == 1:
                        continue  # Skip the first scf.for
                    matmul_instances = int(m.group(1))
                    if tile not in core_loop_bounds:
                        core_loop_bounds[tile] = {}
                        core_loop_bounds[tile]['instances'] = matmul_instances
                    else:
                        core_loop_bounds[tile]['instances'] *= matmul_instances
                else:
                    core_loop_bounds.setdefault(tile, {}).setdefault('instances', 1)
                if line.strip() == "}":
                    inside_core = False
                    tile = None
    logger.info("\nTile loop upper bounds and matmul info:")
    for tile, info in sorted(core_loop_bounds.items()):
        if isinstance(info, dict) and 'matmul' in info:
            matmul = info['matmul']
            instances = info.get('instances', 1)
            logger.info(f"Tile {tile}: {instances} iterations of {matmul['m']}x{matmul['k']}x{matmul['n']}, matmul dtype={matmul['dtype']}, MACs={matmul['macs_one_iter'] * instances}")
        else:
            logger.info(f"Tile {tile}: {info.get('instances', 1)} iterations")

    # Find tiles with and without 'matmul'/'macs_one_iter'
    tiles_with_matmul = []
    tiles_without_matmul = []
    for tile, info in core_loop_bounds.items():
        if isinstance(info, dict) and 'matmul' in info and 'macs_one_iter' in info['matmul']:
            tiles_with_matmul.append(tile)
        else:
            tiles_without_matmul.append(tile)
    logger.info(f"Tiles with 'matmul' and 'macs_one_iter': {tiles_with_matmul}")
    logger.info(f"Tiles without 'matmul' and/or 'macs_one_iter': {tiles_without_matmul}")

    # Sort the lists
    tiles_with_matmul.sort(key=lambda t: (int(t.strip("()").split(",")[0]), int(t.strip("()").split(",")[1])))
    tiles_without_matmul.sort(key=lambda t: (int(t.strip("()").split(",")[0]), int(t.strip("()").split(",")[1])))

    # Find the largest 'macs_one_iter' value among tiles_with_matmul
    max_macs = 0
    for tile in tiles_with_matmul:
        macs = core_loop_bounds[tile]['matmul']['macs_one_iter']
        if macs > max_macs:
            max_macs = macs

    # Normalize 'macs_one_iter' for each tile_with_matmul
    normalized_macs = {}
    for tile in tiles_with_matmul:
        macs = core_loop_bounds[tile]['matmul']['macs_one_iter']
        normalized = macs / max_macs if max_macs > 0 else 0
        normalized_macs[tile] = normalized
        logger.info(f"Tile {tile}: normalized macs_one_iter = {normalized:.3f}")

    # Plot normalized MACs for tiles with matmul
    if normalized_macs:
        plt.figure(figsize=(10, 5))
        tile_labels = [str(tile) for tile in tiles_with_matmul]
        norm_values = [normalized_macs[tile] for tile in tiles_with_matmul]
        plt.bar(tile_labels, norm_values)
        plt.xlabel("Tile (col, row)")
        plt.ylabel(f"Normalized MACs")
        title = "Compute Distribution Across Tiles (Normalized MACs per Output Tile Generated)"
        if tiles_without_matmul:
            title += f"\nTiles not running mmul: {', '.join(str(t) for t in tiles_without_matmul)}"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(compute_util_file)
        logger.info(f"Saved computation utilization plot to {compute_util_file}")
        plt.close()
    else:
        logger.info("No tiles with matmul found for computation utilization plot.")

def analyse_loop_iterations(input_file, output_dir):
    """
    Analyse loop iterations in the MLIR file.
    This function reads the MLIR file and extracts the number of tiles passed between L2/L3 at runtime.
    It prints a summary of the metadata symbols and their shapes, including the number of tiles passed.
    It also counts unique shapes and their occurrences.
    It is useful for understanding the L2/L3 behavior of the MHA implementation.
    Args:
        input_file (str): Path to the MLIR file.
        output_dir (str): Directory to save the output files.
    Returns:
        None
    Raises:
        FileNotFoundError: If the MLIR file does not exist.
        Exception: For any other unexpected errors.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Check if the MLIR file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"MLIR file {input_file} does not exist.")
    
    # Configure logger
    logger = logging.getLogger("analyse_tiling")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Set the file names to write the plots to
    l3_tiling_file = os.path.join(output_dir, "l3_tiling.png")

    # Delete the output files if they already exist
    if os.path.exists(l3_tiling_file):
        os.remove(l3_tiling_file)

    # Map bit width to bytes
    type_bytes = {
        'i8': 1,
        'i16': 2,
        'bf16': 2,
        'i32': 4,
        'f32': 4,
    }

    with open(input_file, 'r') as f:
        inside_sequence = False
        count_begin = 0
        metadata_symbols = OrderedDict()
        metadata_shapes = {}
        metadata_shape_counts = {}
        metadata_l2_size = {}
        for line in f:
            if not inside_sequence:
                if re.search(r'aiex\.runtime_sequence\s+@sequence\(', line):
                    inside_sequence = True
                    count_begin = 1
                # Look for aie.objectfifo lines with %mem_tile
                meta_match = re.search(r'aie\.objectfifo\s+@([^\s(]+)\(', line)
                objfifo_match = re.search(
                    r'aie\.objectfifo\s+@[^(\s]+\([^)]*%mem_tile[^)]*\)[^<]*<memref<(\d+)(?:\s*x\s*(\d+))?\s*x\s*([a-zA-Z0-9]+)>',
                    line
                )
                if meta_match and objfifo_match:
                    if objfifo_match.group(2):
                        num_elem = int(objfifo_match.group(1)) * int(objfifo_match.group(2))
                    else:
                        num_elem = int(objfifo_match.group(1))
                    dtype = objfifo_match.group(3)
                    logger.info(f"Found objectfifo {meta_match.group(1)}: num_elem={num_elem}, dtype={dtype}")
                    metadata_l2_size[meta_match.group(1)] = num_elem
            else:
                if "{" in line:
                    count_begin += 1
                if line.strip() == "}":
                    count_begin -= 1
                    if count_begin == 0:
                        logger.info("End of runtime sequence found.")
                        break
                # Match lines with metadata
                meta_match = re.search(r'@([A-Za-z0-9_]+)', line)
                shape_match = re.search(r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]', line)
                dtype_match = re.search(r'memref<\d+\s*x\s*([a-zA-Z0-9]+)>', line)
                if meta_match and shape_match and dtype_match:
                    meta_str = meta_match.group(1)
                    shape_str = shape_match.group(2)
                    dtype = dtype_match.group(1)
                    shape_nums = tuple(int(x.strip()) for x in shape_str.split(','))
                    if meta_str not in metadata_symbols:
                        metadata_symbols[meta_str] = {"dtype": dtype}
                    else:
                        # Only set dtype if not already set
                        if "dtype" not in metadata_symbols[meta_str]:
                            metadata_symbols[meta_str]["dtype"] = dtype
                    if meta_str not in metadata_shapes:
                        metadata_shapes[meta_str] = []
                    metadata_shapes[meta_str].append(shape_nums)
                    # Count unique shapes
                    if meta_str not in metadata_shape_counts:
                        metadata_shape_counts[meta_str] = {}
                    metadata_shape_counts[meta_str][shape_nums] = metadata_shape_counts[meta_str].get(shape_nums, 0) + 1
                # Handle the 2-line metadata form
                dma_configure_match = re.search(r'%\d+\s*=\s*aiex\.dma_configure_task_for\s+@([^\s{]+)\s*{', line)
                if dma_configure_match:
                    meta_str = dma_configure_match.group(1)
                    logger.info(f"Found metadata symbol: {meta_str}")
                    # Read the next line for aie.dma_bd
                    next_line = next(f, "")
                    dma_bd_match = re.search(
                        r'aie\.dma_bd\([^\)]*\s*:\s*memref<[^>]+>,\s*\d+,\s*\d+,\s*\[([^\]]+)\]\)', next_line)
                    dtype_match = re.search(r'memref<\d+\s*x\s*([a-zA-Z0-9]+)>', next_line)
                    if dma_bd_match and dtype_match:
                        sizes_str = dma_bd_match.group(1)
                        logger.info(f"Found sizes: {sizes_str} for metadata symbol: {meta_str}")
                        # Extract all <size = N, stride = M> blocks
                        size_matches = re.findall(r'<size\s*=\s*(\d+),\s*stride\s*=\s*[\d]+>', sizes_str)
                        if size_matches and len(size_matches) == 4:
                            shape_nums = tuple(int(sz) for sz in size_matches)
                            dtype = dtype_match.group(1)
                            if meta_str not in metadata_symbols:
                                metadata_symbols[meta_str] = {"dtype": dtype}
                            else:
                                if "dtype" not in metadata_symbols[meta_str]:
                                    metadata_symbols[meta_str]["dtype"] = dtype
                            if meta_str not in metadata_shapes:
                                metadata_shapes[meta_str] = []
                            metadata_shapes[meta_str].append(shape_nums)
                            if meta_str not in metadata_shape_counts:
                                metadata_shape_counts[meta_str] = {}
                            metadata_shape_counts[meta_str][shape_nums] = metadata_shape_counts[meta_str].get(shape_nums, 0) + 1

    # Prepare and print metadata summary in a more readable format
    logger.info("\nMetadata counts and shapes:")
    metadata_json = OrderedDict()
    for meta in metadata_symbols.keys():
        entry = {
            "unique_shapes": [],
            "num_tiles": 0
        }
        if meta in metadata_shapes:
            unique_shapes = metadata_shape_counts[meta]

            for shape, shape_count in sorted(unique_shapes.items()):
                entry["unique_shapes"].append({
                    "shape": shape,
                    "shape_count": shape_count
                })
                entry["num_tiles"] += shape[0] * shape[1] * shape[2] * shape[3] * shape_count // metadata_l2_size.get(meta, 1)  # Adjust num_tiles based on L2 size
        else:
            entry["unique_shapes"] = []
            entry["num_tiles"] = 0
        metadata_json[meta] = entry

    # Print in a more human-friendly way
    for meta, entry in metadata_json.items():
        logger.info(f"\nMetadata symbol: {meta}")
        logger.info(f"  Total tiles passed: {entry['num_tiles']}")
        if entry["unique_shapes"]:
            logger.info("  Unique shapes and counts:")
            for shape_info in entry["unique_shapes"]:
                logger.info(f"    Shape: {shape_info['shape']}, Count: {shape_info['shape_count']}")
        else:
            logger.info("  No shapes found.")

    # Prepare data for plotting
    symbol_names = []
    total_tiles = []
    l2_tile_sizes = []

    # Check for only one unique shape per metadata symbol
    for meta, entry in metadata_json.items():
        if len(entry["unique_shapes"]) != 1:
            raise RuntimeError(
                f"Error: The current implementation expects only 1 unique shape in the runtime sequence for each symbol for generating the plot. "
                f"Found {len(entry['unique_shapes'])} unique shapes for symbol '{meta}'."
            )
        symbol_names.append(meta)
        total_tiles.append(entry["num_tiles"])
        l2_tile_size = metadata_l2_size[meta] * type_bytes.get(metadata_symbols[meta]["dtype"], 1) / 1024.0 if meta in metadata_l2_size else 0
        l2_tile_sizes.append(l2_tile_size)

    # Plot bar chart with line
    fig, ax1 = plt.subplots(figsize=(max(8, len(symbol_names) * 1.2), 5))
    bars = ax1.bar(symbol_names, total_tiles, color='skyblue', label='Total Tiles Passed')
    ax1.set_xlabel("Metadata Symbol")
    ax1.set_ylabel("Total Tiles Passed Between L2/L3", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticklabels(symbol_names, rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(symbol_names, l2_tile_sizes, color='orange', marker='o', label='Tile Size (KB)')
    ax2.set_ylabel("L2 Tile Size (KB)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title("Tiles Passed Between L3/L2 and L2 Tile Size in the Runtime Sequence")
    fig.tight_layout()
    plt.savefig(l3_tiling_file)
    logger.info(f"Saved L3 tiling plot to {l3_tiling_file}")
    plt.close()

    if "mha" in output_dir:
        logger.info("Found MHA output directory, analyzing workloads...")
        workload_at_tile = {
            # row,col
            "0,0": {"type": "GEMM", "size": 64*64*64},
            "0,1": {"type": "GEMM", "size": 16*32*256},
            "0,2": {"type": "GEMM", "size": 64*64*64},
            "0,3": {"type": "GEMM", "size": 16*32*256},
            "1,0": {"type": "GEMM", "size": 64*64*64},
            "1,1": {"type": "Softmax", "size": 16*256, "num_exps": 6*(256/16)*(768/256)}, # 6 heads, (256/16) = 16 iters in sequence dim, 768/256 = 3 iters in hidden dim
            "1,2": {"type": "GEMM", "size": 64*64*64},
            "1,3": {"type": "Softmax", "size": 16*256, "num_exps": 6*(256/16)*(768/256)}, # 6 heads, (256/16) = 16 iters in sequence dim, 768/256 = 3 iters in hidden dim
            "2,0": {"type": "GEMM", "size": 64*64*64},
            "2,1": {"type": "GEMM", "size": 16*256*32},
            "2,2": 0,
            "2,3": {"type": "GEMM", "size": 16*256*32},
            "3,0": {"type": "GEMM", "size": 64*64*64},
            "3,1": {"type": "GEMM", "size": 16*32*256},
            "3,2": {"type": "Add", "size": 16*256, "num_adds": (256/16)*(768/256)}, # (256/16) = 16 iters in sequence dim, 768/256 = 3 iters in hidden dim
            "3,3": {"type": "GEMM", "size": 16*32*256},
        }
        l3_symbol_to_workload = {
            "Wq_L3L2": ["0,0", "1,0"],
            "Wk_L3L2": ["2,0", "3,0"],
            "Wv_L3L2": ["0,2", "1,2"],
            "q_L3L2": ["0,1", "0,3"],
            "v_L3L2": ["2,1", "2,3"],
            "Wo_L3L2": ["3,1", "3,3"],
        }
        # Calculate total MACs, Exps, and Adds per tile for each L3 symbol
        total_macs_per_tile = {}
        total_exps_per_tile = {}
        total_adds_per_tile = {}

        # Use metadata_json to get num_tiles for each meta_symbol and assign workloads to tiles
        for meta_symbol, tile_names in l3_symbol_to_workload.items():
            if meta_symbol in metadata_json:
                num_tiles = metadata_json[meta_symbol]["num_tiles"]
                for tile_name in tile_names:
                    workload = workload_at_tile.get(tile_name)
                    if not workload:
                        continue
                    # MACs
                    if workload.get("type") == "GEMM":
                        total_macs_per_tile[tile_name] = total_macs_per_tile.get(tile_name, 0) + workload["size"] * num_tiles
                        logger.info(f"Tile {tile_name} workload: {workload['type']} with size {workload['size']} and num_tiles {num_tiles}, total MACs: {total_macs_per_tile[tile_name]}")
        # Basically the softmax and add workloads don't have L3 to L2 data movement, so the number of tiles was manually calculated
        for tile_name, workload in workload_at_tile.items():
            if isinstance(workload, dict) and workload["type"] in ["Softmax", "Add"]:
                # Exps
                if workload.get("type") == "Softmax":
                    num_tiles = workload.get("num_exps", 0)
                    total_exps_per_tile[tile_name] = total_exps_per_tile.get(tile_name, 0) + workload["size"] * num_tiles
                    logger.info(f"Tile {tile_name} workload: {workload['type']} with size {workload['size']} and num_exps {num_tiles}, total Exps: {total_exps_per_tile[tile_name]}")
                # Adds
                if workload.get("type") == "Add":
                    num_tiles = workload.get("num_adds", 0)
                    total_adds_per_tile[tile_name] = total_adds_per_tile.get(tile_name, 0) + workload["size"] * num_tiles
                    logger.info(f"Tile {tile_name} workload: {workload['type']} with size {workload['size']} and num_adds {num_tiles}, total Adds: {total_adds_per_tile[tile_name]}")

        # Prepare data for plotting
        all_tiles = sorted(set(list(total_macs_per_tile.keys()) +
                            list(total_exps_per_tile.keys()) +
                            list(total_adds_per_tile.keys())))
        macs_vals = [total_macs_per_tile.get(tile, 0) for tile in all_tiles]
        exps_vals = [total_exps_per_tile.get(tile, 0) for tile in all_tiles]
        adds_vals = [total_adds_per_tile.get(tile, 0) for tile in all_tiles]

        fig, ax = plt.subplots(figsize=(max(8, len(all_tiles) * 1.2), 5))
        x = np.arange(len(all_tiles))
        bar_macs = ax.bar(x, macs_vals, color='tab:blue', label='Total MACs')
        bar_exps = ax.bar(x, exps_vals, color='tab:orange', label='Total Exps')
        bar_adds = ax.bar(x, adds_vals, color='tab:green', label='Total Adds')
        ax.set_xticks(x)
        ax.set_xticklabels(all_tiles, ha='center')
        ax.set_ylabel('Total Operations')
        ax.set_xlabel('Tile (row,col)')
        ax.set_title('Total MACs, Exps, and Adds per Tile')
        ax.set_yscale('log')
        # Only show legend for bars that have nonzero values
        handles = []
        labels = []
        if any(macs_vals):
            handles.append(bar_macs)
            labels.append('Total MACs')
        if any(exps_vals):
            handles.append(bar_exps)
            labels.append('Total Exps')
        if any(adds_vals):
            handles.append(bar_adds)
            labels.append('Total Adds')
        ax.legend(handles, labels)
        plt.tight_layout()
        op_plot_file = os.path.join(output_dir, "tile_workload_ops.png")
        plt.savefig(op_plot_file)
        logger.info(f"Saved tile workload operations plot to {op_plot_file}")
        plt.close()
    else:
        logger.info("Not in MHA output directory, skipping workload analysis.")


def analyse_execution_times(csv_file, output_dir):
    """
    Analyse execution times from a CSV file.
    This function reads the CSV file containing avg, min, and max execution times,
    and generates plots of avg, min, and max execution times for each design.
    Args:
        csv_file (str): Path to the CSV file containing execution times.
        output_dir (str): Directory to save the plots.
    Returns:
        None
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        Exception: For any other unexpected errors.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} does not exist.")

    designs = []
    avg_times = []
    min_times = []
    max_times = []
    M = []
    K = []
    N = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            designs.append(row['design'])
            avg_times.append(float(row['avg_us']))
            min_times.append(float(row['min_us']))
            max_times.append(float(row['max_us']))
            M.append(int(row['M']))
            K.append(int(row['K']))
            N.append(int(row['N']))

        # Add a combined data point for designs containing 'ffn'
        ffn_indices = [i for i, d in enumerate(designs) if 'ffn' in d]
        if len(ffn_indices) == 2:
            combined_design = 'ffn'
            combined_avg = avg_times[ffn_indices[0]] + avg_times[ffn_indices[1]]
            combined_min = min_times[ffn_indices[0]] + min_times[ffn_indices[1]]
            combined_max = max_times[ffn_indices[0]] + max_times[ffn_indices[1]]
            designs.append(combined_design)
            avg_times.append(combined_avg)
            min_times.append(combined_min)
            max_times.append(combined_max)
            M.append(M[ffn_indices[0]])
            K.append(K[ffn_indices[0]])
            N.append(N[ffn_indices[0]])
        for idx in sorted(ffn_indices, reverse=True):
            del designs[idx]
            del avg_times[idx]
            del min_times[idx]
            del max_times[idx]
            del M[idx]
            del K[idx]
            del N[idx]
        # Sum times for 'mha_by_steps/only_attn_steps' and 'mha_by_steps/only_proj_steps'
        attn_idx = None
        proj_idx = None
        mha_idx = None
        for i, d in enumerate(designs):
            if d == 'mha_by_steps/only_attn_steps':
                attn_idx = i
            elif d == 'mha_by_steps/only_proj_steps':
                proj_idx = i
            elif d == 'mha':
                mha_idx = i

        if attn_idx is not None and proj_idx is not None and mha_idx is not None:
            sum_avg = avg_times[attn_idx] + avg_times[proj_idx]
            # sum_min = min_times[attn_idx] + min_times[proj_idx]
            # sum_max = max_times[attn_idx] + max_times[proj_idx]
            mha_avg = avg_times[mha_idx]
            # mha_min = min_times[mha_idx]
            # mha_max = max_times[mha_idx]

            # Compare with mha, assert difference < 5%. No need to check min/max
            # since they can vary more. 
            for label, sum_val, mha_val in [
            ('avg', sum_avg, mha_avg),
            # ('min', sum_min, mha_min),
            # ('max', sum_max, mha_max)
            ]:
                if mha_val > 0:
                    diff = abs(sum_val - mha_val) / mha_val
                    if diff >= 0.05:
                        logging.getLogger("analyse_tiling").warning(
                            f"Sum of {label} times for attn+proj ({sum_val}) differs from mha ({mha_val}) by more than 5% ({diff*100:.2f}%)"
                        )
                        for idx in [attn_idx, proj_idx]:
                            logging.getLogger("analyse_tiling").warning(
                                f"Removing design {designs[idx]} with {label} time {avg_times[idx]} from the plot"
                            )
                            del designs[idx]
                            del avg_times[idx]
                            del min_times[idx]
                            del max_times[idx]
                            del M[idx]
                            del K[idx]
                    else:
                        # Delete mha entry
                        del designs[mha_idx]
                        del avg_times[mha_idx]
                        del min_times[mha_idx]
                        del max_times[mha_idx]
                        del M[mha_idx]
                        del K[mha_idx]
            del N[mha_idx]
        # Replace 'mha_by_steps/only_attn_steps' with 'mha-attn' in designs
        designs = ['mha-attn' if d == 'mha_by_steps/only_attn_steps' else d for d in designs]
        # Replace 'mha_by_steps/only_proj_steps' with 'mha-proj' in designs
        designs = ['mha-proj' if d == 'mha_by_steps/only_proj_steps' else d for d in designs]

    # Sort the data by avg_times (lowest to highest)
    sorted_data = sorted(zip(avg_times, min_times, max_times, designs, M, K, N))
    avg_times, min_times, max_times, designs, M, K, N = map(list, zip(*sorted_data))
    fig, ax = plt.subplots(figsize=(max(8, len(designs) * 1.2), 5))
    x = range(len(designs))
    ax.bar(x, avg_times, color='skyblue', label='Avg (us)')
    ax.errorbar(x, avg_times, yerr=[ [avg - min for avg, min in zip(avg_times, min_times)],
                                        [max - avg for avg, max in zip(avg_times, max_times)] ],
                fmt='o', color='orange', label='Min/Max (us)')
    ax.set_xticks(x)
    # ax.set_xticklabels(designs, rotation=45, ha='right')
    ax.set_xticklabels(designs, ha='center')
    ax.set_ylabel('Execution Time (us)')
    ax.set_xlabel('Design')
    ax.set_title('Execution Times Across Designs')
    ax.legend()

    # Annotate each bar with "MxKxN" and the product
    # TODO: The number of AIEs is hardcoded, but it should be determined from the design files.
    y_max = ax.get_ylim()[1]
    # Increase the y-axis upper limit by 50% to add whitespace at the top
    ax.set_ylim(top=y_max * 1.5)
    y_text = y_max * 1.45  # Place annotation at the original top (now below the new limit)
    for idx, (m, k, n, bar_height, design) in enumerate(zip(M, K, N, avg_times, designs)):
        text = ""
        if design == 'ffn':
            text = f"{2*m*k*n/1e6:.1f}×10$^9$ MACs\n16 cores"
        elif design == 'mha':
            total_macs = 4 * m * k * n  + m * k * m + m * m * n
            total_exps = m * m
            total_adds = m * n
            text = f"{total_macs/1e6:.1f}×10$^9$ MACs\n12 cores\n{total_exps/1e3:.1f}×10$^3$ Exps\n2 cores\n{total_adds/1e3:.1f}×10$^3$ Adds\n1 core"
        elif design == 'mha-attn':
            total_macs = m * k * n + m * k * m + m * m * n
            total_exps = m * m
            total_adds = m * n
            text = f"{total_macs/1e6:.1f}×10$^9$ MACs\n6 cores\n{total_exps/1e3:.1f}×10$^3$ Exps\n2 cores\n{total_adds/1e3:.1f}×10$^3$ Adds\n1 core"
        elif design == 'mha-proj':
            total_macs = 3 * m * k * n
            text = f"{total_macs/1e6:.1f}×10$^9$ MACs\n6 cores"
        if text:
            ax.text(idx, y_text, text, ha='center', va='top', fontsize=9, rotation=0)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "execution_times.png")
    plt.savefig(plot_file)
    logging.getLogger("analyse_tiling").info(f"Saved execution times plot to {plot_file}")
    plt.close()


def analyse_fine_grained_times(csv_file, output_dir):
    """
    Analyse execution times from a CSV file.
    This function reads the CSV file containing avg, min, and max execution times,
    and generates plots of avg, min, and max execution times for each design.
    Args:
        csv_file (str): Path to the CSV file containing execution times.
        output_dir (str): Directory to save the plots.
    Returns:
        None
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        Exception: For any other unexpected errors.
    """
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} does not exist.")

    step = []
    avg_times = []
    min_times = []
    max_times = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            step.append(int(row['step']))
            avg_times.append(float(row['avg_us']))
            min_times.append(float(row['min_us']))
            max_times.append(float(row['max_us']))

    # Sort the data by avg_times (lowest to highest)
    sorted_data = sorted(zip(avg_times, min_times, max_times, step))
    avg_times, min_times, max_times, step = map(list, zip(*sorted_data))
    fig, ax = plt.subplots(figsize=(max(8, len(step) * 1.2), 5))
    x = range(len(step))
    ax.bar(x, avg_times, color='skyblue', label='Avg (us)')
    ax.errorbar(x, avg_times, yerr=[ [avg - min for avg, min in zip(avg_times, min_times)],
                                        [max - avg for avg, max in zip(avg_times, max_times)] ],
                fmt='o', color='orange', label='Min/Max (us)')
    ax.set_xticks(x)
    # ax.set_xticklabels(step, rotation=45, ha='right')
    ax.set_xticklabels(step, ha='center')
    ax.set_ylabel('Execution Time (us)')
    ax.set_xlabel('Step')
    ax.set_title('Execution Times Across Steps')
    ax.legend()

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "fine_grained_times.png")
    plt.savefig(plot_file)
    logging.getLogger("analyse_tiling").info(f"Saved execution times plot to {plot_file}")
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse tiling and memory usage for MHA.")
    parser.add_argument("--dev", type=str, choices=["npu", "npu2"], required=True, help="Type of device: 'npu' or 'npu2'.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file to parse.")
    parser.add_argument("--task", type=str, choices=["mem-util", "comp-dist", "comp-util", "loop-iters", "exec-times", "fine-grained"], required=True, help="Task to perform: 'mem-util', 'comp-dist', 'comp-util', 'loop-iters', 'exec-times', or 'fine-grained'.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to write output files to.")
    args = parser.parse_args()

    task = args.task
    if task == "mem-util":
        analyse_memory_utilization(args.dev, args.input_file, args.output_dir)
    elif task == "comp-dist":
        analyse_computation_distribution(args.input_file, args.output_dir)
    elif task == "loop-iters":
        analyse_loop_iterations(args.input_file, args.output_dir)
    elif task == "exec-times":
        analyse_execution_times(args.input_file, args.output_dir)
    elif task == "fine-grained":
        analyse_fine_grained_times(args.input_file, args.output_dir)