import os
import re
from collections import defaultdict
import json
from collections import OrderedDict

# Path to the input file
input_file = os.path.join(
    os.path.dirname(__file__),
    "build/aie_mha.mlir.prj/input_physical.mlir"
)

# Regex to match buffer lines
buffer_re = re.compile(
    r'%\w+\s*=\s*aie\.buffer\((?:%mem_tile_|%tile_)?(\d+)_(\d+)\).*memref<(\d+)\s*x\s*(i\d+|bf\d+|f\d+)>'
)

# Regex to match the aie.flow line
flow_re = re.compile(r'aie\.flow\(%tile_\d+_\d+, DMA : \d+, %tile_\d+_\d+, DMA : \d+\)')

# Map bit width to bytes
type_bytes = {
    'i8': 1,
    'i16': 2,
    'bf16': 2,
    'i32': 4,
    'f32': 4,
}

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
                print(f"Unknown dtype {dtype} in line: {line.strip()}")
                continue
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

print("Tile memory usage (in bytes and kB):")
for tile, total_bytes in sorted(tile_mem.items()):
    print(f"Tile {tile}: {total_bytes} bytes ({total_bytes / 1024:.2f} kB)")
    for buf in tile_buffers[tile]:
        print(f"  Buffer: {buf['sym_name']}, {buf['num_elem']} x {buf['dtype']} = {buf['bytes']} bytes")

# Parse loop upper bounds per tile from aie_mha.mlir
core_loop_bounds = {}

aie_mha_file = os.path.join(
    os.path.dirname(__file__),
    "build/aie_mha.mlir"
)

core_block_re = re.compile(r'%core_(\d+)_(\d+)\s*=\s*aie\.core\(%tile_(\d+)_(\d+)\)\s*{')
scf_for_re = re.compile(r'scf\.for\s+%arg\d+\s*=\s*%c\d+(?:_\d+)?\s+to\s+%c(\d+)(?:_\d+)?\s+step\s+%c\d+(?:_\d+)?\s*{')

with open(aie_mha_file, 'r') as f:
    core_loop_bounds = {}
    inside_core = False
    tile = None
    for line in f:
        if not inside_core:
            m = core_block_re.match(line.strip())
            if m:
                tile = f"({int(m.group(3))}, {int(m.group(4))})"
                inside_core = True
                scf_for_count = 0
        else:
            # Check for matmul call
            if "matmul" in line:
                # Example: func.call @matmul_bf16_bf16_32_24_256(...)
                call_match = re.search(r'@matmul_([^(\s]+)\(', line)
                if call_match:
                    matmul_str = call_match.group(1)  # e.g., 'bf16_bf16_32_24_256'
                    parts = matmul_str.split('_')
                    if len(parts) >= 5:
                        dtype = f"{parts[0]}_{parts[1]}"
                        m, k, n = map(int, parts[-4:-1])
                        macs = m * k * n
                        if tile not in core_loop_bounds:
                            core_loop_bounds[tile] = {}
                        core_loop_bounds[tile]['matmul'] = {
                            'dtype': dtype.split('_')[0],  # e.g., 'bf16'
                            'm': m,
                            'k': k,
                            'n': n,
                            'macs_one_iter': macs
                        }
            m = scf_for_re.search(line)
            if m:
                scf_for_count += 1
                if scf_for_count == 1:
                    continue  # Skip the first scf.for
                matmul_instances = int(m.group(1))
                if tile not in core_loop_bounds:
                    core_loop_bounds[tile] = {}
                    core_loop_bounds[tile]['instances'] = matmul_instances
                else:
                    core_loop_bounds[tile]['instances'] *= matmul_instances
            if line.strip() == "}":
                inside_core = False
                tile = None

print("\nTile loop upper bounds and matmul info:")
for tile, info in sorted(core_loop_bounds.items()):
    if isinstance(info, dict) and 'matmul' in info:
        matmul = info['matmul']
        instances = info.get('instances', 1)
        print(f"Tile {tile}: {instances} iterations of {matmul['m']}x{matmul['k']}x{matmul['n']}, matmul dtype={matmul['dtype']}, MACs={matmul['macs_one_iter'] * instances}")
    else:
        print(f"Tile {tile}: {info.get('instances', 1)} iterations")

with open(aie_mha_file, 'r') as f:
    inside_sequence = False
    metadata_symbols = OrderedDict()
    metadata_shapes = {}
    metadata_shape_counts = {}

    for line in f:
        if not inside_sequence:
            if re.search(r'aiex\.runtime_sequence\s+@sequence\(', line):
                inside_sequence = True
        else:
            if line.strip() == "}":
                break

            # Match lines with metadata
            meta_match = re.search(r'@([^}\s]+)}', line)
            shape_match = re.search(r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]', line)
            if meta_match and shape_match:
                meta_str = meta_match.group(1)
                shape_str = shape_match.group(2)
                shape_nums = tuple(int(x.strip()) for x in shape_str.split(','))
                if meta_str not in metadata_symbols:
                    metadata_symbols[meta_str] = None
                if meta_str not in metadata_shapes:
                    metadata_shapes[meta_str] = []
                metadata_shapes[meta_str].append(shape_nums)
                # Count unique shapes
                if meta_str not in metadata_shape_counts:
                    metadata_shape_counts[meta_str] = {}
                metadata_shape_counts[meta_str][shape_nums] = metadata_shape_counts[meta_str].get(shape_nums, 0) + 1

# Prepare and print metadata summary in a more readable format
print("\nMetadata counts and shapes:")
metadata_json = {}
for meta in metadata_symbols.keys():
    entry = {
        "unique_shapes": [],
        "num_tiles": 1
    }
    if meta in metadata_shapes:
        unique_shapes = metadata_shape_counts[meta]
        for shape, shape_count in sorted(unique_shapes.items()):
            entry["unique_shapes"].append({
                "shape": shape,
                "shape_count": shape_count
            })
            entry["num_tiles"] *= shape[0] * shape[1] * shape_count
    else:
        entry["unique_shapes"] = []
        entry["num_tiles"] = 0
    metadata_json[meta] = entry

# Print in a more human-friendly way
for meta, entry in metadata_json.items():
    print(f"\nMetadata symbol: {meta}")
    print(f"  Total tiles passed: {entry['num_tiles']}")
    if entry["unique_shapes"]:
        print("  Unique shapes and counts:")
        for shape_info in entry["unique_shapes"]:
            print(f"    Shape: {shape_info['shape']}, Count: {shape_info['shape_count']}")
    else:
        print("  No shapes found.")

# Optionally, still print the JSON for reference
#print(json.dumps(metadata_json, indent=2))
