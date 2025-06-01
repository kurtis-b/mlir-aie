import os
import re
from collections import defaultdict

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

print("Tile memory usage (in bytes and kB):")
for tile, total_bytes in sorted(tile_mem.items()):
    print(f"Tile {tile}: {total_bytes} bytes ({total_bytes / 1024:.2f} kB)")

# Parse loop upper bounds per tile from aie_mha.mlir
core_loop_bounds = {}

aie_mha_file = os.path.join(
    os.path.dirname(__file__),
    "build/aie_mha.mlir"
)

core_block_re = re.compile(r'%core_(\d+)_(\d+)\s*=\s*aie\.core\(%tile_(\d+)_(\d+)\)\s*{')
scf_for_re = re.compile(r'scf\.for\s+%arg\d+\s*=\s*%c\d+(?:_\d+)?\s+to\s+%c(\d+)(?:_\d+)?\s+step\s+%c\d+(?:_\d+)?\s*{')

with open(aie_mha_file, 'r') as f:
    inside_core = False
    tile = None
    for line in f:
        if not inside_core:
            m = core_block_re.match(line.strip())
            if m:
                tile = (int(m.group(3)), int(m.group(4)))
                inside_core = True
                scf_for_count = 0
        else:
            m = scf_for_re.search(line)
            if m:
                scf_for_count += 1
                if scf_for_count == 1:
                    continue  # Skip the first scf.for
                matmul_instances = int(m.group(1))
                if tile not in core_loop_bounds:
                    core_loop_bounds[tile] = matmul_instances
                else:
                    core_loop_bounds[tile] *= matmul_instances
            if line.strip() == "}":
                inside_core = False
                tile = None

print("\nTile loop upper bounds:")
for tile, matmul_instances in sorted(core_loop_bounds.items()):
    print(f"Tile {tile}: {matmul_instances}")