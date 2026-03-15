# AIE Design Visualization Plan

## Goal

Add a visualization flow for AIE MLIR designs that shows:

- the physical NPU array layout
- which tiles are active
- where compute is placed
- how DMA engines and buffers are configured
- how streams are routed across the array

The intended input is a compiled AIE MLIR design, not high-level Python IR.
The best-supported starting point is a routed physical AIE module after:

- `aie-create-pathfinder-flows`
- `aie-find-flows`

The richer configuration view should work even better after DMA lowering and
BD assignment, when the IR contains:

- `aie.mem`
- `aie.memtile_dma`
- `aie.shim_dma`
- `aie.dma_start`
- `aie.dma_bd`
- `aie.buffer`
- `aie.lock`

## Existing Pieces To Reuse

This project already has most of the individual pieces needed for a complete
design visualizer.

### 1. Tile-layout viewer

[aie-visualize.cpp](/home/agi-demo/mlir-aie/tools/aie-visualize/aie-visualize.cpp)
already parses an MLIR file, looks up the target model, and renders a terminal
grid showing which tiles are used.

Useful parts to reuse:

- device parsing
- target-model lookup
- tile classification (`core`, `mem`, `shim`)

Limitation:

- it only shows tile occupancy
- it does not show routed streams or DMA configuration

### 2. Routed-flow JSON exporter

[AIETargets.cpp](/home/agi-demo/mlir-aie/lib/Targets/AIETargets.cpp) already
registers:

- `aie-translate --aie-flows-to-json`
- `aie-translate --aie-generate-json`

The first is the most useful reference for the new tool because it already
exports routed hop-by-hop flow data from MLIR.

Useful parts to reuse:

- translator registration pattern
- JSON formatting conventions
- flow hop serialization

### 3. Routing ASCII viewer

[visualize.py](/home/agi-demo/mlir-aie/tools/aie-routing-command-line/visualize.py)
renders routed-flow JSON into per-flow ASCII diagrams. The user-facing workflow
is documented in [AIERouting.md](/home/agi-demo/mlir-aie/docs/AIERouting.md#L99).

Useful parts to reuse:

- JSON assumptions about routed hop lists
- visual conventions for sources, destinations, and highlighted paths
- the distinction between circuit-switched and packet-switched flows

Limitation:

- output is static text files
- the viewer is per-flow, not design-wide
- it does not show compute placement, buffers, locks, or DMA BD chains

### 4. Experimental browser prototype

[aie-vis.html](/home/agi-demo/mlir-aie/tools/aie-vis/aie-vis.html) is an early
HTML viewer prototype for AI Engine switch structures.

Useful parts to reuse:

- browser-based presentation
- tile/switch rendering concepts

Limitation:

- it is not integrated with MLIR or JSON export
- it currently pulls Konva from a CDN, which is not ideal for an offline dev
  tool

## Proposed Architecture

Build the visualization flow as two separate components:

1. a new MLIR-to-JSON exporter
2. a browser-based viewer that consumes that JSON

This is better than extending the terminal tools because:

- routed-stream inspection quickly becomes too dense for plain text
- the UI needs filtering and click/hover inspection
- JSON creates a stable testing boundary between compiler export and frontend

## Proposed Exporter

Add a new translation in
[AIETargets.cpp](/home/agi-demo/mlir-aie/lib/Targets/AIETargets.cpp):

- `aie-translate --aie-design-to-json`

This should follow the same registration style as:

- `--aie-flows-to-json`
- `--aie-generate-json`

### Input Contract

The exporter should accept a physical AIE MLIR module. It should support
multiple levels of detail depending on what ops are present.

Minimum supported input:

- `aie.device`
- `aie.tile`
- routed flows (`aie.flow`, `aie.packet_flow`, or already materialized routed
  state that can be traced back to them)

Enhanced input:

- `aie.core`
- `aie.mem`
- `aie.memtile_dma`
- `aie.shim_dma`
- `aie.buffer`
- `aie.lock`
- `aie.dma_start`
- `aie.dma_bd`

### Output Schema

Top-level JSON object:

- `schema_version`
- `device`
- `tiles`
- `streams`
- `buffers`
- `locks`
- `dmas`
- `packet_flows`
- `metadata`

#### `device`

- target device name
- number of columns
- number of rows
- row/column metadata from the target model

#### `tiles[]`

One entry per tile:

- `col`
- `row`
- `kind`: `core`, `mem`, `shim_noc`, `shim_pl`
- `used`
- `core_present`
- `mem_present`
- `memtile_dma_present`
- `shim_dma_present`
- `symbols`

Optional per-tile children:

- `buffers`: symbol names and types
- `locks`: symbol names and init values
- `dma_channels`: summarized by direction and channel index

#### `streams[]`

One entry per routed stream:

- `id`
- `kind`: `flow` or `packet_flow`
- `source`
- `destinations`
- `route`

`route` should reuse the shape already produced by
`--aie-flows-to-json` wherever possible.

#### `dmas[]`

One entry per DMA channel start:

- owning tile
- tile kind
- direction
- channel index
- chain head
- BD list

Per BD:

- `bd_id`
- `next_bd_id`
- `buffer`
- `offset`
- `length`
- lock acquire/release summary if present

#### `metadata`

- input filename if available
- whether the design was already routed
- whether BD IDs were assigned
- exporter version

## Proposed Viewer

Add a browser UI under either:

- [tools/aie-vis](/home/agi-demo/mlir-aie/tools/aie-vis)
- or a new directory such as `tools/aie-design-vis`

The recommended path is to reuse `tools/aie-vis` and replace the current
prototype with a data-driven viewer.

### Viewer Requirements

#### Phase 1: Array + route inspection

- render the array grid
- color tiles by type
- highlight used tiles
- draw routed streams over the grid
- click a stream to highlight its path
- click a tile to show tile metadata

This phase gives immediate value for routed design inspection.

#### Phase 2: Configuration overlays

Add layer toggles for:

- compute tiles / core presence
- DMA channels
- buffers
- locks
- packet flows

Add a side panel for tile details:

- buffers on the tile
- locks on the tile
- DMA starts and their channels
- BD chains associated with each start

#### Phase 3: Higher-level grouping

Group related resources by origin where possible:

- objectfifo-generated paths
- row-store-generated paths
- explicit user flows
- packet-switched routes

This requires symbol and use-site correlation in the exporter, but it is the
best way to make complex designs readable.

Current status:

- initial grouping is implemented for lowered `row_store` and `objectfifo`
  resource families using recovered symbol prefixes
- `packet_flow` groups are exported directly from packet-flow endpoints
- routed `streams` now export `kind`, endpoint metadata, and `group_candidates`
- unmatched user-defined circuit routes are now exported as explicit `flow`
  groups keyed by source endpoint, with fanout-safe `destinations` metadata on
  circuit streams
- overlapping circuit routes now use DMA-channel-aware provenance scoring and
  export `group_matches` details alongside `group_candidates`
- circuit-stream grouping is still not a full provenance model when multiple
  lowered families cannot be disambiguated by DMA channel ownership

## UI Direction

Prefer a static HTML + JS viewer that works offline from a generated JSON file.

Do not make the viewer depend on a CDN. The existing
[aie-vis.html](/home/agi-demo/mlir-aie/tools/aie-vis/aie-vis.html) references
Konva from `unpkg`, but the implementation for this tool should avoid network
dependencies. A plain SVG or Canvas renderer is enough for v1.

Suggested command flow:

```bash
aie-translate --aie-design-to-json design.mlir > design.json
python3 -m http.server
# then open the viewer and load design.json
```

Later, a helper wrapper can make this simpler:

```bash
aie-design-vis design.mlir
```

where the wrapper:

1. exports JSON
2. launches a local static viewer

## Implementation Plan

### Phase 1: Export tile and route data

Compiler work:

- add `--aie-design-to-json`
- export:
  - device geometry
  - tile list
  - routed streams

Viewer work:

- render the grid
- render routed paths
- add stream selection

Tests:

- one simple flow test
- one packet-flow test
- schema check with `FileCheck`

### Phase 2: Export compute and DMA config

Compiler work:

- export `aie.core`, `aie.mem`, `aie.memtile_dma`, `aie.shim_dma`
- export `aie.buffer`, `aie.lock`
- export `aie.dma_start` and `aie.dma_bd` chains

Viewer work:

- tile detail sidebar
- DMA/BD overlay
- buffer/lock summary

Tests:

- one lowered core DMA case
- one memtile DMA case
- one shim DMA case

### Phase 3: Design-aware grouping

Compiler work:

- attach optional provenance labels where practical
- correlate stream and DMA resources to higher-level constructs

Viewer work:

- filter by symbol
- filter by tile
- filter by flow kind
- group related resources in the sidebar

Current status:

- symbol, tile, and flow-kind filtering are implemented
- selected groups now expand in the sidebar into related buffers, locks, DMA
  containers, routed streams, and packet flows
- exported designs now include top-level `cores` data with core placement,
  attributes, operation names, and printed core bodies
- Phase 2 DMA export now covers both `aie.dma_start` BD chains and structured
  `aie.dma` regions inside `aie.mem`, `aie.memtile_dma`, and `aie.shim_dma`
- overlapping circuit-flow endpoints now use DMA-aware provenance scoring in
  the exporter
- circuit streams now export explicit `provenance` state so unresolved ties are
  marked as `ambiguous` or `fallback` instead of being silently guessed
- the viewer now presents selection details as collapsed sections instead of a
  single raw JSON blob, and it auto-fits the array SVG to the visible pane
- the main remaining Phase 3 gap is stronger provenance for cases where
  endpoint, DMA, and tile-membership evidence are still insufficient to
  disambiguate intent

Tests:

- one objectfifo-heavy design
- one row-store-heavy design

## Testing Strategy

### Export tests

Add lit tests similar to the existing JSON exporter tests:

- [simple.mlir](/home/agi-demo/mlir-aie/test/Targets/AIEFlowsToJSON/simple.mlir)
- [shim_alloc.mlir](/home/agi-demo/mlir-aie/test/Targets/AIEGenerateJSON/shim_alloc.mlir)

Recommended new test directories:

- `test/Targets/AIEDesignToJSON/`

Initial test set:

1. simple routed flow
2. packet flow
3. lowered DMA chain
4. memtile row-store lowered design

### Viewer tests

Keep the viewer mostly data-driven and testable without a browser framework.

Minimum:

- one golden JSON example checked into the repo
- a small JS unit test for parsing/render preparation if a JS test harness is
  added

## Non-Goals

The first version should not try to:

- visualize runtime trace timing
- animate execution
- display every switchbox micro-connection on first load
- replace Perfetto-based trace visualization

Those are separate tools. This one is for static compiled-design inspection.

## Recommended First Milestone

The best first implementation slice is:

1. add `--aie-design-to-json`
2. export device, tiles, and routed streams
3. render a clickable HTML grid with route highlighting

That already solves the main use case:

"Given this compiled MLIR design, show me where compute landed and how the
streams move through the array."

Only after that should the exporter grow DMA, buffer, lock, and BD detail.
