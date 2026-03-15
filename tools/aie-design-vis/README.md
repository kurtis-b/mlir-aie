# AIE Design Visualization Tool

This directory is the staging area for a design-level AIE visualization tool.

The goal is to build a viewer for compiled MLIR designs that can show:

- physical tile layout
- compute placement
- DMA and buffer configuration
- routed stream paths across the array

The implementation plan for the full tool lives in
[docs/DesignVisualizationPlan.md](/home/agi-demo/mlir-aie/docs/DesignVisualizationPlan.md).

## Seed Files

This directory currently starts from copies of existing project tools and now
contains the first Phase 1 browser viewer for design JSON.

- [index.html](/home/agi-demo/mlir-aie/tools/aie-design-vis/index.html)
  - copied from [tools/aie-vis/aie-vis.html](/home/agi-demo/mlir-aie/tools/aie-vis/aie-vis.html)
  - serves as the browser-viewer prototype

- [visualize.py](/home/agi-demo/mlir-aie/tools/aie-design-vis/visualize.py)
  - copied from [tools/aie-routing-command-line/visualize.py](/home/agi-demo/mlir-aie/tools/aie-routing-command-line/visualize.py)
  - serves as the current ASCII-routing reference

- [mlir2json.sh](/home/agi-demo/mlir-aie/tools/aie-design-vis/mlir2json.sh)
  - copied from [tools/aie-routing-command-line/mlir2json.sh](/home/agi-demo/mlir-aie/tools/aie-routing-command-line/mlir2json.sh)
  - updated to call `aie-translate --aie-design-to-json`
  - serves as the current design-JSON export helper

- [switchbox.json](/home/agi-demo/mlir-aie/tools/aie-design-vis/switchbox.json)
  - copied from [tools/aie-routing-command-line/switchbox.json](/home/agi-demo/mlir-aie/tools/aie-routing-command-line/switchbox.json)
  - serves as sample input for the ASCII renderer

## Intended Direction

This directory should evolve toward:

1. a browser-based viewer driven by exported JSON
2. a new `aie-translate --aie-design-to-json` exporter
3. a design-wide view that combines tile usage, DMA config, and routed streams

## Current Status

The initial Phase 1 pieces now exist:

- `aie-translate --aie-design-to-json`
- a simple browser viewer in [index.html](/home/agi-demo/mlir-aie/tools/aie-design-vis/index.html)

The current viewer can:

- load exported design JSON
- render the tile grid
- scale the array view to the visible viewer pane automatically
- color tiles by kind and usage
- show routed stream segments
- filter by route
- filter by packet flow
- filter by DMA
- filter by group
- filter by flow kind
- filter by tile coordinate
- filter by symbol or resource name
- toggle route, switchbox, resource, and packet-endpoint overlays
- inspect tile and switchbox metadata
- inspect selected groups as first-class sidebar objects, including resolved
  buffers, locks, DMA containers, streams, and packet flows
- inspect core programs on a selected tile through a dedicated collapsed
  `Core Programs` section
- show per-tile buffer, lock, and DMA counts
- show packet source/destination badges on tiles
- highlight the tile of the selected DMA container
- inspect exported buffer, lock, DMA channel, BD-block, and core-body detail in
  collapsible sidebar sections
- inspect packet-flow sources and destinations for the selected tile
- keep the raw selection payload in a separate closed-by-default `Raw Selection
  JSON` section

The current exporter emits:

- `device`, `metadata`, `tiles`
- `cores` with core placement, attributes, operation names, and printed core
  bodies
- `streams`, `switchboxes`
- `buffers`, `locks`
- `dmas` with channel summaries and BD-block detail for both `aie.dma_start`
  chains and structured `aie.dma` regions
- `packet_flows` with packet IDs and endpoint summaries
- `groups` for currently recoverable lowered families:
  - `row_store`
  - `objectfifo`
  - `packet_flow`
  - `flow` for unmatched circuit-stream families grouped by source endpoint

The current viewer uses those exports to drive three layers of inspection:

- selection by route, packet flow, or DMA container
- selection by recovered group
- filtering by flow kind, tile coordinate, and symbol text
- per-tile and per-group drilldown for buffers, locks, DMA channels, BD
  blocks, routed streams, and packet endpoints

Stream entries now include:

- `kind` (`circuit` or `packet`)
- source and destination endpoint metadata where available
- `destinations` for fanout-safe circuit and packet stream summaries
- `group_candidates` so recovered groups can narrow routed paths as well as
  static resources
- `group_matches` for circuit streams, recording DMA-aware provenance scores
  when multiple lowered families share the same endpoint tiles
- `provenance` for circuit streams, marking whether grouping was `resolved`,
  `ambiguous`, or `fallback`, and whether that came from `dma_channel`,
  `tile_membership`, or `flow_group` fallback

Current grouping is based on lowered symbol names and packet-flow endpoints, so
it is strongest for row-store and objectfifo-heavy designs. Circuit-stream
routes now carry endpoint metadata and candidate-group associations, and
unmatched user-defined circuit routes are exported as explicit `flow` groups.
Correlation for overlapping circuit flows is now improved with DMA-channel-aware
scoring. The remaining unresolved cases are now surfaced explicitly through
stream-level `provenance` instead of being hidden as silent ties, but this is
still not a full provenance model.

The next planned step is still stronger provenance for cases that cannot be
disambiguated by DMA channel ownership plus tile membership alone.
