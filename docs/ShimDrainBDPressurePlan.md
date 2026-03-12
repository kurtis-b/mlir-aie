# Shim-Drain BD Pressure Follow-Up

## Status

The frozen encoder repros in
`/home/agi-demo/iron/operators/encoder_pipeline/shim_drain_bd_repro/README.md`
now compile cleanly through `aiecc` with the current compiler tree.

The earlier shim-drain compaction plan turned out to be the wrong diagnosis for
those repros. The actual blocker was static memtile DMA-channel selection in
`aie-objectFifo-stateful-transform`, not runtime shim-drain task lowering.

## Correct Diagnosis

The failing path in both frozen repros was:

1. `aie-objectFifo-stateful-transform`
2. `aie-assign-bd-ids`

The failure happened before runtime shim tasks could matter.

The exact exposed hotspot was the memtile-local `outLN2_cons` chain on
`%mem_tile_7_1`, not `aiex.dma_configure_task_for @memLN2`.

The root cause was:

- memtile BD IDs are split by channel parity: even channels share BD IDs
  `0..23`, odd channels share BD IDs `24..47`
- `AIEObjectFifoStatefulTransform` used first-fit DMA-channel assignment
- deep replay links and later small drains could land on the same parity bank
  even when another free channel on the opposite bank would fit

That meant the final LN2 drain looked like a shim-drain problem in the frozen
source, but the real failure was static memtile BD-bank pressure created before
runtime lowering.

## Implemented Fix

Two compiler changes address the exposed issue.

### 1. Runtime-sequence BD assignment now respects task channel

`AIEAssignRuntimeSequenceBDIDs` now allocates BD IDs using the actual DMA
channel of each `aiex.dma_configure_task`, rather than always assuming channel
`0`.

This is required for any tile whose BD IDs are partitioned by channel parity.

### 2. Memtile objectfifo channel selection is now BD-bank-aware

`AIEObjectFifoStatefulTransform` now:

- estimates the number of BDs each objectfifo lowering will need on the chosen
  producer/consumer tile
- tracks provisional BD usage per tile with `BdIdGenerator`
- chooses among currently free DMA channels by remaining provisional BD
  capacity, while still honoring adjacent-tile channel restrictions

This keeps deep objectfifo chains from consuming a parity bank that later DMA
starts on the same memtile still need.

In the reduced `%mem_tile_7_1` repro, the stable channel shape is now:

- deep replay: `MM2S 0`, `S2MM 1`
- side input: `S2MM 0`, `MM2S 1`
- tail drain: `MM2S 2`, `S2MM 3`

That assignment keeps even-bank demand under `24` BDs and odd-bank demand under
`24` BDs, so `aie-assign-bd-ids` succeeds.

## Regression Coverage

The implemented regression tests are:

- [good-7.mlir](/home/agi-demo/mlir-aie/test/bd-chains-and-dma-tasks/assign-runtime-sequence-bd-ids/good-7.mlir)
  for channel-aware runtime-sequence BD assignment on memtiles
- [memtile_channel_bd_aware_assignment.mlir](/home/agi-demo/mlir-aie/test/Passes/assign-bd-ids/memtile_channel_bd_aware_assignment.mlir)
  for the reduced encoder-style memtile pressure pattern that previously failed
  in `aie-assign-bd-ids`

## Validation

The fix was validated in three ways.

### Reduced compiler regression

The reduced `%mem_tile_7_1` repro now passes:

```bash
aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids \
  test/Passes/assign-bd-ids/memtile_channel_bd_aware_assignment.mlir
```

### Frozen encoder repros

Both frozen encoder repros now get through `aiecc` resource allocation and core
compilation:

- `memln2_drain_1ph_1pf_12pacc.aiecc_failure.mlir`
- `memln2_drain_4ph_4pf_12pacc_4opg.aiecc_failure.mlir`

using:

```bash
build/bin/aiecc --verbose --no-compile --no-link --no-xchesscc --no-xbridge \
  /home/agi-demo/iron/operators/encoder_pipeline/shim_drain_bd_repro/...
```

### Scope check

The fix does not introduce a new shim-drain lowering path. It only changes:

- runtime-sequence BD-channel correctness
- static memtile DMA-channel selection during objectfifo lowering

So the observed improvement comes from better compile-time resource placement,
not from changing the runtime drain model.

## Remaining Work

The original shim-drain compaction idea is still a possible future direction,
but it is no longer the planned fix for the frozen encoder repros.

Only revisit compacted shim-drain lowering if a future repro shows a real
runtime-task BD-pressure problem after static memtile channel assignment has
already succeeded.
