# Shim-Drain BD Pressure Plan

## Problem

The remaining high-`part_count` failure after converting same-tile replay paths
to `aie.memtile_row_store` is no longer a row-store issue. The frozen repros in
`/home/agi-demo/iron/operators/encoder_pipeline/shim_drain_bd_repro/README.md`
show that the exposed limit is now the final memtile-to-shim drain path for the
LN2 output stream.

In both frozen cases:

- the source still contains the tail drain objectfifo `@memLN2`
- the runtime sequence still contains `aiex.dma_configure_task_for @memLN2`
- `aiecc` fails with shim-channel BD exhaustion during resource allocation

The important conclusion is that the row-store conversion moved the bottleneck
off the same-tile replay paths and onto the final shim drain.

## Key Observations From The Repro

The repro indicates all of the following.

- The failure is compile-time only.
- The remaining hotspot is the final output drain path, not LN1/LN2/O-proj or
  FFN-down row staging.
- Simplifying the `memLN2` `dimensionsToStream` form to a direct row-major
  drain did not move the boundary.
- Therefore the problem is not the row-store lowering and not the specific
  `dimensionsToStream` permutation used on `@memLN2`.
- The real pressure is the number of shim-drain taps and the way the current
  compiler lowers that drain path onto one shim DMA channel.

That means a fix should target shim-drain lowering directly, not the row-store
implementation.

## Existing Compiler Pieces To Preserve

The current compiler already has useful machinery that should remain part of
the solution.

- `AIEObjectFifoStatefulTransform` rewrites shim-facing objectfifos into
  `aie.shim_dma_allocation` metadata and runtime-facing accesses.
- `AIESubstituteShimDMAAllocations` turns
  `aiex.dma_configure_task_for @alloc` into tile/channel-specific
  `aiex.dma_configure_task`.
- `AIEAssignRuntimeSequenceBDIDs` already reuses BD IDs after
  `aiex.dma_free_task` and `aiex.dma_await_task`.

So the right direction is not "raise the BD limit" or "replace the runtime-task
reuse pass." The missing piece is a lowering shape that keeps the number of
live shim-drain tasks and shim-drain BD configurations bounded before that pass
runs.

## Recommended Plan

### Phase 0: Lock in the current failure as regression coverage

Before changing the lowering, add compiler-side tests that model the failing
shape in a reduced form.

Required tests:

- one synthetic `aie.objectfifo` memtile-to-shim drain that exhausts shim BDs
  under the current lowering
- one test that proves the failure is insensitive to the exact
  `dimensionsToStream` layout when total drain tap count is unchanged
- one test that captures the expected shim channel in the diagnostic

The goal is to preserve the actual failure mode while developing the fix.

### Phase 1: Treat large shim drains as a distinct lowering case

The current compiler effectively treats the final drain like a normal
shim-facing objectfifo. That is too literal for large output drains whose
access pattern is regular and whose pressure comes from repeated host/NPU drain
requests.

Introduce a distinct lowering path for objectfifos that satisfy all of:

- producer side is an AIE tile or memtile
- consumer side is a shim tile
- the drain is only used by runtime-sequence DMA task configuration
- every task uses the same shim tile, same direction, and same channel
- transfer shapes are regular enough to be expressed as one affine family

This is still an objectfifo at the IR surface, but it should stop lowering as a
deep shim-side FIFO with BD pressure proportional to drain tap count.

### Phase 2: Compact the runtime shim-drain task family

For the eligible case above, compact repeated shim-drain tasks into a bounded
number of reusable runtime task configurations.

There are two target forms.

1. Preferred form: one task family with `repeat_count`.

   Use one `aiex.dma_configure_task` per contiguous drain family when:

   - transfer size is constant
   - base address progression is affine
   - the repeated drain order matches the runtime sequence order

   In this form, the compiler should emit one BD chain and drive the full drain
   through `repeat_count` rather than one BD allocation per tap.

2. Fallback form: small fixed-size task ring.

   When one `repeat_count` is not sufficient, lower to a bounded ring of tasks,
   for example `K = 2` or `K = 4`, and insert:

   - `aiex.dma_start_task`
   - `aiex.dma_await_task`
   - `aiex.dma_free_task`

   so that `AIEAssignRuntimeSequenceBDIDs` can recycle BD IDs across batches.

The key property is that live shim-drain BD demand becomes `O(1)` or
`O(batch_size)`, not `O(number_of_output_taps)`.

### Phase 3: Keep the fix aligned with the existing passes

The compact drain lowering should integrate with, not replace, the existing
pipeline.

Recommended placement:

1. `aie-objectFifo-stateful-transform`
2. `aie-substitute-shim-dma-allocations`
3. new pass: `aiex-compact-shim-drain-tasks`
4. `aie-assign-runtime-sequence-bd-ids`
5. `aie-dma-tasks-to-npu`

Responsibilities:

- `aie-objectFifo-stateful-transform` still owns objectfifo semantics
- `aie-substitute-shim-dma-allocations` still resolves shim symbols to concrete
  tile/channel assignments
- `aiex-compact-shim-drain-tasks` groups repeated drain tasks and inserts the
  right `await` / `free` boundaries
- `aie-assign-runtime-sequence-bd-ids` then reuses BD IDs over the compacted
  live task set

### Phase 4: Add a dedicated op only if compaction is not enough

If the compacted objectfifo-based approach still cannot express the needed
drain pattern cleanly, introduce a dedicated runtime-facing drain op rather
than continuing to overload shim objectfifo lowering.

Possible direction:

```mlir
aiex.shim_drain @ln2_out(
  %producer_tile,
  %shim_tile
) {
  channel = 3 : i32,
  task_ring = 2 : i32
} : memref<32x64xbf16>
```

That op would mean:

- drain a regular stream family from one producer path to one shim channel
- use a bounded task ring or repeat-driven lowering
- expose the host/NPU drain behavior directly instead of inferring it from a
  generic objectfifo

This should be a fallback plan, not the first implementation step. The first
attempt should reuse the existing objectfifo + runtime-task infrastructure.

## Validation Plan

### Compiler tests

Add:

- a reduced `assign-bd-ids` style repro that currently exhausts shim BDs
- a task-compaction test showing repeated shim drains collapse to one task with
  `repeat_count`, or a small fixed task ring
- a runtime-sequence BD-reuse test proving that `dma_await_task` /
  `dma_free_task` boundaries actually recycle BD IDs on the shim tile

### Integration tests

Use the frozen encoder repros as end-to-end regressions:

- smallest `1ph / 1pf / 12pacc` case
- grouped `4ph / 4pf / 12pacc / 4opg` case

Success criteria:

- the row-store-converted design gets past the current shim-drain BD boundary
- the failure, if any remains, moves to a different resource class
- changing `dimensionsToStream` on the final drain does not materially change
  the new result when drain tap count is held fixed

### Runtime tests

If the compacted drain reaches board execution, add one NPU runtime test that:

- drains a regular memtile-to-shim output in multiple repeated taps
- checks output order and data correctness
- uses enough taps to have exhausted shim BDs before compaction

## Expected Outcome

If this plan works, the effect should be:

- row-store fixes continue to handle same-tile replay pressure
- the final LN2 output drain stops scaling shim BD demand with output tap count
- high-`part_count` encoder topologies stop failing on the remaining shim-drain
  allocator boundary

The intended result is not "more total shim resources." The intended result is
"a more appropriate lowering for regular high-tap shim drains."
