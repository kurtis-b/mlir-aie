# MemTile Row-Store Lowering

## Current Status

The compiler support described here is implemented in-tree and covered by:

- dialect verification tests
- lowering and BD-allocation tests
- direct CDO / AIERT regression coverage for partial BD lock pairs
- an `npu2` roundtrip runtime test that checks output data correctness
- an explicit `buffer_count = 2` mode with a fixed two-bank ping-pong memtile
  lowering
- a second `npu2` roundtrip test that exercises the overlap pattern the
  single-row form cannot support safely

Today the implementation is split by mode:

- `buffer_count = 1` keeps the original per-slice memtile BD lowering
- `buffer_count = 2` uses the compressed-bank memtile lowering described later
  in this note

## Goal

Add a first-class AIE lowering for a single-stream memtile row-store pattern:

- one compute-tile producer stream into a memtile row store
- one compute-tile consumer stream out of the same memtile row store
- one compute-visible producer scratch buffer
- one compute-visible consumer scratch buffer
- memtile-local aggregation over `part_count` contiguous slices
- no explosion of `aie.objectfifo` endpoints on the compute tile

This is the "larger option" compared to improving `aie.objectfifo` channel
assignment. The intent is to make this construction expressible in the compiler
IR instead of forcing it through ad hoc client-side IR generation.

## Example Use Case

Consider a two-phase kernel running on one compute tile:

- phase 1 consumes a logical row in `part_count` smaller tiles
- phase 1 must preserve the entire logical row for later reuse
- phase 2 rereads the same logical row, again one tile at a time

For example, if the compute-visible tile type is `memref<32x96xbf16>` and
`part_count = 4`, then one logical row is made of four such tiles. A natural
execution pattern is:

1. The compute tile produces four `32x96` tiles.
2. A memtile aggregates them into one contiguous row store.
3. The same compute tile later rereads the four slices in order.

What this change is trying to address is the gap between that simple execution
model and the abstractions that exist today. Modeling the same behavior with a
network of standard objectfifos tends to make the number of FIFO endpoints,
locks, DMA channels, and BDs scale with the number of row parts. The row-store
abstraction keeps the compute side constant:

- one outgoing producer stream
- one incoming consumer stream
- one producer scratch buffer
- one consumer scratch buffer

while moving the row assembly and replay state into the memtile-local storage
that actually owns the persistent row.

## Current Semantics And Limitation

The implemented lowering is a single-live-row abstraction.

- ingress cannot start writing the next logical row until the current row has
  been fully drained by egress
- egress cannot start until ingress has finished assembling the current row
- on the memtile, the row-level empty/full semaphore pair therefore protects
  exactly one logical row at a time

This is sufficient for patterns of the form:

1. fill one full row
2. consume that same row
3. fill the next row

It is not sufficient for overlapped steady-state schedules of the form:

1. consume row `N`
2. while row `N` is still draining, start producing row `N + 1` into the same
   logical store

That overlap requires either:

- a pair of separate row stores used in ping-pong fashion
- a conventional forwarded FIFO / replay structure
- a future explicit double-buffered row-store abstraction in the compiler

This limitation is semantic, not just an implementation quirk in the current
lowering. The v1 op should be read as "one logical row is live at a time."

## Recommended Fix Plan

The fix should happen in phases.

### Phase 0: Keep overlapping clients on the fallback design

For clients whose steady-state schedule is:

1. consume row `N`
2. produce row `N + 1`
3. continue without waiting for row `N` to fully drain

the correct short-term action is to keep using:

- a forwarded replay FIFO, or
- two explicit row stores used in ping-pong fashion at the client level

Do not route that schedule through the current single-row-store lowering. The
compiler/backend fixes already landed here are necessary, but they do not
change the single-live-row contract.

### Phase 1: Explicit double-buffered row-store mode

This phase is implemented in-tree.

The compiler-side fix should be an explicit IR extension, not a silent change
to the semantics of the current op. The recommended form is:

```mlir
aie.memtile_row_store @row_store0(
  %compute_tile,
  %mem_tile
) {
  part_count = 8 : i32,
  buffer_count = 2 : i32
} : memref<32x96xbf16>
```

with:

- `buffer_count = 1` meaning today's semantics
- `buffer_count = 2` meaning the store may hold one row being drained and one
  newer row being filled at the same time

The core-side acquire/release ops do not need a new surface syntax in v2. The
banking remains an implementation detail of the lowering. What changes is the
resource contract and the memtile-side schedule.

### Phase 2a: Lower `buffer_count = 2` as fixed ping-pong banks

This phase is implemented in-tree.

The simplest correct lowering is a fixed two-bank alternation on the memtile.

Materialize on the memtile:

- `%S_row_0`
- `%S_row_1`
- `%S_row_0_empty`, `%S_row_0_full`
- `%S_row_1_empty`, `%S_row_1_full`

with initial state:

- both `row_*_empty = 1`
- both `row_*_full = 0`

The compute side stays unchanged:

- one producer scratch buffer
- one consumer scratch buffer
- one compute MM2S chain
- one compute S2MM chain

Ingress on the memtile alternates banks in fixed order:

1. wait on `row_0_empty`
2. fill all `part_count` slices of `row_0`
3. release `row_0_full`
4. wait on `row_1_empty`
5. fill all `part_count` slices of `row_1`
6. release `row_1_full`
7. loop back to bank 0

Egress alternates in the same order:

1. wait on `row_0_full`
2. drain all `part_count` slices of `row_0`
3. release `row_0_empty`
4. wait on `row_1_full`
5. drain all `part_count` slices of `row_1`
6. release `row_1_empty`
7. loop back to bank 0

This gives the intended steady state:

- ingress can fill bank 1 while egress drains bank 0
- ingress can later refill bank 0 after egress releases it
- no hidden inference is needed from core-side acquire order

This is the simplest correctness-first implementation. It proves the overlap
semantics, but in its straightforward form it still scales memtile DMA blocks
with `part_count`.

### Phase 2b: Compress the memtile BD shape for deployment use

If the goal is only correctness, Phase 2a is enough. If the goal is also to
support larger `part_count` values or avoid pushing replay traffic back through
DDR, the lowering must go one step further and compress the memtile-side BD
shape.

The key observation is that many row-store clients do not need random access on
the memtile. They need:

- in-order assembly of a full logical row
- in-order replay of that same row
- stable compute-side scratch buffers

For that access pattern, the memtile side does not inherently require one BD
per row slice. A deployment-oriented v2 lowering should try to use, per bank:

- one ingress BD that writes the full logical row buffer in order
- one egress BD that reads the full logical row buffer in order

instead of enumerating every slice in separate BD blocks.

This changes the scaling behavior:

- Phase 2a block usage is `O(part_count)`
- compressed-bank lowering is `O(buffer_count)`

In practical terms, the straightforward double-buffered CFG currently materializes
`4 * part_count` BD blocks in the memtile DMA body. Under the current resource
model, `part_count = 12` is the largest value that still fits, while
`part_count = 13` is rejected with:

```text
'aie.memtile_dma' op has more than 48 blocks
```

A compressed-bank lowering should reduce that to a small constant number of
bank blocks, so `part_count` stops being the dominant term in the memtile block
budget.

This does not require the compute side to stop operating one tile at a time.
The intended execution model is:

- compute MM2S still sends one tile-sized producer scratch buffer at a time
- compute S2MM still receives one tile-sized consumer scratch buffer at a time
- the memtile ingress/egress BD spans the whole logical row for a bank
- stream backpressure stitches the tile-sized compute transfers into one
  row-sized memtile transfer

In other words, the memtile sees one long row transfer, while the compute tile
still sees the existing per-tile acquire/release rhythm.

#### Compressed-bank lowering plan

The intended implementation should follow these rules.

Legality:

- use compressed-bank lowering only when the logical row layout on the memtile
  is contiguous and replay order matches fill order
- use it only when the compute side produces and consumes row slices in-order
- fall back to the per-slice BD lowering if the client requests any
  permutation, strided replay order, or other non-contiguous memtile layout

Lowering shape for `buffer_count = 2`:

- materialize two contiguous row buffers, one per bank
- keep the existing compute-side scratch buffers and compute DMA channels
- build one memtile ingress block per bank:
  - acquire `row_i_empty`
  - `aie.dma_bd(%row_i, 0, full_row_len)`
  - release `row_i_full`
- build one memtile egress block per bank:
  - acquire `row_i_full`
  - `aie.dma_bd(%row_i, 0, full_row_len)`
  - release `row_i_empty`
- alternate bank `0 -> 1 -> 0` on both ingress and egress

Expected memtile block count:

- one S2MM `dma_start` block
- two ingress bank blocks
- one MM2S `dma_start` block
- two egress bank blocks
- one shared `aie.end` block

So the compressed double-buffered memtile DMA region should use a constant
seven blocks instead of `4 * part_count + 3`.

Why this is plausible for `encoder_pipeline`:

- the current failing LN1 repro already lowers the memtile row as one flat
  contiguous buffer
- ingress BDs write offsets `0, 3072, 6144, ...`
- egress BDs read the same offsets in the same order
- that is exactly the pattern that can be collapsed into one full-row BD per
  bank

In the concrete frozen repro:

- tile size is `32 x 96 = 3072` bf16 elements
- `part_count = 8`
- one full row is `8 * 3072 = 24576` bf16 elements

So the current eight-BD ingress chain and eight-BD egress chain are strong
evidence that a single `len = 24576` BD per bank is the right deployment
target, provided the runtime test confirms the expected stream backpressure
behavior.

#### Remaining proof obligations for compressed-bank lowering

Before treating the compressed-bank path as deployment-ready, the implementation
must make the following points explicit and test them directly.

1. Stream continuity across tile-sized compute DMAs.

   The compressed-bank plan assumes one memtile full-row BD can consume or
   produce a stream assembled from multiple tile-sized compute DMA transfers on
   the same channel. This is the key runtime assumption behind "one long row
   transfer on the memtile, one tile at a time on the compute tile."

   If that assumption does not hold on the target/runtime path, the fallback
   plan should still keep bank count constant while using a small fixed chain
   per bank instead of one BD per row slice.

2. Precise legality conditions.

   The compressed path should be selected only when all of the following are
   true:

   - all row slices have the same tile shape
   - memtile row layout is contiguous
   - fill order matches replay order
   - no per-slice permutation or gather/scatter semantics are required
   - no extra `dimensions`/padding transforms are needed on the memtile side

3. Acceptance criteria for `encoder_pipeline`-style clients.

   For a client whose row bytes stay fixed while `part_count` increases, the
   compressed-bank path should demonstrate all of:

   - overlap correctness for "consume row `N` while producing row `N + 1`"
   - no linear growth in memtile DMA block count with `part_count`
   - replay remains on-chip in the memtile rather than forcing extra DDR
     rereads for layer-norm reuse

4. Performance validation.

   Correctness tests are necessary but not sufficient. A client-level check
   should compare the compressed-bank row-store path against the current design
   on:

   - total off-chip bytes moved
   - whether finer tiling can be used at the same row footprint
   - runtime or throughput for the replay-heavy stage

When this compression is legal, it is the part of the plan that addresses
bandwidth-sensitive row-replay clients: replay stays on-chip in the memtile and
the compiler no longer forces the client to widen or reshape DDR traffic just
to stay under memtile BD-block limits.

### Phase 3: Validate overlap before migrating clients back

After the compiler work lands, validate it in this order:

1. IR/parser/verifier coverage for `buffer_count = 2`
2. lowering/FileCheck coverage that proves two-bank alternation
3. BD/block-budget tests for the new CFG shape
4. a runtime roundtrip that overlaps produce and consume on one logical store
5. only then migrate clients that currently use the FIFO fallback

This keeps the client rollback simple if the v2 lowering still hits an
unexpected channel, block, or runtime issue.

## Why This Should Not Be Another `aie.objectfifo` Mode

The existing `aie.objectfifo` lowering is built around queue semantics on FIFO
endpoints. The row-store construction is different:

- the persistent state is a memtile-local row buffer, not a queue of independent
  subtile objects
- the core interacts with two stable scratch buffers (`src`, `dst`) and lock
  phases, not with a growing set of FIFO endpoints
- `aie.objectfifo.link` currently forbids the exact compositions that show up in
  this kind of design:
  - one objectfifo cannot appear in more than one link op
  - join and distribute links cannot be directly accessed from core code
- pushing the current design through objectfifos adds compute-tile locks and DMA
  endpoints, which is precisely the resource spike we are trying to avoid

For this reason the new functionality should be modeled as a dedicated AIE
abstraction with its own lowering pass.

## IR Additions

### 1. Device-scope declaration

Add a new symbol op to the AIE dialect:

```mlir
aie.memtile_row_store @row_store0(
  %compute_tile,
  %mem_tile
) {
  part_count = 4 : i32,
  compute_mm2s_channel = 0 : i32,
  compute_s2mm_channel = 0 : i32,
  memtile_ingress_channel = 0 : i32,
  memtile_egress_channel = 1 : i32
} : memref<32x96xbf16>
```

Suggested TD shape:

```tablegen
def AIE_MemTileRowStoreOp : AIE_Op<"memtile_row_store",
    [HasParent<"DeviceOp">, Symbol]> {
  let arguments = (
    ins SymbolNameAttr:$sym_name,
        Index:$computeTile,
        Index:$memTile,
        I32Attr:$part_count,
        DefaultValuedAttr<AIEI32Attr, "0">:$compute_mm2s_channel,
        DefaultValuedAttr<AIEI32Attr, "0">:$compute_s2mm_channel,
        DefaultValuedAttr<AIEI32Attr, "0">:$memtile_ingress_channel,
        DefaultValuedAttr<AIEI32Attr, "1">:$memtile_egress_channel,
        TypeAttrOf<AnyMemRef>:$elemType
  );
  let hasVerifier = 1;
}
```

Semantics:

- `elemType` is the compute-visible tile type, for example
  `memref<32x96xbf16>`
- `part_count` is the number of such tiles that make one full replay row
- the memtile-local storage buffer is implied and is lowered as one contiguous
  `part_count * num_elements(elemType)` backing buffer
- the default memtile channels intentionally split ingress and egress across
  channel parities (`0` and `1`) so the two DMA directions allocate from
  different memtile BD banks
- the row-store owns those DMA channels for the duration of the lowered design;
  the lowering should reject conflicts with pre-existing DMA starts on the same
  tile, direction, and channel
- whether the implied storage fits in memtile memory is not checked by this op's
  verifier; that remains the responsibility of the existing buffer allocation /
  address assignment pipeline

### 2. Core-side acquire op

Add:

```mlir
%buf = aie.memtile_row_store.acquire @row_store0(Produce)
  : memref<32x96xbf16>
%buf2 = aie.memtile_row_store.acquire @row_store0(Consume)
  : memref<32x96xbf16>
```

Suggested TD shape:

```tablegen
def AIE_MemTileRowStoreAcquireOp : AIE_Op<"memtile_row_store.acquire", []> {
  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$store_name
  );
  let results = (outs AnyMemRef:$buffer);
  let hasVerifier = 1;
}
```

Reuse `ObjectFifoPort` so the parser and verifier can use `Produce` / `Consume`
instead of inventing a new enum.

### 3. Core-side release op

Add:

```mlir
aie.memtile_row_store.release @row_store0(Produce)
aie.memtile_row_store.release @row_store0(Consume)
```

Suggested TD shape:

```tablegen
def AIE_MemTileRowStoreReleaseOp : AIE_Op<"memtile_row_store.release", []> {
  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$store_name
  );
  let hasVerifier = 1;
}
```

## Verifier Rules

Implement in `lib/Dialect/AIE/IR/AIEDialect.cpp`.

### `aie.memtile_row_store`

- `computeTile` must be a core tile
- `memTile` must be a memtile
- `elemType` must be a statically shaped `memref`
- `part_count >= 1`
- `compute_mm2s_channel` must be a valid MM2S DMA channel index for the compute
  tile
- `compute_s2mm_channel` must be a valid S2MM DMA channel index for the compute
  tile
- `memtile_ingress_channel` must be a valid S2MM DMA channel index for the
  memtile
- `memtile_egress_channel` must be a valid MM2S DMA channel index for the
  memtile
- the byte size of `elemType` must be a multiple of 4 so the implied DMA
  transfer length and slice offsets remain 32-bit aligned

### `aie.memtile_row_store.acquire`

- referenced symbol must resolve to `aie.memtile_row_store`
- result type must match the store `elemType`
- op must appear inside a `aie.core`
- the enclosing core tile must be the row store's `computeTile`

### `aie.memtile_row_store.release`

- referenced symbol must resolve to `aie.memtile_row_store`
- op must appear inside a `aie.core`
- the enclosing core tile must be the row store's `computeTile`

The verifier should stay structural. Do not try to prove global acquire/release
balance in SSA form in v1.

## New Lowering Pass

Add a new pass:

- pass name: `aie-lower-memtile-row-stores`
- type: `DeviceOp`
- file: `lib/Dialect/AIE/Transforms/AIELowerMemtileRowStores.cpp`

Add the declaration in:

- `include/aie/Dialect/AIE/Transforms/AIEPasses.td`
- `include/aie/Dialect/AIE/Transforms/AIEPasses.h`

Register it in:

- `lib/Dialect/AIE/Transforms/CMakeLists.txt`

### Pipeline placement

Run it before `aie-objectFifo-stateful-transform`.

Recommended order in a typical pipeline:

```text
aie-register-objectFifos
aie-lower-memtile-row-stores
aie-objectFifo-stateful-transform
aie-assign-bd-ids
...
```

This ordering matters because `AIEObjectFifoStatefulTransform` already scans
existing `aie.mem`, `aie.memtile_dma`, and `aie.flow` ops when choosing DMA
channels. Lowering row stores first ensures later objectfifo lowering accounts
for the channels already consumed by the row store.

## Lowering Algorithm

For each `aie.memtile_row_store @S`:

### 1. Materialize compute-side resources

Create on `computeTile`:

- `%S_src = aie.buffer(...) : elemType`
- `%S_dst = aie.buffer(...) : elemType`
- `%S_src_empty = aie.lock(...) {init = 1}`
- `%S_src_full = aie.lock(...) {init = 0}`
- `%S_dst_empty = aie.lock(...) {init = 1}`
- `%S_dst_full = aie.lock(...) {init = 0}`

This intentionally uses empty/full semaphore pairs rather than a single binary
state lock, because the memtile row-store lowering targets AIE2-class semaphore
semantics.

### 2. Materialize memtile-side resources

Create on `memTile`:

- `%S_row = aie.buffer(...) : memref<part_count x flat(elemType)>`
  - or an equivalent statically shaped contiguous memref
- one row-level empty/full semaphore pair:
  - `%S_row_empty` with `init = 1`
  - `%S_row_full` with `init = 0`

### 3. Materialize flows

Create:

```mlir
aie.flow(%computeTile, DMA : compute_mm2s_channel,
         %memTile, DMA : memtile_ingress_channel)
aie.flow(%memTile, DMA : memtile_egress_channel,
         %computeTile, DMA : compute_s2mm_channel)
```

### 4. Materialize compute-tile DMA

Create one `aie.mem(%computeTile)` region with two self-looping DMA chains.

#### Compute MM2S chain

- waits on `%S_src_full >= 1`
- transfers `%S_src`
- releases `%S_src_empty += 1`
- self-loops forever

#### Compute S2MM chain

- waits on `%S_dst_empty >= 1`
- fills `%S_dst`
- releases `%S_dst_full += 1`
- self-loops forever

This gives a stable producer/consumer handshake for the compute-visible scratch
buffers.

### 5. Materialize memtile DMA

Create one `aie.memtile_dma(%memTile)` region with:

#### Ingress chain on `memtile_ingress_channel`

- `part_count` BDs
- BD `i` writes into row slice `i`
- first ingress BD additionally:
  - waits on `%S_row_empty >= 1`
- each BD DMA writes slice `i`
- the final ingress BD additionally releases `%S_row_full += 1`

#### Egress chain on `memtile_egress_channel`

- separate `dma_start` block, not merged into the ingress chain
- `part_count` BDs
- first egress BD waits on `%S_row_full >= 1`
- BD `i` DMA reads slice `i`
- final egress BD additionally releases `%S_row_empty += 1`

The dedicated egress start block is important. Keeping the output `dma_start` in
its own chain block avoids an awkward control-flow shape and makes the DMA
region easier to reason about.

### 5a. Backend compatibility requirement

The ingress and egress chains intentionally use one-sided lock operations on
their boundary BDs:

- the first ingress BD carries the row-empty acquire
- the final ingress BD carries the row-full release
- the first egress BD carries the row-full acquire
- the final egress BD carries the row-empty release

That means direct CDO / AIERT lowering must accept BD blocks that contain only
an acquire or only a release lock. It must not assume that every BD block has a
full acquire+release pair. This is now covered by a dedicated direct-CDO
regression test:

- `test/Targets/AIETargetCDODirect/partial_bd_locks.mlir`

### 6. Rewrite core-side acquire/release ops

Inside the core bound to `computeTile`:

- `aie.memtile_row_store.acquire @S(Produce)` lowers to:
  - `aie.use_lock(%S_src_empty, AcquireGreaterEqual, 1)`
  - replace result with `%S_src`
- `aie.memtile_row_store.release @S(Produce)` lowers to:
  - `aie.use_lock(%S_src_full, Release, 1)`
- `aie.memtile_row_store.acquire @S(Consume)` lowers to:
  - `aie.use_lock(%S_dst_full, AcquireGreaterEqual, 1)`
  - replace result with `%S_dst`
- `aie.memtile_row_store.release @S(Consume)` lowers to:
  - `aie.use_lock(%S_dst_empty, Release, 1)`

After rewriting all uses, erase the acquire/release ops and the original
`aie.memtile_row_store` declaration.

## Long-Term Reuse: Explicit Double Buffering

Why `buffer_count = 2` should stay explicit:

- it changes the resource contract from one live row to two live rows
- it doubles the memtile row-storage footprint
- it adds extra row-bank synchronization state and DMA scheduling constraints
- clients that are correct today with `buffer_count = 1` should keep the same
  semantics and resource usage

For long-term reuse, this gives clients a clear choice between:

- lower resource usage with `buffer_count = 1`
- overlap-friendly semantics with `buffer_count = 2`

The second independent choice is lowering shape:

- correctness-first `buffer_count = 2` with one BD per slice
- deployment-ready `buffer_count = 2` with compressed memtile bank BDs

Both choices should remain explicit in the implementation plan, because only
the second one materially improves the usable `part_count` ceiling.

## Files To Touch

Minimum implementation surface:

- `include/aie/Dialect/AIE/IR/AIEOps.td`
- `lib/Dialect/AIE/IR/AIEDialect.cpp`
- `include/aie/Dialect/AIE/Transforms/AIEPasses.td`
- `include/aie/Dialect/AIE/Transforms/AIEPasses.h`
- `lib/Dialect/AIE/Transforms/AIELowerMemtileRowStores.cpp` (new)
- `lib/Dialect/AIE/Transforms/CMakeLists.txt`

No changes are required in `aie-assign-bd-ids` if the lowering emits standard
`aie.dma_bd` ops. The existing pass will allocate BD IDs correctly.

For the double-buffered extension, the same files remain the main
implementation surface. The implemented incremental changes are:

- add `buffer_count` to `AIE_MemTileRowStoreOp`
- extend verification to accept only the initial supported set, ideally
  `{1, 2}`
- extend `AIELowerMemtileRowStores.cpp` with the two-bank lowering path
- use the compressed-bank lowering for the currently supported
  `buffer_count = 2` path, whose access pattern is always contiguous and
  in-order
- add overlap-specific lowering, BD, and runtime tests

## Why The Default Memtile Channels Should Be `0` / `1`

The current target model partitions memtile BD accessibility by channel index
parity. If ingress and egress both use channel `0`, they share the same 24
accessible BD IDs. If ingress uses `0` and egress uses `1`, the row-store uses
both BD banks:

- ingress BDs allocate from `0..23`
- egress BDs allocate from `24..47`

That makes the first-class compiler lowering materially better than a naive
single-channel lowering, while keeping the same single producer / single
consumer stream shape.

## Test Plan

The implementation should be covered in six layers, with the last one
remaining optional.

### 1. Dialect verification tests

Add:

- `test/dialect/AIE/memtile_row_store.mlir`
- `test/dialect/AIE/memtile_row_store_bad.mlir`

Coverage:

- valid parse/print round-trip of the three new ops
- invalid `memTile` operand is not a memtile
- invalid `computeTile` operand is not a core tile
- invalid `part_count = 0`
- acquire/release outside the bound core tile
- acquire result type mismatch

Representative RUN lines:

```mlir
// RUN: aie-opt %s | FileCheck %s
// RUN: not aie-opt --verify-diagnostics %s 2>&1 | FileCheck %s --check-prefix=ERR
```

### 2. Lowering tests for the new pass

Add:

- `test/Passes/lower-memtile-row-stores/basic.mlir`
- `test/Passes/lower-memtile-row-stores/with-objectfifo.mlir`
- `test/Passes/lower-memtile-row-stores/channel_conflict.mlir`
- `test/Passes/lower-memtile-row-stores/encoder_pipeline_o_proj_like.mlir`
- `test/Passes/lower-memtile-row-stores/encoder_pipeline_o_proj_like_finer_tiles.mlir`

`basic.mlir` should check:

- the row-store declaration is gone
- the core-side acquire/release ops are gone
- two compute buffers and four compute locks are created
- one memtile row buffer and one row-level empty/full pair are created
- one `aie.flow` in each direction is created
- compute tile gets exactly one MM2S and one S2MM DMA start
- memtile gets distinct ingress and egress channel indices by default
- memtile ingress offsets are contiguous:
  - `0`
  - `elem_len`
  - `2 * elem_len`
  - ...
- the first ingress path waits on `row_empty >= 1`
- the final ingress path releases `row_full += 1`
- the first egress path waits on `row_full >= 1`
- the final egress path releases `row_empty += 1`
- the egress `dma_start` is emitted in its own block

`with-objectfifo.mlir` should run:

```mlir
// RUN: aie-opt --aie-lower-memtile-row-stores --aie-objectFifo-stateful-transform %s | FileCheck %s
```

and check that the later objectfifo lowering allocates channels around the
already-materialized row-store DMAs instead of colliding with them.

`channel_conflict.mlir` should verify that the lowering rejects DMA-channel
reuse on the same tile, direction, and channel.

The two encoder-shaped tests should verify that the lowering remains stable for
realistic NPU2 placements and for a finer-grained tiling that keeps the total
memtile row footprint constant while increasing `part_count`.

### 3. BD allocation tests

Add:

- `test/Passes/assign-bd-ids/memtile_row_store_basic.mlir`
- `test/Passes/assign-bd-ids/memtile_row_store_exhausted.mlir`
- `test/Passes/assign-bd-ids/memtile_row_store_encoder_pipeline_o_proj_like.mlir`
- `test/Passes/assign-bd-ids/memtile_row_store_encoder_pipeline_o_proj_like_finer_tiles.mlir`

`memtile_row_store_basic.mlir` should run:

```mlir
// RUN: aie-opt --aie-lower-memtile-row-stores --aie-assign-bd-ids %s | FileCheck %s
```

and check that:

- ingress BDs on memtile channel `0` get low-bank IDs
- egress BDs on memtile channel `1` get high-bank IDs
- compute BDs get normal tile-local IDs

`memtile_row_store_exhausted.mlir` should choose a large enough `part_count`
that the memtile DMA region runs out of blocks before BD IDs run out. In the
current lowering shape, this happens before channel-0 BD exhaustion:

- the memtile DMA region uses `2 * part_count + 3` blocks
- with the current 48-block limit, `part_count = 22` still fits, while
  `part_count = 23` is the first value that exceeds the region budget

So the test should verify the actual failure mode from the current resource
model, not an older expected BD-allocation failure. In practice the diagnostic
is:

```text
'aie.memtile_dma' op has more than 48 blocks in its body
```

This ensures the new lowering is accounted for correctly by the current region
and BD resource model.

The double-buffered extension now has:

- `test/Passes/lower-memtile-row-stores/double_buffered.mlir`
- `test/Passes/assign-bd-ids/memtile_row_store_double_buffered.mlir`
- `test/Passes/lower-memtile-row-stores/double_buffered_finer_tiles.mlir`
- `test/Passes/assign-bd-ids/memtile_row_store_double_buffered_finer_tiles.mlir`

Coverage should include:

- two memtile row buffers and two row-level empty/full pairs
- ingress bank order `0 -> 1 -> 0`
- egress bank order `0 -> 1 -> 0`
- unchanged compute-side scratch-buffer contract
- one ingress BD per bank instead of one per row slice
- one egress BD per bank instead of one per row slice
- memtile block usage no longer scales linearly with `part_count`
- large `part_count` values with the same total row footprint still lower
  successfully

### 4. Direct target/backend coverage

Add:

- `test/Targets/AIETargetCDODirect/partial_bd_locks.mlir`

Coverage:

- direct CDO generation succeeds when a BD block has only an acquire lock
- direct CDO generation succeeds when a BD block has only a release lock
- this matches the boundary-BD pattern emitted by the memtile row-store
  lowering

### 5. Runtime roundtrip test

Add:

- `test/npu-xrt/memtile_row_store_roundtrip/run.lit`
- `test/npu-xrt/memtile_row_store_roundtrip/aie.mlir`
- `test/npu-xrt/memtile_row_store_roundtrip/test.cpp`

Pipeline:

```mlir
// RUN: aie-opt --aie-lower-memtile-row-stores --aie-assign-lock-ids --aie-assign-bd-ids %s > %t
// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host %t
```

Coverage:

- host/NPU -> compute tile -> memtile row store -> compute tile -> host/NPU
- actual data correctness, not just structural lowering
- a concrete `npu2` roundtrip that checks output equals `input + 101`

For the double-buffered extension, the second runtime test now exists and
proves the overlap case instead of only the fill-then-drain case. Its kernel
structure is:

1. produce one full row
2. start consuming that row
3. before the consume phase completes, start producing the next row
4. verify the final output still preserves row order and data integrity

That test is the key proof that the v2 semantics actually solve the motivating
schedule.

The remaining optional follow-on here is a third runtime-oriented test whose
total row footprint stays constant while `part_count` increases. That test
would demonstrate the performance effect directly, not just the structural
lowering behavior.

### 6. Optional NPU lowering smoke test

If we want one more IR-only confidence test, add:

- `test/Conversion/DmaToNpu/memtile_row_store.mlir`

Pipeline:

```mlir
// RUN: aie-opt --aie-lower-memtile-row-stores --aie-assign-lock-ids --aie-assign-bd-ids --aie-dma-to-npu %s | FileCheck %s
```

Check only for:

- expected `aiex.npu.writebd`
- expected `aiex.npu.push_queue`
- the memtile ingress and egress channel numbers that came from the row-store
  lowering

## Suggested First Implementation Scope

Keep v1 intentionally narrow:

- one compute tile
- one memtile
- one producer scratch buffer
- one consumer scratch buffer
- one row buffer
- `part_count >= 1`
- no double-buffered compute scratch in v1
- no multi-consumer broadcast semantics
- no sharing of the selected DMA channels with pre-existing DMA starts on the
  same tile and direction

That scope is enough to cover common single-producer / single-consumer row-store
use cases.

Suggested v2 scope should also stay narrow:

- still one compute tile and one memtile
- still one producer scratch buffer and one consumer scratch buffer
- support only `buffer_count in {1, 2}`
- implement `buffer_count = 2` only as fixed two-bank alternation
- no dynamic bank selection
- no multi-consumer broadcast semantics
- keep the compressed-bank lowering restricted to the explicit two-bank mode

## Expected Outcome

This patch set is aimed at the compiler-side representation problem:

- the row-store becomes a compiler-level construct
- the generated design uses a constant number of compute-side endpoints
- for `buffer_count = 1`, memtile BD usage still scales with `2 * part_count`
- for `buffer_count = 2`, memtile BD usage now tracks row banks rather than
  row slices
- the lowering is testable with existing `aie-opt` and FileCheck coverage

For the implemented v2 extension, the expected outcome is:

- overlapping consume/write schedules become representable without changing the
  core-side programming model
- clients with FIFO fallbacks can migrate back selectively where the doubled
  memtile storage fits
- the extra cost is explicit: roughly `2x` memtile row storage
- the memtile DMA CFG for `buffer_count = 2` now tracks row banks rather than
  row slices

- `part_count` is no longer the main limiter when total row bytes stay fixed
- clients can keep replay traffic on-chip instead of reshaping the design
  around avoidable DDR rereads
