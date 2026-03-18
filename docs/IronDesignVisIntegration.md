# IRON Design-Vis Migration Plan

## Goal

Make the design-visualization tool an IRON-owned user experience.

After migration, the normal workflow should start and end in
[/home/agi-demo/iron](/home/agi-demo/iron):

```bash
python scripts/design_vis.py
```

or:

```bash
python scripts/design_vis.py --operator encoder_pipeline
python scripts/design_vis.py --operator encoder_pipeline_ddr --case full_pipeline_seqpar_base
python scripts/design_vis.py --directory build/design_vis/encoder_pipeline_ddr/full_pipeline_seqpar_base
python scripts/design_vis.py --file build/some_design.mlir.prj/input_with_addresses.mlir
```

The user should not need to know where `mlir-aie` is checked out, how to run
`aie-opt`, how to generate JSON by hand, or how to manually find the right MLIR
artifact after a case is generated.

## Ownership Decision

The tool should be owned by IRON.

That means:

- IRON owns the user-facing command
- IRON owns the browser UI files
- IRON owns operator discovery
- IRON owns case discovery and case generation
- IRON owns design discovery under IRON build outputs
- IRON owns generated-design directory management
- IRON owns the README/user documentation for the workflow

`mlir-aie` should only retain the pieces that fundamentally belong there:

- the `aie-translate --aie-design-to-json` exporter
- a wheel-installed Python helper/API needed to invoke the exporter
- those pieces must be obtainable through a rebuilt `mlir-aie` wheel

So the long-term split becomes:

- `iron`: product/tool owner
- `mlir-aie`: wheel-provided compiler/backend dependency

## Why This Is The Right Split

The workflow starts in IRON:

- IRON users build operators from `design.py` and `test.py`
- IRON users choose between operators and operator-specific cases
- IRON users need the tool to generate MLIR for those cases, not just browse
  already-known artifacts
- IRON users care about IRON build outputs and operator names
- IRON already contains operator-local README flows that point users to
  visualization tooling by hand, for example
  [/home/agi-demo/iron/operators/ffn/README.md](/home/agi-demo/iron/operators/ffn/README.md)

The exporter starts in `mlir-aie`:

- `aie-translate --aie-design-to-json` is a compiler/export feature
- the route/materialization logic belongs next to the AIE dialect and targets

That means the clean architecture is:

- IRON drives the workflow
- `mlir-aie` ships the exporter/backend through its wheel

## Target User Experience

### Default

```bash
python scripts/design_vis.py
```

This should:

1. discover available operators that expose design-vis generation
2. start the IRON-owned visualization service if needed
3. open the browser
4. pre-populate the operator selector
5. if previously generated designs exist, offer the newest generated directory
   immediately

### Operator + Case Workflow

```bash
python scripts/design_vis.py --operator encoder_pipeline_ddr
```

This should:

1. preselect the operator in the UI
2. populate the list of available cases for that operator
3. allow the user to click `Generate MLIR`
4. write artifacts into a managed IRON output directory
5. automatically open that generated directory in the file selector
6. default to the most useful file inside it, typically:
   - `*.mlir.prj/input_with_addresses.mlir`
   - else `*.mlir.prj/input.mlir`
   - else another generated `.mlir`

### Explicit Case

```bash
python scripts/design_vis.py --operator encoder_pipeline_ddr --case full_pipeline_seqpar_base
```

This should:

1. skip operator browsing
2. generate that exact case
3. open the generated output directory
4. render the default MLIR artifact from that directory immediately

### Directory-focused

```bash
python scripts/design_vis.py --directory build/design_vis/encoder_pipeline_ddr/full_pipeline_seqpar_base
```

This should skip generation and open that exact generated directory first.

### Explicit-file

```bash
python scripts/design_vis.py --file build/foo.mlir.prj/input_with_addresses.mlir
```

This should skip generation and directory discovery and open that exact design
first.

## What Must Stay In `mlir-aie`

These parts should remain there and be exposed through the wheel:

### 1. C++ exporter

Keep:

- `aie-translate --aie-design-to-json`

This is already registered from:

- [/home/agi-demo/mlir-aie/lib/Targets/AIETargets.cpp](/home/agi-demo/mlir-aie/lib/Targets/AIETargets.cpp)

This stays a compiler feature.

### 2. Python render helper / API

Move the render/export logic into wheel-installed Python code, not just a
source-tree script.

The current source prototype is:

- [/home/agi-demo/mlir-aie/tools/aie-design-vis/server.py](/home/agi-demo/mlir-aie/tools/aie-design-vis/server.py)

The preferred wheel contract is a direct Python API, for example:

- `from aie.tools.design_vis import render_design_json`

The backend CLI remains optional as a fallback, but IRON should not be forced
to depend on a second standalone HTTP service if the render API can be imported
directly from the installed wheel.

The wheel may also provide:

- a Python module such as `aie.tools.design_vis_server`

or:

- a console script such as `aie-design-vis-backend`

The wheel already exposes console scripts through
[/home/agi-demo/mlir-aie/utils/mlir_aie_wheels/pyproject.toml](/home/agi-demo/mlir-aie/utils/mlir_aie_wheels/pyproject.toml),
so this should follow the same packaging model as `aie-opt`, `aie-translate`,
and `aie-visualize`.

### 3. What Must Not Stay In `mlir-aie`

The following should not remain compiler-owned:

- operator discovery
- case discovery
- case metadata
- MLIR generation for IRON test cases
- generated-directory browsing logic
- the user-facing browser UI

## What Should Move To IRON

Everything user-facing should move.

### 1. Browser UI

Move the following into IRON, likely under:

- `tools/design_vis/`

or:

- `scripts/design_vis/`

Specifically:

- the HTML/JS frontend from
  [/home/agi-demo/mlir-aie/tools/aie-design-vis/index.html](/home/agi-demo/mlir-aie/tools/aie-design-vis/index.html)
- any local README/help text for the tool
- any IRON-specific branding, defaults, and discovery affordances

The frontend should stop living in `mlir-aie` once the migration is complete.

### 2. IRON wrapper / launcher

Add a native IRON command:

- `scripts/design_vis.py`

This should:

1. resolve the IRON repo root
2. discover operators that expose design-vis case generation
3. discover cases for the selected operator
4. generate MLIR for a selected case into a managed output directory
5. locate the wheel-installed render/export tooling in the active `ironenv`
6. start the IRON-owned local server if needed
7. open the browser

### 3. Operator and case discovery

IRON should own operator and case discovery.

Recommended model:

1. each operator may expose a small design-vis adapter module
2. the launcher discovers those adapters
3. the adapter provides:
   - operator display name
   - case list
   - generation entrypoint
   - default output-file preference

Recommended adapter shape:

- `operators/<name>/design_vis.py`

with a contract such as:

```python
def list_cases() -> list[CaseSpec]: ...
def generate_case_mlir(case_id: str, output_dir: Path) -> GeneratedDesignDir: ...
```

This avoids scraping pytest internals or README text to learn what a case means.

### 4. Generated design directory management

IRON should own discovery rules because only IRON knows which generated
artifacts are most useful to its users.

Recommended priority:

1. `*.mlir.prj/input_with_addresses.mlir`
2. `*.mlir.prj/input.mlir`
3. `*.mlir`

Search roots:

- default generated root: `./build/design_vis`
- operator mode: `./build/design_vis/<operator>`
- case mode: `./build/design_vis/<operator>/<case>`
- legacy browse mode: `./build`, then operator-local build roots
- explicit file mode: exact path only

The frontend should:

1. show a directory selector after generation
2. list all generated MLIR files from that directory
3. open the preferred file by default
4. let the user switch to sibling files from the same generated directory

### 5. Documentation

Update:

- top-level IRON README
- operator READMEs that currently point at manual visualization steps

The first one to update should be:

- [/home/agi-demo/iron/operators/ffn/README.md](/home/agi-demo/iron/operators/ffn/README.md)

## Runtime Contract After Migration

To keep the IRON side simple, the installed wheel should support:

### Preferred: direct Python render API

Example:

```python
from aie.tools.design_vis import render_design_json
```

with behavior equivalent to:

```text
aie-opt --aie-create-pathfinder-flows --aie-find-flows | aie-translate --aie-design-to-json
```

and fallback to direct translation when routing is not applicable.

### Optional: wheel-provided fallback server/CLI

If IRON cannot import the render helper directly, a wheel-provided fallback is
acceptable:

- `python -m aie.tools.design_vis_server`
- or `aie-design-vis-backend`

### IRON-owned local server endpoints

The IRON-owned service should expose the user workflow:

- `GET /api/health`
- `GET /api/operators`
- `GET /api/operators/<name>/cases`
- `POST /api/generate-case`
- `GET /api/design-files`
- `POST /api/render-design`

`POST /api/render-design` should be implemented by calling the installed
`mlir-aie` wheel API, not by depending on source-tree paths.

## Migration Strategy

### Phase 1: Make render/export functionality wheel-available

In `mlir-aie`:

1. package `aie-translate --aie-design-to-json` as today
2. expose a wheel-installed Python render helper
3. keep a CLI/backend fallback only if needed
4. verify that after rebuilding and reinstalling the wheel, IRON can render
   design JSON without using a `~/mlir-aie/tools/...` source path

Success criterion:

- from `ironenv`, after reinstalling the rebuilt wheel, IRON can import or
  launch the render/export helper with no direct reference to `~/mlir-aie/tools/...`

### Phase 2: Move the frontend and service into IRON

In `iron`:

1. copy the browser UI into an IRON-owned location
2. add an IRON-owned local server that serves the frontend and exposes operator,
   case, generation, directory, and render endpoints
3. update the frontend to call that IRON service instead of assuming
   `mlir-aie/tools/aie-design-vis/server.py`

Success criterion:

- IRON serves its own frontend files
- IRON also owns the interactive service layer
- `mlir-aie` no longer owns the user-facing HTML page or workflow server

### Phase 3: Add operator/case generation and directory browsing

In `iron`:

1. add `scripts/design_vis.py`
2. add an operator/case adapter contract
3. implement MLIR generation into `build/design_vis/...`
4. implement generated-directory browsing
5. wire rendering through the wheel-installed API
6. open the browser automatically

Success criterion:

- `python scripts/design_vis.py --operator encoder_pipeline_ddr --case full_pipeline_seqpar_base`
  works from the IRON repo root and opens the generated MLIR directory

### Phase 4: Update documentation and operator flows

In `iron`:

1. add a short “Visualize a compiled design” section to the top-level README
2. replace old manual route-visualization instructions in operator READMEs
3. mention the only dependency requirement:
   rebuild/reinstall the `mlir-aie` wheel when the exporter/backend changes

Success criterion:

- users can discover the visualization flow from IRON docs alone

### Phase 5: Retire the source-owned frontend in `mlir-aie`

After IRON fully owns the UX:

1. keep the exporter/backend in `mlir-aie`
2. remove or de-emphasize the browser frontend under
   `mlir-aie/tools/aie-design-vis`
3. keep only dev/test fixtures there if they are still useful

Success criterion:

- there is one obvious user-facing home for the tool: IRON

## Rebuild/Install Rule

Anything that must stay in `mlir-aie` should be consumable through a rebuilt
wheel.

That means the IRON plan must not depend on:

- source-tree paths like `~/mlir-aie/tools/...`
- ad hoc imports from an editable checkout
- manually copied frontend assets from `mlir-aie`

The acceptable dependency is:

1. make the required `mlir-aie` change
2. rebuild the wheel
3. reinstall it into `ironenv`
4. IRON uses that installed render/export functionality while continuing to own
   operator/case generation and directory browsing

## Recommendation

The migration target should be:

- IRON owns the command, frontend, operator selection, case generation,
  generated-directory browsing, discovery logic, and docs
- `mlir-aie` owns the exporter and wheel-shipped render/export API only

That gives IRON the easiest workflow while keeping compiler-specific logic in
the compiler project.
