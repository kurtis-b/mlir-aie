#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AIE_OPT_BIN="${AIE_OPT_BIN:-$(command -v aie-opt || true)}"
AIE_TRANSLATE_BIN="${AIE_TRANSLATE_BIN:-$(command -v aie-translate || true)}"

if [[ -z "${AIE_OPT_BIN}" && -x "${REPO_ROOT}/build/bin/aie-opt" ]]; then
  AIE_OPT_BIN="${REPO_ROOT}/build/bin/aie-opt"
fi
if [[ -z "${AIE_TRANSLATE_BIN}" && -x "${REPO_ROOT}/build/bin/aie-translate" ]]; then
  AIE_TRANSLATE_BIN="${REPO_ROOT}/build/bin/aie-translate"
fi

if [[ -z "${AIE_OPT_BIN}" || -z "${AIE_TRANSLATE_BIN}" ]]; then
  echo "Could not find aie-opt and aie-translate. Source the env or set AIE_OPT_BIN/AIE_TRANSLATE_BIN." >&2
  exit 1
fi

"${AIE_OPT_BIN}" --aie-create-pathfinder-flows --aie-find-flows "$1" \
  | "${AIE_TRANSLATE_BIN}" --aie-design-to-json > "$2".json
