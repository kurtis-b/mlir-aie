#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent


def find_tool(env_var: str, repo_relpath: str, fallback_name: str) -> str:
    env_value = os.environ.get(env_var)
    if env_value:
      candidate = Path(env_value).expanduser().resolve()
      if candidate.is_file():
        return str(candidate)
    repo_candidate = REPO_ROOT / repo_relpath
    if repo_candidate.is_file():
        return str(repo_candidate)
    fallback = shutil.which(fallback_name)
    if fallback:
        return fallback
    raise FileNotFoundError(
        f"Could not find {fallback_name}. Set {env_var} or build/source the project first."
    )


def run_command(argv, *, input_text=None):
    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def render_design_json(name: str, mlir_text: str) -> dict:
    aie_opt = find_tool("AIE_OPT_BIN", "build/bin/aie-opt", "aie-opt")
    aie_translate = find_tool("AIE_TRANSLATE_BIN", "build/bin/aie-translate", "aie-translate")

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as tmp:
        tmp.write(mlir_text)
        tmp_path = Path(tmp.name)

    errors = []
    try:
        routed = run_command(
            [aie_opt, "--aie-create-pathfinder-flows", "--aie-find-flows", str(tmp_path)]
        )
        if routed.returncode == 0:
            translated = run_command(
                [aie_translate, "--aie-design-to-json", "-"], input_text=routed.stdout
            )
            if translated.returncode == 0:
                return {
                    "design": json.loads(translated.stdout),
                    "generation": {
                        "input_name": name,
                        "pipeline": "aie-opt --aie-create-pathfinder-flows --aie-find-flows | aie-translate --aie-design-to-json",
                    },
                }
            errors.append(
                {
                    "stage": "translate_routed",
                    "stderr": translated.stderr.strip(),
                }
            )
        else:
            errors.append(
                {
                    "stage": "route",
                    "stderr": routed.stderr.strip(),
                }
            )

        direct = run_command([aie_translate, "--aie-design-to-json", str(tmp_path)])
        if direct.returncode == 0:
            return {
                "design": json.loads(direct.stdout),
                "generation": {
                    "input_name": name,
                    "pipeline": "aie-translate --aie-design-to-json",
                },
            }
        errors.append(
            {
                "stage": "translate_direct",
                "stderr": direct.stderr.strip(),
            }
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    message = "\n\n".join(
        f"[{entry['stage']}]\n{entry['stderr'] or '(no stderr)'}" for entry in errors
    )
    raise RuntimeError(message)


class DesignVisHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(TOOL_DIR), **kwargs)

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "tool_dir": str(TOOL_DIR),
                    "repo_root": str(REPO_ROOT),
                }
            )
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/render-design":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            name = payload.get("name", "design.mlir")
            mlir_text = payload.get("mlir_text", "")
            if not isinstance(name, str) or not isinstance(mlir_text, str) or not mlir_text.strip():
                raise ValueError("Expected JSON body with non-empty string fields 'name' and 'mlir_text'.")
            result = render_design_json(name, mlir_text)
            self._send_json(result)
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                },
                status=HTTPStatus.BAD_REQUEST,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the AIE design visualization tool.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DesignVisHandler)
    print(f"Serving AIE design vis from {TOOL_DIR} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
