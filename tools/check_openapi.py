#!/usr/bin/env python3
"""Checks that docs/openapi.yaml documents exactly the routes the server registers.

Usage: check_openapi.py <vectorforge executable> [docs/openapi.yaml]

The server's route table comes from `vectorforge serve --list-routes`. The spec is read without a
YAML library: operations are the `get:`/`post:`/`delete:` keys (4-space indent) under the
2-space-indented path keys of the top-level `paths:` mapping, which is how the spec is written.
"""

import pathlib
import re
import subprocess
import sys

METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def spec_routes(text):
    routes = set()
    in_paths = False
    path = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_paths = line.rstrip() == "paths:"
            path = None
            continue
        if not in_paths:
            continue
        m = re.match(r"^  (/\S*):\s*$", line)
        if m:
            path = m.group(1)
            continue
        m = re.match(r"^    ([a-z]+):", line)
        if m and path and m.group(1) in METHODS:
            routes.add((m.group(1).upper(), path))
    return routes


def server_routes(executable):
    executable = str(pathlib.Path(executable).resolve())
    out = subprocess.run([executable, "serve", "--list-routes"], check=True,
                         capture_output=True, text=True).stdout
    routes = set()
    for line in out.splitlines():
        if line.strip():
            method, path = line.split()
            routes.add((method, path))
    return routes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    spec_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "docs/openapi.yaml")
    spec = spec_routes(spec_path.read_text(encoding="utf-8"))
    server = server_routes(sys.argv[1])
    ok = True
    for method, path in sorted(server - spec):
        print(f"not documented: {method} {path}")
        ok = False
    for method, path in sorted(spec - server):
        print(f"documented but not served: {method} {path}")
        ok = False
    if ok:
        print(f"OK: {len(server)} routes documented in {spec_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
