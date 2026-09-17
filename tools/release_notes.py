#!/usr/bin/env python3
"""Prints the CHANGELOG.md section of a version (for the GitHub release body).

    python tools/release_notes.py 0.1.0 [CHANGELOG.md]

Uses the `## [0.1.0]` section, or `## [Unreleased]` when the version has no section yet.
"""

import pathlib
import re
import sys


def section(text: str, version: str) -> str | None:
    heading = re.compile(r"^## \[(?P<name>[^\]]+)\]", re.M)
    matches = list(heading.finditer(text))
    for wanted in (version, "Unreleased"):
        for i, m in enumerate(matches):
            if m.group("name") == wanted:
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                return text[m.end():end].strip()
    return None


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    version = sys.argv[1].removeprefix("v")
    path = pathlib.Path(sys.argv[2] if len(sys.argv) == 3 else "CHANGELOG.md")
    body = section(path.read_text(encoding="utf-8"), version)
    if body is None:
        print(f"no section for {version} in {path}", file=sys.stderr)
        return 1
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
