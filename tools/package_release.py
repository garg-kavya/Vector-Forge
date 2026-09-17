#!/usr/bin/env python3
"""Packages an installed build as a release archive.

    python tools/package_release.py --build-dir out/build/msvc-release --out dist [--format zip]

Runs `cmake --install` into a staging directory and writes
dist/vectorforge-<version>-<os>-<arch>.<zip|tar.gz> plus a .sha256 file next to it. The archive
contains bin/vectorforge, the static library, headers, the CMake package and the docs.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def project_version(build_dir: pathlib.Path) -> str:
    cache = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^CMAKE_PROJECT_VERSION:STATIC=(.+)$", cache, re.M)
    if not m:
        raise SystemExit(f"no CMAKE_PROJECT_VERSION in {build_dir}/CMakeCache.txt")
    return m.group(1).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", choices=["zip", "tar.gz"],
                        default="zip" if platform.system() == "Windows" else "tar.gz")
    args = parser.parse_args()

    build_dir = pathlib.Path(args.build_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    version = project_version(build_dir)
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system(),
                                                                            platform.system())
    arch = {"AMD64": "x86_64", "x86_64": "x86_64"}.get(platform.machine(), platform.machine())
    name = f"vectorforge-{version}-{system}-{arch}"

    with tempfile.TemporaryDirectory() as tmp:
        stage = pathlib.Path(tmp) / name
        subprocess.run(["cmake", "--install", str(build_dir), "--prefix", str(stage)], check=True)
        archive = out / f"{name}.{args.format}"
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        if args.format == "zip":
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    info = zipfile.ZipInfo(f"{name}/{f.relative_to(stage).as_posix()}",
                                           date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (0o755 if f.parent.name == "bin" else 0o644) << 16
                    z.writestr(info, f.read_bytes())
        else:
            with tarfile.open(archive, "w:gz") as t:
                for f in files:
                    ti = t.gettarinfo(str(f), arcname=f"{name}/{f.relative_to(stage).as_posix()}")
                    ti.mtime = 0
                    ti.uid = ti.gid = 0
                    ti.uname = ti.gname = ""
                    with open(f, "rb") as fh:
                        t.addfile(ti, fh)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(f"{archive} ({archive.stat().st_size} bytes) sha256 {digest}")
    listing = [f.relative_to(out).as_posix() for f in out.iterdir()]
    print("\n".join(sorted(listing)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
