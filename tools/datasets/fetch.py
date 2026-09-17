#!/usr/bin/env python3
"""Downloads public ANN benchmark datasets into $VF_DATA_DIR (docs/benchmarking.md, "Datasets").

    python tools/datasets/fetch.py --list
    python tools/datasets/fetch.py sift1m --yes

Nothing is downloaded without --yes: the listed size is shown first (DESIGN §16.2 requires the
user's approval). Files are verified against the SHA-256 recorded below; a dataset without a
recorded hash prints the hash it computed so that it can be pinned after an independent check
(no dataset has been downloaded by the project so far, so none is pinned yet).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import tarfile
import urllib.request

DATASETS = {
    "sift1m": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "approx_bytes": 168_000_000,
        "sha256": None,
        "license": "TEXMEX corpus (INRIA), free for research use",
        "note": "1M x 128 base, 10K queries, ground truth (.fvecs/.ivecs); use `vectorforge build` directly.",
        "unpack": "tar",
    },
    "glove-100": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "approx_bytes": 485_000_000,
        "sha256": None,
        "license": "GloVe: PDDL 1.0 (Stanford NLP); ann-benchmarks packaging MIT",
        "note": "1.18M x 100, cosine; convert with tools/datasets/hdf5_to_npy.py (needs h5py).",
        "unpack": None,
    },
    "nytimes-256": {
        "url": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "approx_bytes": 301_000_000,
        "sha256": None,
        "license": "ann-benchmarks packaging MIT; derived from the UCI Bag of Words data set",
        "note": "290K x 256, cosine; the closest ann-benchmarks set to a real 384-d embedding "
                "distribution. A licence-checked 384-d/1536-d embedding set has not been chosen.",
        "unpack": None,
    },
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", choices=sorted(DATASETS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the download")
    parser.add_argument("--data-dir", default=os.environ.get("VF_DATA_DIR", "datasets"))
    args = parser.parse_args()

    if args.list or not args.name:
        for name, d in DATASETS.items():
            pinned = "pinned" if d["sha256"] else "not pinned"
            print(f"{name:12} ~{d['approx_bytes'] / 1e6:6.0f} MB  sha256 {pinned}  {d['url']}")
            print(f"{'':12} {d['license']}; {d['note']}")
        return 0

    d = DATASETS[args.name]
    target_dir = pathlib.Path(args.data_dir) / args.name
    target = target_dir / pathlib.PurePosixPath(d["url"]).name
    print(f"{args.name}: {d['url']} (~{d['approx_bytes'] / 1e6:.0f} MB) -> {target}")
    print(f"licence: {d['license']}")
    if not args.yes:
        print("re-run with --yes to download")
        return 1
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(d["url"], partial)  # noqa: S310 - fixed, listed URLs only
        partial.replace(target)
    digest = sha256(target)
    if d["sha256"] is None:
        print(f"sha256 {digest} (not pinned; verify independently before recording it)")
    elif digest != d["sha256"]:
        print(f"sha256 mismatch: got {digest}, expected {d['sha256']}", file=sys.stderr)
        return 1
    else:
        print("sha256 OK")
    if d["unpack"] == "tar":
        with tarfile.open(target) as tar:
            tar.extractall(target_dir, filter="data")
        print(f"unpacked into {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
