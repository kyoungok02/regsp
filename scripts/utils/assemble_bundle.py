#!/usr/bin/env python
"""Assemble a ReGSP weight bundle from a research-tree layout (symlinks by default) and verify every SHA-256 in configs/bundle_manifest.json.

usage: assemble_bundle.py --v21-root <outputs/robotwin_v21_4task_counterfactual> --orig-root <outputs/robotwin_safediff_original_10task> --out <bundle root> [--copy]
       assemble_bundle.py --verify <bundle root>
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "configs" / "bundle_manifest.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(root):
    man = json.loads(MANIFEST.read_text()); bad = 0
    for task, files in man["tasks"].items():
        for name, meta in files.items():
            p = Path(root) / task / name
            if not p.exists():
                print("MISSING", p); bad += 1
            elif sha(p) != meta["sha256"]:
                print("HASH MISMATCH", p); bad += 1
    print("bundle OK" if not bad else f"{bad} problem(s)")
    return bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v21-root"); ap.add_argument("--orig-root"); ap.add_argument("--out"); ap.add_argument("--copy", action="store_true"); ap.add_argument("--verify")
    a = ap.parse_args()
    if a.verify:
        sys.exit(1 if verify(a.verify) else 0)
    man = json.loads(MANIFEST.read_text()); roots = {"v21_4task_counterfactual": a.v21_root, "original_10task": a.orig_root}
    for task, files in man["tasks"].items():
        (Path(a.out) / task).mkdir(parents=True, exist_ok=True)
        for name, meta in files.items():
            tree, rel = meta["research_tree"].split("/", 1); src = Path(roots[tree]) / rel; dst = Path(a.out) / task / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copy2(src, dst) if a.copy else os.symlink(src.resolve(), dst)
    sys.exit(1 if verify(a.out) else 0)


if __name__ == "__main__":
    main()
