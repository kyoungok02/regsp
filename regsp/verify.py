"""Digest helpers used by tests/ and scripts/utils/verify_bundle_load.py to compare loaded modules with the frozen reference."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _mh(m) -> str:
    h = hashlib.sha256()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _ah(a) -> str:
    a = a.detach().cpu() if hasattr(a, "detach") else a
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]


def module_digests(pol) -> dict:
    out = {}
    for n in ("v2", "sc", "csc", "psc", "ah", "D", "Q", "Vg", "TPH", "bimdec", "adec", "PH"):
        m = getattr(pol, n, None)
        if m is not None:
            out[n] = (type(m).__name__, _mh(m), sum(p.numel() for p in m.parameters()))
    for c, m in pol.heads.items():
        out["head_" + c] = (type(m).__name__, _mh(m), 0)
    out["backbone"] = ("pol", _mh(pol.pol), 0)
    for n in ("pm", "ps", "dm", "dsd"):
        out[n] = ("arr", _ah(getattr(pol, n)), 0)
    if pol.anchor:
        out["anchor_dec"] = ("anchor", _ah(pol.anchor["mean_off"]) + str(pol.anchor["keep"]), 0)
    out["stats"] = ("json", hashlib.sha256(json.dumps(pol.stats, sort_keys=True).encode()).hexdigest()[:16], 0)
    if getattr(pol, "bst", None) is not None:
        out["bst"] = ("json", hashlib.sha256(json.dumps(pol.bst, sort_keys=True).encode()).hexdigest()[:16], 0)
    for n in ("K", "max_retry", "use_prog", "maxcalls", "clc_approach", "clc_fb", "clc_reverify", "newg", "anch_max", "ref_tol", "tau", "open_min", "nview", "maxcyc", "post_chunks", "defer"):
        if hasattr(pol, n):
            out["cfg_" + n] = ("val", str(getattr(pol, n)), 0)
    out["opt"] = ("val", json.dumps(pol.opt, sort_keys=True), 0); out["cal"] = ("val", json.dumps(pol.cal, sort_keys=True), 0); out["clc_stats"] = ("val", json.dumps(pol.clc_stats, sort_keys=True), 0)
    return out
