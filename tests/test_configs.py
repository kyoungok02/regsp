import json
import re
from pathlib import Path

import pytest

from regsp import bundle, paths
from regsp.config import load_final_config, task_env

ROOT = Path(__file__).resolve().parents[1]


def test_final_config_covers_the_four_released_tasks():
    cfg = load_final_config(); assert set(cfg["tasks"]) == set(paths.TASKS)
    assert cfg["evaluation_protocol"]["clean_seed_start"] == 8200000 and cfg["evaluation_protocol"]["randomized_seed_start"] == 8700000
    for t in paths.TASKS:
        assert cfg["tasks"][t]["bimanual"] == (t in paths.BIMANUAL_TASKS)
    assert task_env("handover_block")["CLC_APPROACH"] == "2" and "CLC_APPROACH" not in task_env("stack_blocks_two")
    e = task_env("place_container_plate"); assert (e["CLC_FB"], e["CLC_REVERIFY"], e["ACL_NEWG"]) == ("1", "1", "1")


def test_bundle_manifest_matches_required_files():
    man = json.loads((ROOT / "configs" / "bundle_manifest.json").read_text())["tasks"]
    for t in paths.TASKS:
        assert {v["component"] for v in man[t].values()} == set(bundle.REQUIRED[t]), t
        assert all(len(v["sha256"]) == 64 for v in man[t].values())


def test_packaged_statistics_present():
    for t in paths.TASKS:
        assert paths.data_file("tasks", t, "runtime_stats.json").exists()
    for t in paths.BIMANUAL_TASKS:
        assert paths.data_file("tasks", t, "bim_stats.json").exists()


def test_no_machine_specific_paths_in_tracked_text():
    pat = re.compile(r"(/home/[a-z_]+|/tmp/safediff|/lerobot/outputs|/lerobot/examples|/lerobot/src|/lerobot_wt|miniforge3)")
    bad = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".json", ".toml", ".txt", ".sh", ".yml", ".yaml", ".csv", ".env", ".svg"} and ".git" not in p.parts and p.name != "test_configs.py":
            if pat.search(p.read_text(errors="ignore")):
                bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad
