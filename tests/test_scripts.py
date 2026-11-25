import importlib.util
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.py"))


@pytest.mark.parametrize("p", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_script_compiles(p):
    py_compile.compile(str(p), doraise=True, cfile=str(Path(__import__("tempfile").gettempdir()) / "regsp_pyc" / (p.stem + ".pyc")))


@pytest.mark.skipif(importlib.util.find_spec("sapien") is None, reason="needs sapien")
def test_eval_cli_help():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate/robotwin_eval.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--st_seed" in r.stdout and "demo_randomized" in r.stdout


def test_bundle_tool_cli_help():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/utils/assemble_bundle.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--verify" in r.stdout
