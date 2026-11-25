import os
import pytest

REQ_ROBOTWIN = pytest.mark.skipif(not os.environ.get("REGSP_ROBOTWIN_ROOT"), reason="needs RoboTwin checkout (REGSP_ROBOTWIN_ROOT)")


def _weights_ready():
    try:
        import torch
    except Exception:
        return False
    return bool(os.environ.get("REGSP_CKPT_ROOT") and os.environ.get("REGSP_BACKBONE_CKPT") and os.environ.get("REGSP_ROBOTWIN_ROOT") and torch.cuda.is_available())


REQ_WEIGHTS = pytest.mark.skipif(not _weights_ready(), reason="needs weight bundle + backbone checkpoint + RoboTwin + CUDA (docs/CHECKPOINTS.md)")
