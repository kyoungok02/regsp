"""Visual grounding: proposals -> referent scoring -> two-view triangulation -> AnchorHead.

``import regsp.grounding as R`` exposes the same names as the research tree's ``v21rt`` module.
"""
from .geometry import (K, KINV, W, Hh, THR, GAP_MAX, KCAP, CROP_PX, MATCH_PX, CAMS, half, sq, w2c4, project, ray, triangulate, associate)  # noqa: F401
from .proposals import Head, focal, targets, decode  # noqa: F401
from .scorer import D, Scorer, CycleScorer, SubgoalScorer, SD  # noqa: F401
from .anchor import AnchorHead  # noqa: F401
