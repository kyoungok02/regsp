"""StagePlan: learned stage variables (grasp rotation, approach direction, pre-grasp distance, waypoint) and the geometric pre-grasp chunk builder."""
from regsp.backbone.stageplan_v2 import (  # noqa: F401
    PRE_DIST_SCALE_M,
    WP_SCALE_M,
    SafeDiffVLAStagePlanV2Policy,
    StagePlanV2Head,
    StageVars,
    build_pregrasp_chunk,
)
