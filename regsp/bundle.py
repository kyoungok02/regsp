"""Per-task weight bundle: canonical file names for every learned component that the frozen ReGSP pipeline loads.

Layout (``$REGSP_CKPT_ROOT/<task>/``)::

    stageplan_head.pt          StagePlanV2Head (+ position statistics for the StagePlan fallback)
    decoder.pt                 task-adapted TemporalActionDecoder (+ TargetQueryExtractor, VisualGroundingCrossAttention, TargetPointHead, statistics)
    proposal_head_camera.pt    class-agnostic 64x64 proposal heads (head / left wrist / right wrist camera)
    proposal_left_camera.pt
    proposal_right_camera.pt
    anchor_head.pt             AnchorHead (+ keep/mean-offset decision)
    referent_scorer.pt         HR referent scorer (counterfactual-trained)          [not stack_blocks_two]
    subgoal_scorer.pt          stage-conditioned subgoal scorer                      [stack_blocks_two]
    bim_decoder.pt             bimanual decoder initialisation                       [two-arm tasks]
    progress.pt                adaptive decoder + progress head + statistics         [two-arm tasks]
    place_scorer.pt            place-referent scorer                                 [handover_block]

``configs/bundle_manifest.json`` records, for every file, the SHA-256 and the research-tree path it was taken from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from regsp import paths

FILES = {
    "stageplan_head": "stageplan_head.pt",
    "decoder": "decoder.pt",
    "proposal_head_camera": "proposal_head_camera.pt",
    "proposal_left_camera": "proposal_left_camera.pt",
    "proposal_right_camera": "proposal_right_camera.pt",
    "anchor_head": "anchor_head.pt",
    "referent_scorer": "referent_scorer.pt",
    "subgoal_scorer": "subgoal_scorer.pt",
    "bim_decoder": "bim_decoder.pt",
    "progress": "progress.pt",
    "place_scorer": "place_scorer.pt",
}
REQUIRED = {
    "click_bell": ("stageplan_head", "decoder", "proposal_head_camera", "proposal_left_camera", "proposal_right_camera", "anchor_head", "referent_scorer"),
    "place_container_plate": ("stageplan_head", "decoder", "proposal_head_camera", "proposal_left_camera", "proposal_right_camera", "anchor_head", "referent_scorer"),
    "stack_blocks_two": ("stageplan_head", "decoder", "proposal_head_camera", "proposal_left_camera", "proposal_right_camera", "anchor_head", "subgoal_scorer", "bim_decoder", "progress"),
    "handover_block": ("stageplan_head", "decoder", "proposal_head_camera", "proposal_left_camera", "proposal_right_camera", "anchor_head", "referent_scorer", "bim_decoder", "progress", "place_scorer"),
}


@dataclass
class TaskBundle:
    task: str
    root: Path

    @classmethod
    def for_task(cls, task: str, ckpt_root: str | Path | None = None) -> "TaskBundle":
        return cls(task, Path(ckpt_root) / task if ckpt_root else paths.ckpt_root() / task)

    def path(self, name: str) -> Path:
        return self.root / FILES[name]

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def missing(self) -> list[str]:
        return [n for n in REQUIRED[self.task] if not self.has(n)]

    def referent_scorer_kind(self) -> str:
        """``"cycle"`` if the stored scorer carries the cycle-index embedding (CycleScorer), else ``"plain"`` (Scorer)."""
        import torch

        sd = torch.load(self.path("referent_scorer"), map_location="cpu")
        return "cycle" if "cyc.weight" in sd else "plain"

    def manifest(self) -> dict:
        return json.loads((paths.CONFIG_DIR / "bundle_manifest.json").read_text())["tasks"][self.task]
