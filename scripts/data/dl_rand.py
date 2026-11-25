"""[4-task study] download ONLY meta/data + the first video file (file-000, which holds the first >=100 episodes) of aloha-agilex_randomized_500 of the four tasks (official RoboTwin LeRobot-v3.0 dataset, same pinned revision as the Clean-50 data); training-domain demonstrations, official seeds, disjoint from every evaluation seed of this study."""
from huggingface_hub import snapshot_download
from regsp import paths as P
T = ["click_bell", "place_container_plate", "stack_blocks_two", "handover_block"]
print(snapshot_download("hxma/RoboTwin-LeRobot-v3.0", repo_type="dataset", revision="5d857cc7", local_dir=(str(P.work_root()) + "/data_rand/hf"), allow_patterns=[f"{t}/aloha-agilex_randomized_500/{x}" for t in T for x in ("meta/*", "data/*", "videos/*/chunk-000/file-000.mp4")], max_workers=4))
