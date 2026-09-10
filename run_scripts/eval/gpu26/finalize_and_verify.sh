#!/bin/bash
#SBATCH --job-name=finalize_retimed_meta_and_verify_gr00t_can_load_it
#SBATCH --qos=eval
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/final_%j.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/final_%j.out
set -uo pipefail
BENCH=${BENCH:-robocasa}
case "$BENCH" in
  robocasa) SRC=/s3data/robocasa_mg_gr00t_300/v1; CFG=single_panda_gripper; TAG=new_embodiment ;;
  libero)   SRC=/s3data/libero_gr00t_delta/v1;    CFG=libero;               TAG=libero         ;;
esac
# The build shards ran across partitions, so the only complete copy is the S3 mirror of
# /ckpt. Assemble the dataset there and keep it there -- training reads it from /s3ckpt.
DST=/s3ckpt/$USER/datasets/${BENCH}_demospeedup
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate quant_gate
export HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
cd "$HOME/quant_label_workspace/quantization_label_gr00t"

echo "[$(date '+%T')] === meta for $BENCH ==="
python -u baselines/finalize_retimed_dataset.py --src "$SRC" --dst "$DST"

echo "[$(date '+%T')] === load check: does GR00T open it, and is video aligned with action? ==="
python -u - "$DST" "$CFG" "$TAG" <<'PY'
import sys, numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
root, cfg, tag = sys.argv[1:4]
dc = DATA_CONFIG_MAP[cfg]
ds = LeRobotSingleDataset(dataset_path=root, modality_configs=dc.modality_config(),
                          video_backend="decord", transforms=None, embodiment_tag=tag)
print(f"  len(dataset)={len(ds)}  trajectories={len(ds.trajectory_ids)}")
s = ds[0]
for k in sorted(s):
    v = s[k]
    print(f"    {k:44} {getattr(v,'shape',type(v).__name__)}")
# every episode's video must have exactly as many frames as the parquet says
bad = 0
for tid in list(ds.trajectory_ids)[:40]:
    T = int(ds.trajectory_lengths[tid])
    for key in dc.video_keys:
        try:
            f = ds.get_video(tid, key, T - 1)          # last frame must exist
            if f is None or f.shape[0] == 0: bad += 1
        except Exception as e:
            bad += 1; print(f"    ep{tid} {key}: {e}")
print(f"  last-frame readable on 40 episodes x {len(dc.video_keys)} views: "
      f"{'OK' if bad==0 else f'{bad} FAILURES'}")
assert bad == 0
PY
echo "[$(date '+%T')] done"
