#!/bin/bash
#SBATCH --job-name=demospeedup_stage_dataset_then_finetune_sixty_thousand_steps
#SBATCH -p a100
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=32
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/dstrain_%j.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/dstrain_%j.out
set -euo pipefail
# DemoSpeedup stage 3: train on the entropy-accelerated demonstrations.
# Recipe pinned to each benchmark's existing control checkpoint, so the re-timed data is
# the only variable: same 60k steps, same global batch, same action_horizon (16).
#   robocasa control prehj/GR00T-N1.5-robocasa-baseline        global 64
#   libero   control prehj/GR00T-N1.5-libero-baseline-bs32-60k global 32
BENCH=${BENCH:-robocasa}
case "$BENCH" in
  robocasa) SRC=/s3data/robocasa_mg_gr00t_300/v1; CFG=single_panda_gripper; TAG=new_embodiment; PER_GPU=32 ;;
  libero)   SRC=/s3data/libero_gr00t_delta/v1;    CFG=libero;               TAG=libero;         PER_GPU=16 ;;
esac
MIRROR=/s3ckpt/$USER/datasets/${BENCH}_demospeedup     # complete copy (build shards ran across AZs)
DATA=/ckpt/$USER/datasets/${BENCH}_demospeedup_staged  # Lustre copy for training reads
CKPT=/ckpt/$USER/${BENCH}/demospeedup_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DATA" "$CKPT"

source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate quant_gate
export HF_TOKEN=$(cat "$HOME/quant_label_workspace/hf_token.txt")
export HF_HUB_DISABLE_XET=1 NO_ALBUMENTATIONS_UPDATE=1
export WANDB_PROJECT=GR00T-demospeedup WANDB_MODE=${WANDB_MODE:-offline}
cd "$HOME/quant_label_workspace/quantization_label_gr00t"

echo "[$(date '+%F %T')] node=$(hostname) bench=$BENCH"
echo "[$(date '+%T')] staging $MIRROR -> $DATA (training must read Lustre, not the S3 mount)"
cp -ru "$MIRROR"/. "$DATA"/
np=$(find "$DATA/data" -name '*.parquet' | wc -l); nv=$(find "$DATA/videos" -name '*.mp4' | wc -l)
echo "  staged: $np parquet, $nv mp4"
echo "[$(date '+%T')] writing meta on the Lustre copy (the S3 mount rejects chmod)"
python -u baselines/finalize_retimed_dataset.py --src "$SRC" --dst "$DATA"

echo "[$(date '+%T')] load check: GR00T opens it, video aligned with action"
python -u - "$DATA" "$CFG" "$TAG" <<'PY2'
import sys, numpy as np, decord
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
root, cfg, tag = sys.argv[1:4]
dc = DATA_CONFIG_MAP[cfg]
ds = LeRobotSingleDataset(dataset_path=root, modality_configs=dc.modality_config(),
                          video_backend="decord", transforms=None, embodiment_tag=tag)
print(f"  len(dataset)={len(ds)} trajectories={len(ds.trajectory_ids)}")
bad = []
for tid in list(ds.trajectory_ids)[:40]:
    T = int(ds.trajectory_lengths[tid])
    # go through get_step_data: it populates curr_traj_data, and it is the path training
    # actually uses. Calling get_video directly asserts on curr_traj_data being unset.
    for f in (0, T // 2, T - 1):
        step = ds.get_step_data(tid, f)
        for k in dc.video_keys:
            v = step[k]
            if v is None or v.shape[0] == 0 or v.ndim != 4:
                bad.append((tid, f, k, "empty"))
    # the encoded video must have exactly as many frames as the parquet, or observations
    # drift out of step with actions part-way through an episode
    for k in dc.video_keys:
        n = len(decord.VideoReader(ds.get_video_path(tid, k.replace("video.", "")).as_posix()))
        if n != T:
            bad.append((tid, None, k, f"video {n} frames vs parquet {T}"))
print(f"  40 episodes x {len(dc.video_keys)} views, 3 frames each + length match: "
      f"{'OK' if not bad else str(len(bad)) + ' FAILURES'}")
for b in bad[:5]:
    print("   ", b)
assert not bad
PY2

python - "$DATA" "$np" "$nv" <<'PY'
import json,sys
root,np_,nv_=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
d=json.load(open(f"{root}/meta/info.json"))
cams=len([k for k in d["features"] if k.startswith("observation.images.")])
assert np_==d["total_episodes"], f"parquet {np_} != episodes {d['total_episodes']}"
assert nv_==d["total_episodes"]*cams, f"mp4 {nv_} != {d['total_episodes']}*{cams}"
print(f"  meta agrees: {d['total_episodes']} episodes, {d['total_frames']} frames, {cams} cameras")
PY

echo "[$(date '+%F %T')] === training ==="
python scripts/gr00t_finetune.py \
    --dataset-path "$DATA" --output-dir "$CKPT" \
    --data-config $CFG --embodiment-tag $TAG \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-$BENCH-demospeedup \
    --batch-size $PER_GPU --num-gpus 2 \
    --max-steps ${MAX_STEPS:-60000} --save-steps 10000 \
    --dataloader-num-workers 16 --report-to wandb
echo "[$(date '+%F %T')] checkpoints:"; ls -d "$CKPT"/checkpoint-* 2>/dev/null
