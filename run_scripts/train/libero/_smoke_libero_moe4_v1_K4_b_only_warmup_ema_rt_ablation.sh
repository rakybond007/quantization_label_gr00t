#!/bin/bash
# Smoke for libero MoE4 v1 K4 ablation set (L1/L2/L3/L5).
# Uses L5 config (warmup=10k + EMA=0.95 + router_temp=1.0) — most-changed; if this
# runs, the other 3 (subset of changes) also run. 100 steps, no save, no wandb.
# Run inside an interactive 2-GPU srun shell.
set -u

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT_DIR="$BASE_DIR/ckpt/libero/_smoke_moe4_v1_K4_ablation"
DATA_DIR="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"

cd "$BASE_DIR"
# Prevent openpi venv / PYTHONPATH from leaking into subprocesses
# (torchrun / accelerate must come from gr00t env, not openpi venv).
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
PY="$CONDA_PATH/envs/gr00t/bin/python"

PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR" "$PY" $BASE_DIR/scripts/gr00t_finetune_fair_moe.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 16 \
    --data-config libero_multi_horizon \
    --embodiment-tag libero \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE_libero_moe4_v1_K4_ablation \
    --batch-size 32 --num-gpus 2 --max-steps 10 --save-steps 200 \
    --report-to tensorboard \
    --discrete-action-dims 6 \
    --use-merged-8-head --merged-8-weight 1.0 \
    --use-merged-4-head --merged-4-weight 1.0 \
    --use-native-8-head --native-8-weight 1.0 \
    --use-moe-routing \
    --moe-body-mode=per_expert_h \
    --moe-num-experts=4 \
    --moe-routing-mode=single_pick \
    --moe-expert-n-layers=2 \
    --moe-router-temp=1.0 --moe-target-temp=0.3 \
    --moe-balance-weight=0.05 --moe-supervise-weight=0.1 \
    --moe-router-warmup-steps=10000 \
    --no-moe-shared-t \
    --moe-normalize-loss-for-kl --moe-loss-ema-momentum=0.95 \
    --moe-balance-type=mean_sq \
    --moe-length-cost-weight=0.0
echo "SMOKE_LIBERO_ABLATION_DONE rc=$?"
