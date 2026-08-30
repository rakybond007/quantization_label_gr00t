#!/bin/bash
# Smoke: dexjoco dual-arm (5 bimanual tasks) baseline gr00t-n1.5, 5 steps.
# Verifies modality.json + dexjoco_dual_arm_multi_horizon data-config loads.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
DSET_ROOT=/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/bimanual
OUT="$BASE_DIR/ckpt/_smoke_dexjoco_dual_arm_baseline"
mkdir -p out

unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA_PATH/envs/gr00t/bin:$PATH"
cd "$BASE_DIR"
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/gr00t_finetune_dual_arm.py" \
    --dataset-path $DSET_ROOT/bimanual_assembly \
        $DSET_ROOT/bimanual_microwave_cook \
        $DSET_ROOT/bimanual_unlock_ipad \
        $DSET_ROOT/bimanual_hanoi \
        $DSET_ROOT/bimanual_photograph \
    --output-dir $OUT --dataloader-num-workers 2 \
    --data-config dexjoco_dual_arm_multi_horizon --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name SMOKE-dexjoco-dual-arm-baseline \
    --batch-size 2 --num-gpus 1 --max-steps 5 --save-steps 1000000 \
    --override-action-dim 64 \
    --report-to tensorboard 2>&1
RC=$?
echo "SMOKE_DUAL_ARM_BASELINE_DONE rc=$RC"
exit $RC
