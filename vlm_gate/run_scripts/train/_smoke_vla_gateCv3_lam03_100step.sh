#!/bin/bash
# srun debug smoke for C-v2 fm_token: 100-step from-scratch-head training,
# then probe the denoised gate on labeled frames.
set -u
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
OUT="$BASE/output/_gate_distill/vla_gateCv3lam03_smoke"
rm -rf "$OUT"
export WANDB_MODE=disabled
export PATH="$HOME/miniconda3/envs/quant_gate/bin:$PATH"
"$HOME/miniconda3/envs/quant_gate/bin/python" -u "$PRIV/scripts/gr00t_finetune.py" \
    --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir "$OUT" \
    --dataloader-num-workers 8 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name gateCv3lam03_smoke_100 \
    --batch-size 16 \
    --num-gpus 1 \
    --max-steps 100 \
    --save-steps 100 \
    --report-to tensorboard \
    --use-quant-gate \
    --quant-gate-mode fm_token \
    --quant-gate-oneway \
    --quant-gate-labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet \
    --quant-gate-loss-weight 0.3 \
    --quant-gate-label-tolerance 4
echo "== training done, probing gate =="
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u "$BASE/scripts/probe_gate_fm.py" \
    "$OUT/checkpoint-100" 8
