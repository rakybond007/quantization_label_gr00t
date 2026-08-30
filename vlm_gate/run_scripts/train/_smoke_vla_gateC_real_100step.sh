#!/bin/bash
# srun debug smoke for gate-C on real DROID pnp: 100-step fm_token+oneway,
# then probe the denoised gate on labeled frames.
set -u
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
OUT="$BASE/output/_gate_distill/vla_gateC_real_smoke"
rm -rf "$OUT"
export WANDB_MODE=disabled
export PATH="$HOME/miniconda3/envs/quant_gate/bin:$PATH"
"$HOME/miniconda3/envs/quant_gate/bin/python" -u "$PRIV/scripts/gr00t_finetune.py" \
    --dataset-path /sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects \
    --output-dir "$OUT" \
    --dataloader-num-workers 8 \
    --data-config real_droid_joint \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name gateC_real_smoke_100 \
    --video-backend torchvision_av \
    --batch-size 16 \
    --num-gpus 1 \
    --max-steps 100 \
    --save-steps 100 \
    --report-to tensorboard \
    --use-quant-gate \
    --quant-gate-mode fm_token \
    --quant-gate-oneway \
    --quant-gate-labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_real_droid_pnp_v1/labels/real_luna_full.parquet \
    --quant-gate-loss-weight 1.0 \
    --quant-gate-label-tolerance 2
echo "== training done, probing gate =="
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u "$BASE/scripts/probe_gate_fm_real.py" \
    "$OUT/checkpoint-100" 8
