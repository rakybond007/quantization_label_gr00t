#!/bin/bash
# srun debug smoke: 100-step joint training, check action loss + gate BCE descend.
set -u
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
OUT="$BASE/output/_gate_distill/vla_gate_smoke"
rm -rf "$OUT"
export WANDB_MODE=disabled
"$HOME/miniconda3/envs/quant_gate/bin/python" -u "$PRIV/scripts/gr00t_finetune.py" \
    --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir "$OUT" \
    --dataloader-num-workers 8 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name gateC_smoke_100 \
    --batch-size 16 \
    --num-gpus 1 \
    --max-steps 100 \
    --save-steps 100000 \
    --report-to tensorboard \
    --use-quant-gate \
    --quant-gate-labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet \
    --quant-gate-loss-weight 0.1 \
    --quant-gate-label-tolerance 4
