#!/bin/bash
#SBATCH --job-name=vla_joint_training_groot_n15_robocasa_gateC_v2_fm_token_from_scratch_head_cosmos_60k
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-vla_gateCv2_fm_cosmos_60k.out
#SBATCH --error=out/%j-vla_gateCv2_fm_cosmos_60k.err

# C-v2 "fm_token": gate is a regular DiT token; flow-matching target extended
# to [action_chunk(16); gate(1)] with gate value 2*p_yes-1. Action head is
# RANDOMLY INITIALIZED (--no-load-action-head); backbone follows the baseline
# recipe (pretrained Eagle, tune projector only). 60k steps, global batch 64.
set -u
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
mkdir -p out

export WANDB_PROJECT=gate-distill
# torchrun relaunch (num_gpus>1) must find the env's torchrun on PATH.
export PATH="$HOME/miniconda3/envs/quant_gate/bin:$PATH"
OUT_DIR="${MODEL_OUTPUT_DIR:?set --export=ALL,MODEL_OUTPUT_DIR=...}"

"$HOME/miniconda3/envs/quant_gate/bin/python" -u "$PRIV/scripts/gr00t_finetune.py" \
    --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir "$OUT_DIR" \
    --dataloader-num-workers 16 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --no-load-action-head \
    --run-name gateCv2_fm_cosmos_60k \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb \
    --use-quant-gate \
    --quant-gate-mode fm_token \
    --quant-gate-labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet \
    --quant-gate-loss-weight 1.0 \
    --quant-gate-label-tolerance 4
