#!/bin/bash
#SBATCH --job-name=vla_joint_training_groot_n15_robocasa_quantizability_gate_token_gemma_labels_60k
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=48:00:00
#SBATCH --output=out/%j-vla_gateC_gemma_60k.out
#SBATCH --error=out/%j-vla_gateC_gemma_60k.err

# Joint VLA training (plan C): baseline robocasa recipe (60k steps, global
# batch 64 = 2 GPU x 32) + quantizability gate token with one-way attention
# and dedicated readout. Loss = flow-matching action loss + 0.1 * BCE(gate
# logit, gemma teacher p_yes). Starting weights identical to the baseline
# recipe (nvidia/GR00T-N1.5-3B pretrained).
set -u
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
mkdir -p out

export WANDB_PROJECT=gate-distill
OUT_DIR="${MODEL_OUTPUT_DIR:?set --export=ALL,MODEL_OUTPUT_DIR=...}"

export PATH="$HOME/miniconda3/envs/quant_gate/bin:$PATH"
"$HOME/miniconda3/envs/quant_gate/bin/python" -u "$PRIV/scripts/gr00t_finetune.py" \
    --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
    --output-dir "$OUT_DIR" \
    --dataloader-num-workers 16 \
    --data-config single_panda_gripper \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name gateC_joint_gemma_60k \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb \
    --use-quant-gate \
    --quant-gate-labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_gemma_v1/labels/full_merged.parquet \
    --quant-gate-loss-weight 0.1 \
    --quant-gate-label-tolerance 4
