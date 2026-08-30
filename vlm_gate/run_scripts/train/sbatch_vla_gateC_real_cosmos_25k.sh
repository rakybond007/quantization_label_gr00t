#!/bin/bash
#SBATCH --job-name=train_vla_gateC_real_droid_fm_oneway_cosmos_labels_25k_gb64_2gpu_teacher_ablation
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-vla_gateC_real_25k.out
#SBATCH --error=out/%j-vla_gateC_real_25k.err

# Gate-C on real DROID pnp_objects (MoSS rig): fm_token gate + one-way mask,
# global batch 64 (2 GPU x 32), 25k steps, luna real labels (every-4, tol 2).
set -u
PRIV="$HOME/quantization_agent_workspace/Isaac-GR00T"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
mkdir -p out

export WANDB_PROJECT=gate-distill
export PATH="$HOME/miniconda3/envs/quant_gate/bin:$PATH"
OUT_DIR="${MODEL_OUTPUT_DIR:?set --export=ALL,MODEL_OUTPUT_DIR=...}"

"$HOME/miniconda3/envs/quant_gate/bin/python" -u "$PRIV/scripts/gr00t_finetune.py" \
    --dataset-path /sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects \
    --output-dir "$OUT_DIR" \
    --dataloader-num-workers 8 \
    --data-config real_droid_joint \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name gateC_real_cosmos_25k \
    --video-backend torchvision_av \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 25000 \
    --save-steps 5000 \
    --report-to wandb \
    --use-quant-gate \
    --quant-gate-mode fm_token \
    --quant-gate-oneway \
    --quant-gate-labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_real_droid_pnp_v1/labels/real_cosmos_full.parquet \
    --quant-gate-loss-weight 1.0 \
    --quant-gate-label-tolerance 2
