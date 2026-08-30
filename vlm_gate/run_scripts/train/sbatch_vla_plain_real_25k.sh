#!/bin/bash
#SBATCH --job-name=train_vla_plain_real_droid_pnp_finetune_25k_gb64_2gpu_baseline_for_gateA_module
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3,worker-node119
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
    --run-name vla_plain_real_25k \
    --video-backend torchvision_av \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 25000 \
    --resume \
    --save-steps 5000 \
    --report-to wandb \
