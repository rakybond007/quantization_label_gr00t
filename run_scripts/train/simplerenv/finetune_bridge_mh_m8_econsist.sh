#!/bin/bash
#SBATCH --job-name=groot_n1_5_bridge_mh_m8_econsist
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on Bridge: multi-horizon + merged_8 + ensemble-consistency loss"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_bridge_mh_m8_econsist.out
#SBATCH --error=out/%j-groot_n1_5_bridge_mh_m8_econsist.err

# Bridge / WidowX (SimplerEnv) full multi-horizon + m8 + ensemble-consistency.
# Loss: 1.0*L_main16 + 1.0*L_m8 + warmup*(1.0*L_f2 + 1.0*L_f4) + c_warmup*0.1*L_consist
# where L_consist = MSE(pair_sum(v_main), v_f2[:8]) + MSE(v_m8, v_f2[:8]), shared-t.
# Global batch = 2 * 32 = 64.
#
# NOTE: Bridge action is x,y,z,roll,pitch,yaw,gripper (7 dims). The gripper
# index (in concatenated action vector) is 6 -> set as discrete-action-dim.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-simplerenv
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/simplerenv/bridge/groot_n1_5_bs64_mh_m8_econsist"
DATA_DIR="$HOME/multigpu_workspace/data/simplerenv/bridge_orig_lerobot"

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path $DATA_DIR \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config simplerenv_bridge_multi_horizon \
    --embodiment-tag new_embodiment \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-bridge-mh-m8-econsist-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb \
    --use-multi-horizon-loss \
    --multi-horizon-factors 2 4 \
    --multi-horizon-loss-weights 1.0 1.0 \
    --multi-horizon-main-weight 1.0 \
    --aux-grad-scale-to-body 0.1 \
    --aux-loss-warmup-steps 5000 \
    --discrete-action-dims 6 \
    --use-merged-8-head \
    --merged-8-weight 1.0 \
    --use-ensemble-consistency-loss \
    --ensemble-consistency-weight 0.1 \
    --ensemble-consistency-warmup-steps 2000
