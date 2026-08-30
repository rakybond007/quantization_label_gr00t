#!/bin/bash
#SBATCH --job-name=groot_n1_5_gr1_tabletop_base
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --comment="GR00T N1.5 finetune on RoboCasa GR-1 Tabletop (24 tasks, baseline, no multi-horizon)"
#SBATCH --partition=batch
#SBATCH --output=out/%j-groot_n1_5_gr1_tabletop_base.out
#SBATCH --error=out/%j-groot_n1_5_gr1_tabletop_base.err

# Baseline training on RoboCasa GR-1 Tabletop (24 tasks).
# Dataset: nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim (mirrored locally).
#   Each task has 1000 teleop demos at 20 fps with a single ego_view camera.
#   Action layout (29 dims selected): left_arm(7) + right_arm(7) + left_hand(6)
#   + right_hand(6) + waist(3). Hands are continuous joint positions (no
#   discrete dims). State uses sin/cos transform (FourierGr1ArmsOnly parent).
# Global batch = 2 * 32 = 64.

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

export WANDB_PROJECT=GR00T-gr1-tabletop
mkdir -p out

CKPT_DIR="$BASE_DIR/ckpt/gr1_tabletop/groot/groot_n1_5_bs64_baseline"
DATA_ROOT="/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"

# All 24 GR-1 ArmsAndWaistFourierHands tasks (1000 demos each).
DATASETS=(
  "$DATA_ROOT/gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000"
  "$DATA_ROOT/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000"
)

source $CONDA_PATH/bin/activate gr00t

python $BASE_DIR/scripts/gr00t_finetune.py \
    --dataset-path "${DATASETS[@]}" \
    --output-dir $CKPT_DIR \
    --dataloader-num-workers 32 \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag gr1 \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --run-name GR00T-N1.5-gr1tabletop-baseline-fromPT-bs64 \
    --batch-size 32 \
    --num-gpus 2 \
    --max-steps 60000 \
    --save-steps 10000 \
    --report-to wandb
