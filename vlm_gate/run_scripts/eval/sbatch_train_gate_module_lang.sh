#!/bin/bash
#SBATCH --job-name=train_gate_module_A_lang_frozen_text_encoder_instruction_conditioning_cosmos_full_262k
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --time=24:00:00
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --output=out/%j-train_gate_lang.out
#SBATCH --error=out/%j-train_gate_lang.err
# Env: VARIANT=full|holdout; MODEL_OUTPUT_DIR required by cluster policy.
set -u
: "${VARIANT:?set VARIANT=full|holdout}"
: "${MODEL_OUTPUT_DIR:?MODEL_OUTPUT_DIR required (must contain \$USER)}"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
CACHE=/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_frame_cache_robocasa
HOLD=""
[ "$VARIANT" = holdout ] && HOLD="OpenDrawer,CoffeeSetupMug,PnPCounterToMicrowave,TurnOffStove"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module_lang.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels /rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet \
  --cache-dir "$CACHE" \
  --out-dir "$MODEL_OUTPUT_DIR" --epochs 30 --bs 256 --lr 3e-4 --num-workers 8 \
  --wandb "gateA_lang_${VARIANT}_cosmos_full" \
  --task-emb "$CACHE/task_embeddings.npz" \
  ${HOLD:+--holdout-tasks "$HOLD"}
