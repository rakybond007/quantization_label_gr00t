#!/bin/bash
#SBATCH --job-name=train_small_gate_module_A_9k_teacher_labels_luna_sonnet_cosmos_subset_robocasa
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --time=6:00:00
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --output=out/%j-train_gate_%x.out
#SBATCH --error=out/%j-train_gate_%x.err
# Env: TEACHER in {luna,sonnet,cosmos9k}; MODEL_OUTPUT_DIR required by cluster policy.
set -u
: "${TEACHER:?set TEACHER=luna|sonnet|cosmos9k}"
: "${MODEL_OUTPUT_DIR:?MODEL_OUTPUT_DIR required (must contain \$USER)}"
BASE="$HOME/quantization_agent_workspace/vlm_gate"
cd "$BASE"
CACHE="$BASE/output/_gate_distill/cache9k_local"
TEMB=/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_frame_cache_robocasa/task_embeddings.npz
LAB="$BASE/output/_gate_distill/luna_robocasa_strat/labels_${TEACHER}.parquet"
[ "$TEACHER" = luna ] && LAB="$BASE/output/_gate_distill/luna_robocasa_strat/labels_luna_9k.parquet"
[ "$TEACHER" = sonnet ] && LAB="$BASE/output/_gate_distill/luna_robocasa_strat/labels_sonnet_9k.parquet"
[ "$TEACHER" = cosmos9k ] && LAB="$BASE/output/_gate_distill/luna_robocasa_strat/labels_cosmos9k.parquet"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$LAB" \
  --cache-dir "$CACHE" \
  --out-dir "$MODEL_OUTPUT_DIR" --epochs 30 --bs 256 --lr 3e-4 --num-workers 8 \
  --wandb "gateA_tc_${TEACHER}_9k" \
  --task-emb "$TEMB"
