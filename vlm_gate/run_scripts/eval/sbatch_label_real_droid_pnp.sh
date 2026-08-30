#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=label_real_droid_pnp_teleop_101eps_cosmos_judge_robocasa_evolved_prompt_full
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --time=6:00:00
#SBATCH --output=out/%j-label_real_droid.out
#SBATCH --error=out/%j-label_real_droid.err

# Label the MoSS real DROID pnp_objects teleop dataset (101 eps, 2 cams, 10fps)
# with the cosmos judge + robocasa-evolved v5 prompt. every=4 (~0.4s cadence).
set -u
: "${MODEL_OUTPUT_DIR:?}"
BASE=$HOME/quantization_agent_workspace/vlm_gate
DS=/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects
CV=$HOME/quantization_agent_workspace/cosmos_judge_venv
mkdir -p out "$MODEL_OUTPUT_DIR/labels"
cleanup(){ kill ${JP:-} 2>/dev/null; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=$BASE/scripts \
  $CV/bin/python -u $BASE/scripts/vlm_gate_cosmos.py --serve --model nvidia/Cosmos3-Nano \
  --port 8231 --host 127.0.0.1 > "$MODEL_OUTPUT_DIR/labels/judge_real.log" 2>&1 &
JP=$!
for i in $(seq 1 120); do grep -q "JUDGE READY" "$MODEL_OUTPUT_DIR/labels/judge_real.log" 2>/dev/null && break; sleep 5; done
grep -q "JUDGE READY" "$MODEL_OUTPUT_DIR/labels/judge_real.log" || exit 1
$HOME/miniconda3/envs/quant_gate_eval/bin/python -u $BASE/scripts/label_gate_dataset_real.py \
  --dataset $DS --judge-url http://127.0.0.1:8231 \
  --guidance @$BASE/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt \
  --every 4 \
  --out "$MODEL_OUTPUT_DIR/labels/real_droid_pnp_cosmos_rcprompt.parquet"
