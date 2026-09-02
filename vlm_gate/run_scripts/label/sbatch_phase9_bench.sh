#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=bench_robocasa_phase9_judge_batch_size_throughput_measurement
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=01:00:00
#SBATCH --output=out/%j-phase9_bench.out
#SBATCH --error=out/%j-phase9_bench.err
#SBATCH --comment="Measure frames/second at batch 1,4,8,16 before committing ~1000 GPU-hours."
set -u
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
PORT=$((11700 + SLURM_JOB_ID % 200))
LOG="/tmp/judge_bench_${SLURM_JOB_ID}.log"
"$JUDGE_PY" -u scripts/vlm_gate_cosmos.py --serve --port "$PORT" > "$LOG" 2>&1 &
JUDGE=$!
trap 'kill $JUDGE 2>/dev/null' EXIT
for _ in $(seq 1 90); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "judge failed"; tail -20 "$LOG"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader
"$RUN_PY" -u bench.py "$PORT"
