#!/bin/bash
# Verify per-episode seeding: run baseline and MoE for the same 1 ep with
# matching seed and compare the step-0 (post-reset, pre-step) EEF position.
# If seeding/reset is deterministic across runs, the two step-0 EEFs MUST match.
set -u
TASK=TurnOnSinkFaucet
SEED=42

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"

run_baseline() {
    local PORT=10310
    local CKPT_DIR="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
    local OUT="$BASE_DIR/output/robocasa/_smoke_seedcheck/baseline"
    local TRAJ="$OUT/traj"
    rm -rf "$OUT"; mkdir -p "$OUT"
    cd "$BASE_DIR"
    export NO_ALBUMENTATIONS_UPDATE=1
    local NVIDIA="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
    export LD_LIBRARY_PATH="${NVIDIA}/cusparselt/lib:${NVIDIA}/cublas/lib:${NVIDIA}/cuda_runtime/lib:${NVIDIA}/cuda_cupti/lib:${NVIDIA}/cudnn/lib:${LD_LIBRARY_PATH:-}"
    "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
        --port $PORT --model_path "$CKPT_DIR" \
        --data_config single_panda_gripper --embodiment_tag new_embodiment \
        --denoising_steps 4 --head main > "$OUT/server.log" 2>&1 &
    local SPID=$!
    for i in $(seq 1 60); do
        grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && break
        sleep 5
    done
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
    "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_trajlog.py" --client \
        --port $PORT --host localhost \
        --env_name "$TASK" --video_dir "$OUT" --trajlog_dir "$TRAJ" \
        --seed $SEED --n_episodes 1 --max_episode_steps 200 --generative_textures \
        > "$OUT/eval.log" 2>&1
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    echo "[baseline] step0:"
    head -1 "$TRAJ/traj_ep00.jsonl"
}

run_moe() {
    local PORT=10311
    local HF_REPO=prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k
    local OUT="$BASE_DIR/output/robocasa/_smoke_seedcheck/moe"
    local TRAJ="$OUT/traj"
    rm -rf "$OUT"; mkdir -p "$OUT"
    cd "$BASE_DIR"
    local CKPT
    CKPT=$("$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")
    local NVIDIA="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
    export LD_LIBRARY_PATH="${NVIDIA}/cusparselt/lib:${NVIDIA}/cublas/lib:${NVIDIA}/cuda_runtime/lib:${NVIDIA}/cuda_cupti/lib:${NVIDIA}/cudnn/lib:${LD_LIBRARY_PATH:-}"
    "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
        --port $PORT --model_path "$CKPT" \
        --data_config single_panda_gripper --embodiment_tag new_embodiment \
        --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
        --moe-stochastic --moe-confidence-threshold 0.7 \
        > "$OUT/server.log" 2>&1 &
    local SPID=$!
    for i in $(seq 1 60); do
        grep -q "Server is ready" "$OUT/server.log" 2>/dev/null && break
        sleep 5
    done
    PYTHONUNBUFFERED=1 \
    "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_moe_trajlog.py" \
        --port $PORT --host localhost \
        --env_name "$TASK" --video_dir "$OUT" --trajlog_dir "$TRAJ" \
        --seed $SEED --n_episodes 1 --max_episode_steps 200 --generative_textures \
        --no_record_video \
        > "$OUT/eval.log" 2>&1
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    echo "[MoE] step0:"
    head -1 "$TRAJ/traj_ep00.jsonl"
}

echo "=== baseline run ==="; run_baseline
echo "=== MoE run ===";      run_moe
echo "=== diff check ==="
python3 - <<'PY'
import json, pathlib
b = json.loads(open("/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/output/robocasa/_smoke_seedcheck/baseline/traj/traj_ep00.jsonl").readline())
m = json.loads(open("/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/output/robocasa/_smoke_seedcheck/moe/traj/traj_ep00.jsonl").readline())
print(f"baseline step0 eef_abs: {b['eef_abs']}")
print(f"MoE      step0 eef_abs: {m['eef_abs']}")
import math
d = math.sqrt(sum((a-b)**2 for a,b in zip(b['eef_abs'], m['eef_abs'])))
print(f"L2 diff = {d:.6f}  (should be ~0 if seeding works)")
PY
echo "SMOKE_SEEDCHECK_DONE"
