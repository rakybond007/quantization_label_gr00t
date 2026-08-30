#!/bin/bash
# Smoke: robocasa moe3_per_expert (local ckpt) with --moe-stochastic via base inference_service.py.
set -u
TASK="TurnSinkSpout"
PORT=9913
N_EP=3
CKPT="/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_moe3_per_expert_body/checkpoint-60000"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
OUT="$BASE_DIR/analysis/_smoke/_stochastic_robocasa_moe3/$TASK"
mkdir -p "$OUT"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

SERVER_LOG="$OUT/server.log"
PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
    --moe-stochastic \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null" EXIT INT TERM
READY=0
for i in $(seq 1 60); do
    grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null && READY=1 && break
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -25 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
[ "$READY" -ne 1 ] && { echo "[ERR] server not ready"; tail -25 "$SERVER_LOG"; exit 1; }
grep -E "MoE stochastic|inference head" "$SERVER_LOG"

PYTHONUNBUFFERED=1 "$CONDA_PATH/envs/robocasa_gr00t/bin/python" -u \
    "$BASE_DIR/scripts/robocasa_service_moe.py" \
    --port $PORT --host localhost \
    --env_name "$TASK" \
    --video_dir "$OUT" --seed 42 --n_episodes $N_EP \
    --max_episode_steps 1500 --generative_textures \
    >& "$OUT/eval.log"
RC=$?
kill $SPID 2>/dev/null
echo "=== success ==="; grep "is_success" "$OUT/prediction.txt" 2>/dev/null | head
echo "SMOKE_MOE3_STOCH_DONE rc=$RC"
exit $RC
