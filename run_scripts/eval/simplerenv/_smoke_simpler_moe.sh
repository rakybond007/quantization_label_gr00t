#!/usr/bin/env bash
# Local SimplerEnv MoE eval smoke (mirrors RLDX-1 run_scripts/eval/simpler/eval_simpler.sh):
#   server (gr00t conda) holds the GPU; client (simpler_uv venv) runs SAPIEN env.
#   No LD_LIBRARY_PATH / VK_* overrides — keep the env clean like RLDX-1.
#
# Usage: bash _smoke_simpler_moe.sh [bridge|fractal]
set -uo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

TARGET="${1:?need: bridge | fractal}"

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
SIMPLER_PY="$BASE_DIR/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
PORT=${PORT:-9491}

case "$TARGET" in
    bridge)
        CKPT="$BASE_DIR/ckpt/simplerenv/bridge/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
        DATA_CFG="simplerenv_bridge"
        ENV_NAME="simpler_env_widowx/widowx_spoon_on_towel"
        ;;
    fractal)
        CKPT="$BASE_DIR/ckpt/simplerenv/fractal/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
        DATA_CFG="simplerenv_fractal"
        ENV_NAME="simpler_env_google/google_robot_pick_coke_can"
        ;;
    *) echo "[ERROR] unknown target $TARGET"; exit 1 ;;
esac

OUT="$BASE_DIR/output/simplerenv/_smoke_${TARGET}"
mkdir -p "$OUT"

echo "[$(date '+%T')] === SMOKE: simpler $TARGET ($ENV_NAME) ==="

"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_simpler.py" \
    --port "$PORT" \
    --model_path "$CKPT" \
    --data_config "$DATA_CFG" \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head moe \
    --discrete-action-dims 6 \
    > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true' EXIT INT TERM

sleep 30

CMD=("$SIMPLER_PY" -u "$BASE_DIR/scripts/simplerenv_service_moe.py"
    --port "$PORT" --host localhost
    --env_name "$ENV_NAME"
    --video_dir "$OUT"
    --seed 42
    --n_episodes 3
    --max_episode_steps 300)

if command -v xvfb-run &>/dev/null; then
    xvfb-run -a "${CMD[@]}" 2>&1 | tee "$OUT/eval.log"
else
    MUJOCO_GL=egl "${CMD[@]}" 2>&1 | tee "$OUT/eval.log"
fi

kill "$SPID" 2>/dev/null || true

succ=$(grep "is_success:" "$OUT/prediction.txt" 2>/dev/null | tail -1)
picks=$(grep "router_picks:" "$OUT/prediction.txt" 2>/dev/null | tail -1)
err=$(grep -cE "Traceback|Error:" "$OUT/eval.log" 2>/dev/null)
echo "[$(date '+%T')] $TARGET → $succ | $picks | errors=$err"
