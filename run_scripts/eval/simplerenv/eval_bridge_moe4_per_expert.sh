#!/bin/bash
#SBATCH --job-name=eval_simplerenv_bridge_moe4_per_expert_60k_chkpt_widowx_full_4tasks_50ep
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --array=0-3
#SBATCH --output=out/%A_%a-eval_simpler_bridge_moe4.out
#SBATCH --error=out/%A_%a-eval_simpler_bridge_moe4.err
#SBATCH --time=1-00:00:00
#SBATCH --comment="Eval SimplerEnv Bridge MoE per_expert (msgpack server + RLDX-1-style client + SAPIEN GPU pin)"

# 4 widowx tasks → 4 array jobs. server (gr00t conda, GPU 0) + client (simpler_uv, SAPIEN GPU 1).
set -uo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

PORT=$((9400 + SLURM_ARRAY_TASK_ID))
N_EPISODES=50

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/simplerenv/bridge/groot_n1_5_bs64_moe4_per_expert_body/checkpoint-60000"
OUTPUT_BASE="$BASE_DIR/output/simplerenv/bridge_moe4_per_expert"
SIMPLER_PYTHON="$BASE_DIR/gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

TASK_NAMES=(
  "widowx_spoon_on_towel"
  "widowx_carrot_on_plate"
  "widowx_put_eggplant_in_basket"
  "widowx_stack_cube"
)
TASK="${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}"
ENV_NAME="simpler_env_widowx/$TASK"
ODIR="$OUTPUT_BASE/$TASK"
mkdir -p "$ODIR"

echo "[i] Array $SLURM_ARRAY_TASK_ID | task=$TASK | port=$PORT"

# msgpack-based server (no torch on the client side).
"$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_simpler.py" \
    --port "$PORT" \
    --model_path "$CKPT" \
    --data_config simplerenv_bridge \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head moe \
    --discrete-action-dims 6 \
    > "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>&1 &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true' EXIT INT TERM

sleep 60
for i in $(seq 1 60); do
    grep -q "Server is ready" "$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log" 2>/dev/null && break
    sleep 5
done

CMD=("$SIMPLER_PYTHON" -u "$BASE_DIR/scripts/simplerenv_service_moe.py"
    --port "$PORT" --host localhost
    --env_name "$ENV_NAME"
    --video_dir "$ODIR"
    --seed 42
    --n_episodes "$N_EPISODES"
    --max_episode_steps 300)

if command -v xvfb-run &>/dev/null; then
    xvfb-run -a "${CMD[@]}" >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log"
else
    MUJOCO_GL=egl "${CMD[@]}" >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log"
fi

kill "$SPID" 2>/dev/null || true
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
