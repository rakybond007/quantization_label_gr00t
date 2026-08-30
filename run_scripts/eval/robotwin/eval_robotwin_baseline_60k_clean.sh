#!/bin/bash
#SBATCH --job-name=eval_robotwin_baseline_clean50_vanilla_60k_demo_clean_50t_50ep_arr10
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --array=0-9
#SBATCH --exclude=worker-node1000,worker-node1001,worker-node1002
#SBATCH --output=out/%A_%a-eval_robotwin_baseline_60k_clean.out
#SBATCH --error=out/%A_%a-eval_robotwin_baseline_60k_clean.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="RoboTwin2.0 baseline ckpt-60000 eval, demo_clean (Easy), 50 task × 50 ep × array=0-9 (5 task per array)."

set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA="$HOME/miniconda3"
RT="$HOME/multigpu_workspace/external_dependencies/RoboTwin"

CKPT="$BASE_DIR/ckpt/robotwin/groot_n1_5_bs64_baseline_clean50/checkpoint-60000"
TASK_CFG=demo_clean
CKPT_SETTING=baseline_60k
PORT=$((5800 + SLURM_ARRAY_TASK_ID))
OUT="$BASE_DIR/output/robotwin/baseline_60k_clean/arr_${SLURM_ARRAY_TASK_ID}"
mkdir -p out "$OUT"

# 50 RoboTwin 2.0 tasks (envs/*.py, sorted)
TASK_NAMES=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock click_bell dump_bin_bigbin grab_roller
  handover_block handover_mic hanging_mug lift_pot
  move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
  open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
  place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
  place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
  place_dual_shoes place_empty_cup place_fan place_mouse_pad
  place_object_basket place_object_scale place_object_stand place_phone_stand
  place_shoe press_stapler put_bottles_dustbin put_object_cabinet
  rotate_qrcode scan_object shake_bottle shake_bottle_horizontally
  stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two
  stamp_seal turn_switch
)

# Pick 5 tasks for this array (idx 5*K, 5*K+1, ..., 5*K+4)
SELECTED=()
for i in 0 1 2 3 4; do
    IDX=$((SLURM_ARRAY_TASK_ID * 5 + i))
    [ $IDX -lt 50 ] && SELECTED+=("${TASK_NAMES[$IDX]}")
done
echo "[i] arr=$SLURM_ARRAY_TASK_ID  port=$PORT  tasks=${SELECTED[*]}"

# ---- Server (gr00t env, our zmq inference_service) ----
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export PATH="$CONDA/envs/gr00t/bin:$PATH"
cd "$BASE_DIR"
SERVER_LOG="$OUT/server.log"
PYTHONUNBUFFERED=1 "$CONDA/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config robotwin_agilex --embodiment_tag new_embodiment \
    > "$SERVER_LOG" 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; pkill -P $SPID 2>/dev/null" EXIT INT TERM

READY=0
for i in $(seq 1 90); do
    grep -q "Server is ready" "$SERVER_LOG" 2>/dev/null && READY=1 && break
    if ! kill -0 "$SPID" 2>/dev/null; then echo "[ERR] server died"; tail -30 "$SERVER_LOG"; exit 1; fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then echo "[ERR] server not ready"; tail -30 "$SERVER_LOG"; exit 1; fi
echo "[$(date +%T)] server up. running ${#SELECTED[@]} tasks..."

# ---- Eval (robotwin env, sequential per task) ----
cd "$RT"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
for TASK in "${SELECTED[@]}"; do
    TASK_OUT="$OUT/$TASK"; mkdir -p "$TASK_OUT"
    ELOG="$TASK_OUT/eval.log"
    ROBOTWIN_EVAL_SAVE_DIR="$TASK_OUT" PYTHONUNBUFFERED=1 "$CONDA/envs/robotwin/bin/python" -u script/eval_policy.py \
        --config policy/gr00t_zmq/deploy_policy.yml \
        --overrides \
            --task_name $TASK \
            --task_config $TASK_CFG \
            --ckpt_setting $CKPT_SETTING \
            --seed 0 \
            --policy_name gr00t_zmq.deploy_policy \
            --port $PORT \
        > "$ELOG" 2>&1 || echo "[WARN] $TASK eval rc=$?"
    echo "[$(date +%T)] done: $TASK"
done

kill $SPID 2>/dev/null
echo "[$(date +%T)] arr=$SLURM_ARRAY_TASK_ID complete"
