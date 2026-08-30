#!/bin/bash
# Smoke eval for the multi-horizon trained checkpoint.
# Runs 3 head variants (main / f2 / f4) on a few representative tasks.
# Each variant: starts its own server, runs N episodes per task, kills server.
#
# Run on a 1-GPU node (e.g. via srun in tmux):
#   bash run_scripts/eval/eval_multi_horizon_smoke.sh

set -u
export NO_ALBUMENTATIONS_UPDATE=1

BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
CKPT="$BASE_DIR/ckpt/robocasa/groot/groot_n1_5_bs64_multi_horizon/checkpoint-60000"
PORT=8830
N_EPISODES=10

# Tasks (mid-difficulty, useful for measuring differences across heads)
TASKS=(PnPStoveToCounter TurnOnMicrowave OpenSingleDoor)

OUTPUT_BASE="$BASE_DIR/output/robocasa/multi_horizon_smoke"

cd "$BASE_DIR"

run_eval_for_head() {
    local HEAD=$1
    local LABEL="${HEAD}"
    echo ""
    echo "============================================================"
    echo " HEAD = $HEAD"
    echo "============================================================"

    # Start server with --head <HEAD>
    source "$CONDA_PATH/bin/activate" gr00t
    python "$BASE_DIR/scripts/inference_service.py" --server \
        --port $PORT \
        --model_path "$CKPT" \
        --data_config single_panda_gripper \
        --embodiment_tag new_embodiment \
        --denoising_steps 4 \
        --head "$HEAD" &
    SPID=$!
    echo "[i] Server PID=$SPID for head=$HEAD"
    sleep 90

    # Run client in the dedicated robocasa_gr00t env (has gr00t + robosuite + robocasa).
    source "$CONDA_PATH/bin/activate" robocasa_gr00t
    for TASK in "${TASKS[@]}"; do
        ODIR="$OUTPUT_BASE/$LABEL/$TASK"
        if [ -f "$ODIR/prediction.txt" ] && grep -q "^is_success:" "$ODIR/prediction.txt"; then
            echo "[SKIP] $LABEL/$TASK already done"
            continue
        fi
        mkdir -p "$ODIR"
        echo "[RUN] $LABEL/$TASK"
        python "$BASE_DIR/scripts/robocasa_service.py" --client \
            --port $PORT --host localhost \
            --env_name "$TASK" \
            --video_dir "$ODIR" \
            --seed 42 \
            --n_episodes $N_EPISODES \
            --max_episode_steps 1500 \
            --generative_textures \
            >& "$ODIR/eval.log"
        SR=$(grep "^is_success" "$ODIR/prediction.txt" 2>/dev/null | tail -1 | awk '{print $2}')
        echo "[DONE] $LABEL/$TASK = $SR"
    done

    kill "$SPID" 2>/dev/null
    wait "$SPID" 2>/dev/null
    sleep 3
}

run_eval_for_head main
run_eval_for_head f2
run_eval_for_head f4

# Summary
echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
for HEAD in main f2 f4; do
    echo "--- $HEAD ---"
    for TASK in "${TASKS[@]}"; do
        P="$OUTPUT_BASE/$HEAD/$TASK/prediction.txt"
        if [ -f "$P" ]; then
            SR=$(grep "^is_success" "$P" | tail -1 | awk '{print $2}')
            echo "  $TASK: $SR"
        else
            echo "  $TASK: -"
        fi
    done
done
