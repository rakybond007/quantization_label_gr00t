#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_robocasa_moe_v1_vlm_gate_router_bias_v9_24tasks_50ep_arr8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_robocasa_moe_vlm_router.out
#SBATCH --error=out/%A_%a-eval_robocasa_moe_vlm_router.err
#SBATCH --comment="MoE-v1 VLA + VLM-gate router bias (conf>=tau -> +bias on m8/m4 logits). GPU0=MoE serve, GPU1=Gemma4 judge. 24 tasks x 50 ep."

# Per array job: GPU0 = MoE policy server, GPU1 = Gemma4 VLM judge. The 3 robocasa
# tasks for this id run as parallel clients; each chunk, the judge's confidence
# (>= TAU) adds BIAS to the router logits of the compressed decoders (m8/m4).
set -u
HF_REPO="prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k"
JUDGE_MODEL="google/gemma-4-12b-it"
: "${SLURM_ARRAY_TASK_ID:=0}"; : "${SLURM_ARRAY_JOB_ID:=$$}"
POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((10000 + POFF + SLURM_ARRAY_TASK_ID))
JUDGE_PORT=$((20000 + POFF + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"; MAX_STEPS="${MAX_STEPS:-1500}"
TAU="${TAU:-0.5}"; BIAS="${BIAS:-2.0}"; BIAS_MODE="${BIAS_MODE:-onesided}"
# MoE routing mode at inference. MOE_CONF>0 enables the paper's confidence-gated
# stochastic routing (max prob >= tau_c -> argmax, else sample). 0 = pure argmax.
MOE_CONF="${MOE_CONF:-0}"
MOE_FLAGS=""
if [ "$(echo "$MOE_CONF > 0" | bc -l 2>/dev/null)" = "1" ]; then
  MOE_FLAGS="--moe-stochastic --moe-confidence-threshold $MOE_CONF"
fi

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"; CONDA_PATH="$HOME/miniconda3"
GUIDANCE_FILE="${GUIDANCE_FILE:-$BASE_DIR/analysis/_evolver/guidance_versions/v9_manual.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/moe_vlm_router_v9}"
mkdir -p out "$OUTPUT_BASE"; cd "$BASE_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

CKPT=$("$CONDA_PATH/envs/quant_gate/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$HF_REPO', repo_type='model'))")

cleanup() { kill ${SPID:-} ${JPID:-} 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT

# GPU0: MoE policy server
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
    --port $PORT --model_path "$CKPT" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps 4 --head moe --discrete-action-dims 6 11 $MOE_FLAGS > "$SERVER_LOG" 2>&1 &
SPID=$!

# GPU1: Gemma4 VLM judge
JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
    --model "$JUDGE_MODEL" --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
JPID=$!

wait_ready() { for i in $(seq 1 150); do grep -qE "$2" "$1" 2>/dev/null && return 0; kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }; sleep 5; done; echo "[ERR] $4 not ready"; tail -40 "$1"; return 1; }
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "MoE server" || exit 1
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "VLM judge" || exit 1
echo "[i] both servers ready (array $SLURM_ARRAY_TASK_ID)"

TASK_NAMES=(
  "TurnSinkSpout" "TurnOnStove" "TurnOnSinkFaucet" "TurnOnMicrowave"
  "TurnOffStove" "TurnOffSinkFaucet" "TurnOffMicrowave" "PnPStoveToCounter"
  "PnPSinkToCounter" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPCounterToSink"
  "PnPCounterToMicrowave" "PnPCounterToCab" "PnPCabToCounter" "OpenSingleDoor"
  "OpenDrawer" "OpenDoubleDoor" "CoffeeSetupMug" "CoffeeServeMug"
  "CoffeePressButton" "CloseSingleDoor" "CloseDrawer" "CloseDoubleDoor"
)
SELECTED=()
[ $SLURM_ARRAY_TASK_ID -lt 8 ] && SELECTED+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
[ $((SLURM_ARRAY_TASK_ID + 8)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
[ $((SLURM_ARRAY_TASK_ID + 16)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")

MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"; mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
        "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_moe.py" \
        --port $PORT --host localhost --env_name "$TASK" \
        --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
        --max_episode_steps $MAX_STEPS --generative_textures \
        --judge-url "http://127.0.0.1:$JUDGE_PORT" --judge-threshold $TAU --bias $BIAS \
        --bias-mode $BIAS_MODE --judge-guidance "@$GUIDANCE_FILE" \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] Array $SLURM_ARRAY_TASK_ID done."
