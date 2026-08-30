#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_dexjoco_base_gr00t_n1_5_vlm_gated_quantize_k2_singlearm_arr6
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --array=0-5
#SBATCH --output=out/%A_%a-eval_dexjoco_gated.out
#SBATCH --error=out/%A_%a-eval_dexjoco_gated.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="DexJoCo single-arm GR00T-N1.5 + gated K2 quantization. GPU0=GR00T serve + MuJoCo EGL client, GPU1=judge. 6 single-arm tasks, 1/job."

# DexJoCo counterpart of eval_robocasa_gated.sh.
#
#   JUDGE_BACKEND=none|gemma|cosmos|module   (default none = naive-K baseline)
#   K=1 -> uncompressed reference rollout.
#
# Process/env split (three separate dependency stacks, no `conda activate`):
#   GR00T policy server : envs/quant_gate      (Isaac-GR00T fork in $WS)
#   DexJoCo eval client : envs/dexjoco         (MuJoCo 3.4 + openpi_client)
#   judge               : envs/vlm_judge | cosmos_judge_venv | envs/quant_gate_eval
set -u

JUDGE_BACKEND="${JUDGE_BACKEND:-none}"
: "${SLURM_ARRAY_TASK_ID:=0}"
: "${SLURM_ARRAY_JOB_ID:=$$}"

TASKS=(water_plant hammer_nail pick_bucket pinch_tongs fold_glasses click_mouse)
TASK="${TASK:-${TASKS[$SLURM_ARRAY_TASK_ID]}}"

POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((10000 + POFF + SLURM_ARRAY_TASK_ID))
JUDGE_PORT=$((20000 + POFF + SLURM_ARRAY_TASK_ID))

N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"
SEED="${SEED:-42}"
K="${K:-2}"
TAU="${TAU:-0.5}"
K3TAU="${K3TAU:-0}"
JUDGE_ACTIONS="${JUDGE_ACTIONS:-0}"
JUDGE_FACTS="${JUDGE_FACTS:-0}"
ACTION_RULES="${ACTION_RULES:-0}"
GATE_SUBCHUNK="${GATE_SUBCHUNK:-0}"
GATE_TTL_MAX="${GATE_TTL_MAX:-0}"
DENOISE="${DENOISE:-4}"
CONFIG_FAMILY="${CONFIG_FAMILY:-rand_obj}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
if [ "$JUDGE_BACKEND" = cosmos ]; then
  GATE_TTL_LO="${GATE_TTL_LO:-0.05}"; GATE_TTL_HI="${GATE_TTL_HI:-0.15}"
else
  GATE_TTL_LO="${GATE_TTL_LO:-0.15}"; GATE_TTL_HI="${GATE_TTL_HI:-0.30}"
fi

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
GROOT_DIR="$HOME/quantization_agent_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
COSMOS_VENV="$HOME/quantization_agent_workspace/cosmos_judge_venv"
# GR00T-N1.5 trained on the six DexJoCo single-arm tasks (multitask baseline).
CKPT_DIR="${CKPT_DIR:-$HOME/multigpu_workspace/Isaac-GR00T/ckpt/dexjoco/groot/groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000}"
# DexJoCo eval yamls (rand_obj family -> camera_mapping {base, wrist}).
DEXJOCO_CONFIG_ROOT="${DEXJOCO_CONFIG_ROOT:-$HOME/multigpu_workspace/external_dependencies/dexjoco/configs}"
GUIDANCE_FILE="${GUIDANCE_FILE:-$BASE_DIR/run_scripts/eval/vlm_gate_guidance.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/dexjoco/${JUDGE_BACKEND}_gated_k${K}}"
ODIR="$OUTPUT_BASE/$TASK"
mkdir -p out "$ODIR"
cd "$BASE_DIR"

export MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-/rlwrld-unified-checkpoints/hojin2/quant_gate_modules}"
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DEXJOCO_CONFIG_ROOT
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} ${JPID:-} 2>/dev/null; }
trap cleanup EXIT

wait_ready() { # <log> <pattern> <pid> <label>
    for i in $(seq 1 200); do
        grep -qE "$2" "$1" 2>/dev/null && return 0
        kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }
        sleep 5
    done
    echo "[ERR] $4 not ready"; tail -40 "$1"; return 1
}

# ---------- GPU0: GR00T policy server (OpenPI websocket adapter) ----------
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$GROOT_DIR/scripts/serve_policy_dexjoco.py" \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config dexjoco_single_arm_multi_horizon --embodiment_tag new_embodiment \
    --denoising_steps $DENOISE --head main > "$SERVER_LOG" 2>&1 &
SPID=$!
wait_ready "$SERVER_LOG" "Creating server|serving" "$SPID" "GR00T server" || exit 1
sleep 10   # websocket_policy_server has no explicit ready banner after bind

# ---------- GPU1: judge ----------
JUDGE_ARG=()
if [ "$JUDGE_BACKEND" != none ]; then
  JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
  case "$JUDGE_BACKEND" in
    gemma)
      CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
      "$CONDA_PATH/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
          --model "${JUDGE_MODEL:-google/gemma-4-12b-it}" --port $JUDGE_PORT \
          --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 & ;;
    cosmos)
      CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
      "$COSMOS_VENV/bin/python" -u "$BASE_DIR/scripts/cosmos_1call_v6.py" --serve \
          --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 & ;;
    module)
      CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
      "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/module_gate_server.py" \
          --ckpt "${GATE_CKPT:?set GATE_CKPT for JUDGE_BACKEND=module}" \
          --task-emb "${GATE_TASK_EMB:?set GATE_TASK_EMB}" \
          --port $JUDGE_PORT --host 127.0.0.1 --tau $TAU > "$JUDGE_LOG" 2>&1 & ;;
    *) echo "[ERR] unknown JUDGE_BACKEND=$JUDGE_BACKEND"; exit 1 ;;
  esac
  JPID=$!
  wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "judge($JUDGE_BACKEND)" || exit 1
  JUDGE_ARG=(--judge-url "http://127.0.0.1:$JUDGE_PORT" --judge-threshold "$TAU"
             --judge-guidance "@$GUIDANCE_FILE" --judge-actions "$JUDGE_ACTIONS"
             --judge-facts "$JUDGE_FACTS" --gate-subchunk "$GATE_SUBCHUNK"
             --gate-k3-threshold "$K3TAU" --gate-ttl-max "$GATE_TTL_MAX"
             --gate-ttl-lo "$GATE_TTL_LO" --gate-ttl-hi "$GATE_TTL_HI")
fi

# ---------- eval client (dexjoco env, MuJoCo EGL on GPU0) ----------
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
"$CONDA_PATH/envs/dexjoco/bin/python" -u "$BASE_DIR/scripts/dexjoco_service_compress.py" \
    --env_name "$TASK" --config-family "$CONFIG_FAMILY" \
    --host 127.0.0.1 --port $PORT \
    --video_dir "$ODIR" --seed $SEED --n_episodes $N_EPISODES \
    --max_episode_steps $MAX_STEPS --save-video $SAVE_VIDEO \
    --compress-k $K --action-rules $ACTION_RULES \
    "${JUDGE_ARG[@]}" \
    >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log"
RC=$?

echo "[i] eval rc=$RC"
tail -20 "$ODIR/prediction.txt" 2>/dev/null
echo "EVAL_DEXJOCO_GATED_DONE task=$TASK rc=$RC"
