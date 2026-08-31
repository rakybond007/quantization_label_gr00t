#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_robocasa_base_gr00t_n1_5_vlm_gated_quantize_k2_24t_arr8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_robocasa_gated.out
#SBATCH --error=out/%A_%a-eval_robocasa_gated.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="robocasa base GR00T-N1.5 @60k + VLM-gated K2 quantization. GPU0=GR00T serve, GPU1=judge (gemma|cosmos). 24 tasks, 3/job. Optional TTL skip policy."

# Unified judge-backend variant of eval_robocasa_cosmos_tau0p5.sh /
# eval_robocasa_vlm_gate_gemma4_tau0p5_mv_guide.sh:
#   JUDGE_BACKEND=gemma|cosmos (default gemma)
#   GATE_TTL_MAX>0 enables the TTL skip policy in the compress client.
set -u
JUDGE_BACKEND="${JUDGE_BACKEND:-gemma}"
CKPT_DIR="$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
COSMOS_VENV="$HOME/quantization_agent_workspace/cosmos_judge_venv"
: "${SLURM_ARRAY_TASK_ID:=0}"
: "${SLURM_ARRAY_JOB_ID:=$$}"
POFF=$(( (SLURM_ARRAY_JOB_ID % 90) * 10 ))
PORT=$((10000 + POFF + SLURM_ARRAY_TASK_ID))
JUDGE_PORT=$((20000 + POFF + SLURM_ARRAY_TASK_ID))
N_EPISODES="${N_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-1500}"
K=2
TAU="${TAU:-0.5}"
K3TAU="${K3TAU:-0}"
CLIP_SCALE="${CLIP_SCALE:-1}"
JUDGE_ACTIONS="${JUDGE_ACTIONS:-0}"
# TTL skip policy (0 = off). Replay-calibrated per-backend defaults.
GATE_TTL_MAX="${GATE_TTL_MAX:-0}"
VARK_BOUND="${VARK_BOUND:-0}"
if [ "$JUDGE_BACKEND" = cosmos ]; then
  GATE_TTL_LO="${GATE_TTL_LO:-0.05}"; GATE_TTL_HI="${GATE_TTL_HI:-0.15}"
else
  GATE_TTL_LO="${GATE_TTL_LO:-0.15}"; GATE_TTL_HI="${GATE_TTL_HI:-0.30}"
fi

BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
GUIDANCE_FILE="${GUIDANCE_FILE:-$BASE_DIR/run_scripts/eval/vlm_gate_guidance.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/${JUDGE_BACKEND}_gated}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

export NO_ALBUMENTATIONS_UPDATE=1
# 정책 서버가 paligemma 프로세서 설정을 HF에서 받으려다 401(게이트드 리포)로 죽는다.
# 체크포인트에 필요한 파일이 다 있으므로 오프라인으로 강제한다.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NV="$CONDA_PATH/envs/quant_gate/lib/python3.10/site-packages/nvidia"
SERVER_LD="${NV}/cusparselt/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_cupti/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}"

cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} ${JPID:-} 2>/dev/null; }
trap cleanup EXIT

# ---------- GPU0: base GR00T policy server (head=main) ----------
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$SERVER_LD" PYTHONUNBUFFERED=1 \
"$CONDA_PATH/envs/quant_gate/bin/python" -u "$BASE_DIR/scripts/inference_service.py" --server \
    --port $PORT --model_path "$CKPT_DIR" \
    --data_config single_panda_gripper --embodiment_tag new_embodiment \
    --denoising_steps ${DENOISE:-4} --head main > "$SERVER_LOG" 2>&1 &
SPID=$!

# ---------- GPU1: judge ----------
JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
if [ "$JUDGE_BACKEND" = moduleB ]; then
  # B-variant student: frozen GR00T backbone + MLP head. Needs its own GPU
  # (backbone forward per judge call), so keep the VLM-style GPU1 slot.
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/module_gate_server_B.py" \
      --ckpt "$MODULE_CKPT" --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
elif [ "$JUDGE_BACKEND" = module ]; then
  # DINOv3 학생은 transformers>=4.56 이 필요하다. 오버레이를 judge 프로세스에만 얹는다
  # (평가 클라이언트는 4.51.3 고정을 유지해야 robocasa 스택이 안 깨진다).
  JUDGE_PP="$BASE_DIR/scripts"
  [ "${GATE_ENCODER:-}" = dinov3s ] && JUDGE_PP="$HOME/quantization_agent_workspace/pylibs/tf4573:$JUDGE_PP"
  # Distilled student gate: MODULE_CKPT (gate_module.pt) required. Shares GPU0
  # with the policy server — the 0.3M CNN is negligible next to GR00T.
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH="$JUDGE_PP" \
  "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/module_gate_server.py" \
      --ckpt "$MODULE_CKPT" \
      --task-emb "${TASK_EMB:-$HOME/quantization_agent_workspace/assets/robocasa_task_embeddings.npz}" \
      --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
elif [ "$JUDGE_BACKEND" = cosmos ]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts" \
  "$COSMOS_VENV/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" --serve \
      --model nvidia/Cosmos3-Nano --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
else
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JUDGE_COMPILE=1 \
  "$CONDA_PATH/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
      --model google/gemma-4-12b-it --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
fi
JPID=$!

wait_ready() { # <log> <pat> <pid> <label>
    for i in $(seq 1 200); do
        grep -qE "$2" "$1" 2>/dev/null && return 0
        kill -0 "$3" 2>/dev/null || { echo "[ERR] $4 died"; tail -40 "$1"; return 1; }
        sleep 5
    done
    echo "[ERR] $4 not ready"; tail -40 "$1"; return 1
}
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "GR00T server" || exit 1
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "judge($JUDGE_BACKEND)" || exit 1
echo "[i] both servers ready (array $SLURM_ARRAY_TASK_ID, judge=$JUDGE_BACKEND)"

# ---------- map array id -> up to 3 tasks ----------
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

# ---------- run the 3 tasks as parallel clients against the shared servers ----------
MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"; mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
        "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_compress.py" \
        --port $PORT --host localhost --env_name "$TASK" \
        --video_dir "$ODIR" --seed 42 --n_episodes $N_EPISODES \
        --max_episode_steps $MAX_STEPS --generative_textures \
        --compress-k $K \
        --judge-url "http://127.0.0.1:$JUDGE_PORT" --judge-threshold $TAU \
        --judge-guidance "@$GUIDANCE_FILE" --gate-subchunk 8 --action-rules ${ACTION_RULES:-0} --gate-k3-threshold $K3TAU --clip-scale $CLIP_SCALE --judge-actions $JUDGE_ACTIONS \
        --gate-ttl-max $GATE_TTL_MAX --vark-bound $VARK_BOUND --gate-ttl-lo $GATE_TTL_LO --gate-ttl-hi $GATE_TTL_HI \
        >& "$ODIR/eval-$SLURM_ARRAY_TASK_ID.log" &
    MAIN_PIDS+=($!)
done
for pid in "${MAIN_PIDS[@]}"; do wait "$pid"; done
echo "[i] Array $SLURM_ARRAY_TASK_ID done (judge=$JUDGE_BACKEND)."
