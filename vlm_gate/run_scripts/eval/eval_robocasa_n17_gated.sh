#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=eval_robocasa_gr00t_n1_7_vlm_gated_quantize_k2_24t_arr8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --exclude=worker-node100
#SBATCH --array=0-7
#SBATCH --output=out/%A_%a-eval_robocasa_n17_gated.out
#SBATCH --error=out/%A_%a-eval_robocasa_n17_gated.err
#SBATCH --time=2-00:00:00
#SBATCH --comment="robocasa GR00T-N1.7 finetune (gate|baseline) + VLM-gated K2 quantization. GPU0=N1.7 serve, GPU1=judge. 24 tasks, 3/job. Same client / compression / gating as eval_robocasa_gated.sh."

# N1.7 twin of eval_robocasa_gated.sh.  EVERYTHING on the client side is identical
# (same robocasa_service_compress.py, same K, tau, sub-chunk, TTL, guidance) so the
# numbers are comparable to the N1.5 track.  Only the policy server changes:
#
#   N1.5:  scripts/inference_service.py            (torch-pickle wire, quant_gate env)
#   N1.7:  Isaac-GR00T-n17 gr00t/eval/run_gr00t_server.py --use-sim-policy-wrapper
#          (msgpack/zmq wire, quant_gate_eval env + transformers 4.57.3 OVERLAY)
#
# ENV SEPARATION — the whole point of this file:
#   * The SERVER gets PYTHONPATH="$WS/pylibs/tf4573:$WS/Isaac-GR00T-n17".  The overlay
#     supplies transformers 4.57.3 and deliberately contains no numpy and no torch,
#     so the process keeps the env's numpy 1.23.5 / torch.
#   * The CLIENT runs in quant_gate_eval WITHOUT the overlay (transformers stays at
#     4.51.3, numpy at 1.23.5 for robocasa) and never imports the N1.7 gr00t package:
#     it speaks the msgpack/zmq protocol directly via scripts/n17_policy_client.py.
#   Mixing the two has broken evaluation for days before.  Do not "simplify" this.
#
# RUN=gate      -> assets/checkpoints/n17_robocasa_gate      (quantizability gate head)
# RUN=baseline  -> assets/checkpoints/n17_robocasa_baseline  (no gate)
# JUDGE_BACKEND=gemma|cosmos|module|moduleB  (default gemma), same as the N1.5 script.
#
# NOTE: --judge-url internal is NOT supported on the N1.7 backend yet — the N1.7
# checkpoints emit no "_gate_prob" and Gr00tSimPolicyWrapper drops non-action keys.
set -u
RUN="${RUN:-gate}"
JUDGE_BACKEND="${JUDGE_BACKEND:-gemma}"
WS="$HOME/quantization_agent_workspace"
N17_DIR="$WS/Isaac-GR00T-n17"
OVERLAY="$WS/pylibs/tf4573"
# Top-level checkpoint dir holds the final weights + processor/; override with
# N17_CKPT=".../checkpoint-9000" to score an intermediate step.
CKPT_DIR="${N17_CKPT:-$WS/assets/checkpoints/n17_robocasa_$RUN}"
COSMOS_VENV="$WS/cosmos_judge_venv"
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
GATE_TTL_MAX="${GATE_TTL_MAX:-0}"
VARK_BOUND="${VARK_BOUND:-0}"
if [ "$JUDGE_BACKEND" = cosmos ]; then
  GATE_TTL_LO="${GATE_TTL_LO:-0.05}"; GATE_TTL_HI="${GATE_TTL_HI:-0.15}"
else
  GATE_TTL_LO="${GATE_TTL_LO:-0.15}"; GATE_TTL_HI="${GATE_TTL_HI:-0.30}"
fi

BASE_DIR="$WS/vlm_gate"
CONDA_PATH="$HOME/miniconda3"
GUIDANCE_FILE="${GUIDANCE_FILE:-$BASE_DIR/run_scripts/eval/vlm_gate_guidance.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-$BASE_DIR/output/robocasa/n17_${RUN}_${JUDGE_BACKEND}_gated}"
mkdir -p out "$OUTPUT_BASE"
cd "$BASE_DIR"

[ -d "$CKPT_DIR" ] || { echo "[ERR] checkpoint not found: $CKPT_DIR"; exit 1; }
[ -d "$OVERLAY" ]  || { echo "[ERR] tf4573 overlay not found: $OVERLAY"; exit 1; }

export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1

cleanup(){ kill ${SPID:-} ${JPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} ${JPID:-} 2>/dev/null; }
trap cleanup EXIT

# ---------- GPU0: GR00T-N1.7 policy server (overlay ON) ----------
# --use-sim-policy-wrapper makes the server accept the FLAT robocasa observation
# layout ("video.left_view", "state.gripper_qpos", ...) and reply with flat
# "action.<key>" chunks -- the same surface the N1.5 server exposed.
SERVER_LOG="$OUTPUT_BASE/server-$SLURM_ARRAY_TASK_ID.log"
# transformers 4.57 은 캐시가 있어도 HF API 를 조회한다(토크나이저 패치 경로).
# 서버 프로세스에만 온라인을 열어준다 — 판정기와 클라이언트는 오프라인 유지.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
HF_TOKEN="$(cat "$HOME/quantization_agent_workspace/hf_token.txt" 2>/dev/null | tr -d '\n\r ')" \
PYTHONPATH="$OVERLAY:$N17_DIR" \
"$CONDA_PATH/envs/quant_gate_eval/bin/python" -u "$N17_DIR/gr00t/eval/run_gr00t_server.py" \
    --model-path "$CKPT_DIR" \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda \
    --host 127.0.0.1 --port $PORT \
    --use-sim-policy-wrapper > "$SERVER_LOG" 2>&1 &
SPID=$!

# ---------- GPU1: judge (identical to the N1.5 script) ----------
JUDGE_LOG="$OUTPUT_BASE/judge-$SLURM_ARRAY_TASK_ID.log"
if [ "$JUDGE_BACKEND" = moduleB ]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/module_gate_server_B.py" \
      --ckpt "$MODULE_CKPT" --port $JUDGE_PORT --host 127.0.0.1 > "$JUDGE_LOG" 2>&1 &
elif [ "$JUDGE_BACKEND" = module ]; then
  JUDGE_PP="$BASE_DIR/scripts"
  [ "${GATE_ENCODER:-}" = dinov3s ] && JUDGE_PP="$OVERLAY:$JUDGE_PP"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH="$JUDGE_PP" \
  "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u "$BASE_DIR/scripts/module_gate_server.py" \
      --ckpt "$MODULE_CKPT" \
      --task-emb "${TASK_EMB:-$WS/assets/robocasa_task_embeddings.npz}" \
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
# PolicyServer.run() prints "Server is ready and listening on ..." AFTER bind.
wait_ready "$SERVER_LOG" "Server is ready" "$SPID" "GR00T-N1.7 server" || exit 1
wait_ready "$JUDGE_LOG" "JUDGE READY" "$JPID" "judge($JUDGE_BACKEND)" || exit 1
echo "[i] both servers ready (array $SLURM_ARRAY_TASK_ID, run=$RUN, judge=$JUDGE_BACKEND)"

# ---------- map array id -> up to 3 tasks (identical mapping to the N1.5 script) ----------
TASK_NAMES=(
  "TurnSinkSpout" "TurnOnStove" "TurnOnSinkFaucet" "TurnOnMicrowave"
  "TurnOffStove" "TurnOffSinkFaucet" "TurnOffMicrowave" "PnPStoveToCounter"
  "PnPSinkToCounter" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPCounterToSink"
  "PnPCounterToMicrowave" "PnPCounterToCab" "PnPCabToCounter" "OpenSingleDoor"
  "OpenDrawer" "OpenDoubleDoor" "CoffeeSetupMug" "CoffeeServeMug"
  "CoffeePressButton" "CloseSingleDoor" "CloseDrawer" "CloseDoubleDoor"
)
SELECTED=()
# 스모크용: SMOKE_TASK 를 주면 그 태스크 하나만 돈다.
# (array id 를 24 이상으로 주면 아래 매핑이 전부 탈락해 클라이언트가 하나도 안 돈다 —
#  잡은 조용히 성공으로 끝나므로 알아채기 어렵다.)
if [ -n "${SMOKE_TASK:-}" ]; then
  SELECTED+=("$SMOKE_TASK")
else
[ $SLURM_ARRAY_TASK_ID -lt 8 ] && SELECTED+=("${TASK_NAMES[$SLURM_ARRAY_TASK_ID]}")
[ $((SLURM_ARRAY_TASK_ID + 8)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 8))]}")
[ $((SLURM_ARRAY_TASK_ID + 16)) -lt 24 ] && SELECTED+=("${TASK_NAMES[$((SLURM_ARRAY_TASK_ID + 16))]}")
fi
[ ${#SELECTED[@]} -eq 0 ] && { echo "[ERR] 선택된 태스크가 없다 (array id=$SLURM_ARRAY_TASK_ID)"; exit 1; }

# ---------- run the 3 tasks as parallel clients (overlay OFF) ----------
MAIN_PIDS=()
for TASK in "${SELECTED[@]}"; do
    ODIR="$OUTPUT_BASE/$TASK"; mkdir -p "$ODIR"
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
        "$CONDA_PATH/envs/quant_gate_eval/bin/python" -u \
        "$BASE_DIR/scripts/robocasa_service_compress.py" \
        --policy-backend n17 \
        --port $PORT --host 127.0.0.1 --env_name "$TASK" \
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
echo "[i] Array $SLURM_ARRAY_TASK_ID done (run=$RUN, judge=$JUDGE_BACKEND)."
