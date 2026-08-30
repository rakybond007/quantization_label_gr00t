#!/bin/bash
# Smoke: load each affected ckpt (K3 pyramid, K4 pyramid, K2 raw16+m8) and
# start the inference_service_fair_moe server with --head moe. The 3 ckpts
# all previously crashed inside the head=moe decoder lookup
# (`m4_action_decoder` AttributeError); this confirms the layout-aware fix in
# flow_matching_action_head_fair_moe.py (commit pending) loads, picks via the
# router, and decodes for batch=1 without crashing.
#
# Server "ready" is the success signal; we don't run a sim client here -- a
# crash would happen earlier (during model wiring) than first inference call.
# Each ckpt tested in series, server torn down between tries.
set -u
BASE_DIR="$HOME/multigpu_workspace/Isaac-GR00T"
CONDA_PATH="$HOME/miniconda3"
NVIDIA_PKG_DIR="$CONDA_PATH/envs/gr00t/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_PKG_DIR}/cusparselt/lib:${NVIDIA_PKG_DIR}/cublas/lib:${NVIDIA_PKG_DIR}/cuda_runtime/lib:${NVIDIA_PKG_DIR}/cuda_cupti/lib:${NVIDIA_PKG_DIR}/cudnn/lib:${LD_LIBRARY_PATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
LOG_DIR="$BASE_DIR/analysis/_smoke/head_moe_layouts"
mkdir -p "$LOG_DIR"

# (label, repo, port)
SPEC=(
  "K3_pyramid|HF:prehj/GR00T-N1.5-robocasa-moe-pyramid-K3-raw16-m8-m4-b-only-no-metaq-no-balance-60k|LOCAL:ckpt/robocasa/groot/groot_n1_5_bs64_moe_pyramid_K3_raw16_m8_m4_b_only_no_metaq_no_balance/checkpoint-60000|9711"
  "K4_pyramid|HF:prehj/GR00T-N1.5-robocasa-moe-pyramid-K4-raw16-m8-m4-m2-b-only-no-metaq-no-balance-60k|LOCAL:ckpt/robocasa/groot/groot_n1_5_bs64_moe_pyramid_K4_raw16_m8_m4_m2_b_only_no_metaq_no_balance/checkpoint-60000|9712"
  "K2_raw16_m8|HF:prehj/GR00T-N1.5-robocasa-moe-v1-K2-raw16-merged8-b-only-no-metaq-no-balance-60k||9713"
)

# Try LOCAL ckpt if it exists, else fall back to HF snapshot.
resolve_ckpt() {
  local hf_part="${1#HF:}"
  local local_part="${2#LOCAL:}"
  if [ -n "$2" ] && [ -d "$BASE_DIR/$local_part" ]; then
    echo "$BASE_DIR/$local_part"
  else
    "$CONDA_PATH/envs/gr00t/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$hf_part', repo_type='model'))"
  fi
}

OVERALL=0
for line in "${SPEC[@]}"; do
  IFS='|' read -r LABEL HF LOCAL PORT <<<"$line"
  echo "[i] === $LABEL: resolving ckpt ==="
  CKPT="$(resolve_ckpt "$HF" "$LOCAL")"
  echo "[i] $LABEL ckpt = $CKPT"
  LOG="$LOG_DIR/$LABEL.log"
  "$CONDA_PATH/envs/gr00t/bin/python" -u "$BASE_DIR/scripts/inference_service_fair_moe.py" --server \
      --port "$PORT" --model_path "$CKPT" \
      --data_config single_panda_gripper --embodiment_tag new_embodiment \
      --denoising_steps 4 --head moe --discrete-action-dims 6 11 \
      > "$LOG" 2>&1 &
  SPID=$!
  READY=0
  for i in $(seq 1 36); do  # up to 3 min
      if grep -q "Server is ready" "$LOG" 2>/dev/null; then READY=1; break; fi
      if ! kill -0 "$SPID" 2>/dev/null; then break; fi
      sleep 5
  done
  if [ "$READY" -eq 1 ]; then
      echo "[OK] $LABEL: head=moe server ready"
  else
      echo "[FAIL] $LABEL: server did not become ready"
      tail -20 "$LOG"
      OVERALL=1
  fi
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done
if [ $OVERALL -eq 0 ]; then echo "HEAD_MOE_SMOKE PASS"; else echo "HEAD_MOE_SMOKE FAIL"; fi
