#!/bin/bash
# ============================================================================
# DRAFT — judge backend switch for the VLM-gate eval run_scripts.
# NOT wired into any existing run_script yet. This is the snippet to splice into
# eval_robocasa_vlm_gate_*.sh (and self_evolve_loop*.sh) to choose between the
# Gemma judge (vlm_gate.py, env vlm_judge) and the Cosmos judge
# (vlm_gate_cosmos.py + a vllm serve, env cosmos_judge).
#
# Usage in the parent script:  JUDGE_BACKEND=cosmos sbatch eval_....sh
#                              (default gemma -> unchanged behaviour)
# ============================================================================
set -u
CONDA_PATH="$HOME/miniconda3"
BASE_DIR="$HOME/quantization_agent_workspace/vlm_gate"
: "${SLURM_ARRAY_TASK_ID:=0}"
JUDGE_PORT=$((8300 + SLURM_ARRAY_TASK_ID))
JUDGE_LOG="${JUDGE_LOG:-$BASE_DIR/output/judge-$SLURM_ARRAY_TASK_ID.log}"
JUDGE_BACKEND="${JUDGE_BACKEND:-gemma}"   # gemma | cosmos

case "$JUDGE_BACKEND" in
  gemma)
    # ---- EXISTING path (unchanged): Gemma4-12B via transformers, env vlm_judge.
    JUDGE_MODEL="google/gemma-4-12b-it"
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "$CONDA_PATH/envs/vlm_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate.py" --serve \
        --model "$JUDGE_MODEL" --port "$JUDGE_PORT" --host 127.0.0.1 \
        > "$JUDGE_LOG" 2>&1 &
    JPID=$!
    JUDGE_READY_PAT="JUDGE READY"
    ;;

  cosmos)
    # ---- NEW path: nvidia/Cosmos3-Nano via a local vLLM OpenAI server +
    #      the Cosmos gate shim (vlm_gate_cosmos.py), env cosmos_judge.
    JUDGE_MODEL="nvidia/Cosmos3-Nano"
    VLLM_PORT=$((9000 + SLURM_ARRAY_TASK_ID))
    VLLM_LOG="${VLLM_LOG:-$BASE_DIR/output/vllm-$SLURM_ARRAY_TASK_ID.log}"

    # (1) vLLM serves the 16B reasoner on GPU1 (OpenAI-compatible). bf16 only.
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
      "$CONDA_PATH/envs/cosmos_judge/bin/vllm" serve "$JUDGE_MODEL" \
        --hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}' \
        --tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling \
        --allowed-local-media-path / --media-io-kwargs '{"video": {"num_frames": -1}}' \
        --port "$VLLM_PORT" \
        > "$VLLM_LOG" 2>&1 &
    VPID=$!
    # wait for vLLM to be ready before the shim starts taking requests
    for i in $(seq 1 240); do
      grep -qE "Application startup complete|Uvicorn running" "$VLLM_LOG" 2>/dev/null && break
      kill -0 "$VPID" 2>/dev/null || { echo "[ERR] vLLM died"; tail -40 "$VLLM_LOG"; exit 1; }
      sleep 5
    done

    # (2) thin gate shim exposes POST /judge on JUDGE_PORT (same contract as Gemma)
    PYTHONUNBUFFERED=1 PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}" \
      "$CONDA_PATH/envs/cosmos_judge/bin/python" -u "$BASE_DIR/scripts/vlm_gate_cosmos.py" --serve \
        --model "$JUDGE_MODEL" --port "$JUDGE_PORT" --host 127.0.0.1 \
        --vllm-url "http://127.0.0.1:$VLLM_PORT" --seed 42 \
        > "$JUDGE_LOG" 2>&1 &
    JPID=$!
    JUDGE_READY_PAT="JUDGE READY"
    ;;

  *)
    echo "[ERR] unknown JUDGE_BACKEND=$JUDGE_BACKEND (use gemma|cosmos)"; exit 1
    ;;
esac

# Downstream wait_ready/cleanup is unchanged — both backends print "JUDGE READY"
# on JUDGE_PORT and the eval client still uses --judge-url http://127.0.0.1:$JUDGE_PORT.
# (For cosmos, remember to also kill $VPID in the cleanup() trap.)
