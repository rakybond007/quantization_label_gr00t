#!/bin/bash
#SBATCH --job-name=finetune_gr00t_n17_robocasa_kitchen_quantization_confidence_gate_joint
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.err
set -u
WS="$HOME/quantization_agent_workspace"
cd "$WS/Isaac-GR00T-n17"

# n1.7 은 Qwen3-VL 백본이라 transformers 4.57.3 이 필요하다. 평가 스택의 4.51.3 고정을
# 깨지 않도록 오버레이로만 얹는다 (numpy 는 환경 것을 유지 — 섞이면 조용히 깨진다).
export PYTHONPATH="$WS/pylibs/tf4573:$WS/Isaac-GR00T-n17${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="$(cat "$WS/hf_token.txt" | tr -d '\n\r ')"
# n1.7 기본 백엔드 torchcodec 은 이 클러스터에 없다. decord 로 동작 확인됨.
export VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"
# launch_finetune 은 백본을 전부 얼린다. 게이트가 레이어를 형성하려면 상위층을 풀어야 하고,
# 베이스라인도 같은 값을 줘야 게이트 유무만 변인이 된다.
export TUNE_TOP_LLM_LAYERS="${TUNE_TOP_LLM_LAYERS:-4}"

# GATE_LABELS 가 있으면 게이트가 붙고, 없으면 게이트 없는 베이스라인이 된다.
# 같은 스크립트로 둘 다 뽑아야 비교가 성립한다.
RUN="${RUN:?RUN=gate 또는 RUN=baseline 을 지정하세요}"
if [ "$RUN" = gate ]; then
  export GATE_LABELS="$WS/assets/labels/robocasa/v6b_phase5_1call_full.parquet"
  export GATE_LAYER="${GATE_LAYER:-14}"          # 액션 탭 16 아래 — 레이어 12·13 을 형성
  export GATE_LOSS_WEIGHT="${GATE_LOSS_WEIGHT:-1.0}"
else
  unset GATE_LABELS
fi

# 산출물은 홈에 둔다 — unified-checkpoints 는 90 일 후 삭제된다.
# OUT_SUFFIX 로 같은 RUN 의 변형을 구분한다 (예: 상위층 unfreeze 를 맞춘 대조군).
# 접미사가 없으면 기존 체크포인트를 덮어쓴다.
OUT="$WS/assets/checkpoints/n17_robocasa_${RUN}${OUT_SUFFIX:-}"
mkdir -p "$OUT"

# 해결된 설정을 눈에 띄게 남긴다. 앞선 실행들은 제출할 때 MAX_STEPS 를 덮어써
# 10000/20000 스텝만 돌았는데 잡 이름은 60k 였고, 그 사실이 어디에도 안 남아
# 비교가 조용히 무효가 되었다. N1.5 기준은 60000 스텝 · 글로벌 배치 64 다.
_MS="${MAX_STEPS:-60000}"; _BS="${BS:-64}"
echo "=========================================================="
echo "[cfg] max_steps=$_MS  global_batch=$_BS  (N1.5 기준 60000 / 64)"
if [ "$_MS" != "60000" ] || [ "$_BS" != "64" ]; then
  echo "[cfg] 경고: N1.5 기준과 다르다. 이 실행은 matched 비교에 쓸 수 없다."
fi
echo "=========================================================="
mkdir -p "$OUT" 2>/dev/null || true
echo "max_steps=$_MS global_batch=$_BS gate_labels=${GATE_LABELS:-none} $(date -Iseconds)" \
  >> "${OUT:-.}/run_settings.txt" 2>/dev/null || true

# 주석은 명령 밖에 둔다. 백슬래시로 이어진 줄 사이에 # 을 넣으면 그 줄이 그대로
# 이어붙어 뒤의 인자가 전부 주석 처리된다 — 실제로 --max-steps 와
# --global-batch-size 가 그렇게 사라져 모든 실행이 기본값으로 돌았다.
# N1.5 비교 대상은 배치 64 · 60k 스텝(3.84M 샘플)이다.
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$WS/assets/datasets/robocasa_n17_mirror" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$WS/vlm_gate/n17/robocasa_modality_config.py" \
  --num-gpus 1 \
  --output-dir "$OUT" \
  --max-steps "${MAX_STEPS:-60000}" \
  --global-batch-size "${BS:-64}" \
  --dataloader-num-workers 4

python - "$OUT" "$RUN" <<'PY' > "$OUT/summary.json"
import json, os, sys, glob
O, R = sys.argv[1], sys.argv[2]
ck = sorted(glob.glob(os.path.join(O, "checkpoint-*")))
print(json.dumps({"run": R, "out": O, "checkpoints": [os.path.basename(c) for c in ck]},
                 ensure_ascii=False))
PY
