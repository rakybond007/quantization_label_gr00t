#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=relabel_allex_with_v2_prompt_and_the_new_measured_task_ceilings_and_floors
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%j-allex_v2_receil.out
#SBATCH --error=out/%j-allex_v2_receil.err
#SBATCH --comment="Re-label allex with the v2 prompt untouched and only the ceilings/floors replaced by the measured ones."
set -u
# 프롬프트는 v2 그대로다. 바뀐 것은 상한과 하한뿐이다:
#   Rotate Box 2.0 -> 1.5, Bring Object 2.0 -> 3.0(박스) / BRING_SOFT 2.5 -> 2.0(봉투),
#   Pass 3.0 유지, Rotate PolyBag 2.5 유지, 하한은 Rotate Box 1.0 나머지 2.0.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/allex_v2_receil
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
export ALLEX_OUT="$PWD/output/allex_v2_receil"
mkdir -p "$ALLEX_OUT" out
PORT=$((13100 + SLURM_JOB_ID % 400))

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > "$ALLEX_OUT/judge.log" 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$ALLEX_OUT/judge.log" && break; sleep 10; done
grep -q "JUDGE READY" "$ALLEX_OUT/judge.log" || { echo "판정기 안 뜸"; tail -20 "$ALLEX_OUT/judge.log"; exit 1; }
echo "판정기 준비됨, 라벨링 시작"

TAG=full "$RUN_PY" -u scripts/allex_v2_label.py "$PORT" 0 1 || exit 1

echo "--- 상한별 분포 ---"
"$RUN_PY" - <<'PY'
import json, collections, os
rs = [json.loads(l) for l in open(os.environ["ALLEX_OUT"] + "/labels_full.jsonl")]
print(f"  청크 {len(rs)}")
by = collections.defaultdict(list)
for r in rs: by[r["task"]].append(r)
for t, v in sorted(by.items()):
    c = collections.Counter(r["K"] for r in v)
    print(f"  {t:<16} n={len(v):5d}  " + "  ".join(f"{k:g}x {100*n/len(v):4.1f}%" for k, n in sorted(c.items())))
PY
