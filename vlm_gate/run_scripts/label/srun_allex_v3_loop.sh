#!/bin/bash
# 문항을 고치는 동안 도는 빠른 한 바퀴: 라벨 → 검증. 영상은 안 만든다.
# 기본은 에피소드 0 하나 -- 네 서브태스크를 다 지나므로 검증 5항목이 다 나온다.
set -u
JUDGE_PY="$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python"
RUN_PY="/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python"
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
SAMPLE="${SAMPLE:-$PWD/output/allex_sample/D.json}"
PORT="${PORT:-8261}"
# 한 바퀴에 하나씩 나온다. 앞 바퀴가 끝나기 전에 다음 바퀴를 걸었더니
# 이 rm -rf 가 앞 바퀴의 라벨을 지웠고, 앞 바퀴의 검증이 30청크만 보고
# 끝났다. 잡 번호를 경로에 넣으면 겹칠 수가 없다.
export ALLEX_OUT="${ALLEX_OUT:-$PWD/output/allex_loop_${SLURM_JOB_ID:-manual}}"
rm -rf "$ALLEX_OUT"; mkdir -p "$ALLEX_OUT"
echo "출력: $ALLEX_OUT"

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$JUDGE_PY" -u scripts/vlm_gate_cosmos.py \
    --serve --port "$PORT" > /tmp/judge_allex_loop.log 2>&1 &
JP=$!
trap 'kill $JP 2>/dev/null' EXIT
for _ in $(seq 1 60); do grep -q "JUDGE READY" /tmp/judge_allex_loop.log && break; sleep 10; done
grep -q "JUDGE READY" /tmp/judge_allex_loop.log || { echo "판정기 안 뜸"; tail -20 /tmp/judge_allex_loop.log; exit 1; }

"$RUN_PY" -u scripts/allex_v3_label.py "$PORT" "$SAMPLE" || exit 1
"$RUN_PY" -u scripts/allex_v3_verify.py
echo "--- 답변 조합 ---"
"$RUN_PY" - <<'PY'
import json, collections, os
rs=[json.loads(l) for l in open(os.environ["ALLEX_OUT"]+"/records.jsonl")]
c=collections.Counter((r["A"],r["B"],r["C"],r["D"]) for r in rs)
for k,v in c.most_common(10):
    print("  %s  %4d  %5.1f%%" % (" ".join(map(str,k)), v, 100*v/len(rs)))
print("  총 %d가지" % len(c))
PY
