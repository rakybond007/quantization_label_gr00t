#!/bin/bash
# 판정기 A/B — **GPU 를 한 번만 잡고** 스모크와 두 모델 라벨링을 이어서 한다.
#
# 따로 잡으면 큐를 세 번 기다린다. 모델을 바꿔 다시 띄우는 것은 같은 할당 안에서
# 하면 되고, 판정 서버는 프로세스라 죽였다 다시 띄우면 그만이다.
#
#   srun --wckey=project-short-name:sub_fast -p debug --gpus=2 --time=02:00:00 \
#     --exclude=worker-node100,worker-node1,worker-node104,worker-node3 \
#     bash vlm_gate/run_scripts/label/srun_judge_ab.sh
#
# GPU 2장인 이유: Qwen3.8-27B 이 bf16 으로 55GB 라 한 장(80GB)에 들어가긴 하지만
# 활성값까지 두면 빠듯하다. Cosmos3-Nano(16B)는 한 장이면 남는다.
# **더 물지 않는다** -- debug 는 남의 것과 나눠 쓰는 자리다.
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate
V=$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python
cd $HOME/quantization_agent_workspace

QWEN=${QWEN:-Qwen/Qwen3.8-27B}
COSMOS=${COSMOS:-nvidia/Cosmos3-Nano}
OUT=${OUT:-$BASE/output/_gate_distill/judge_ab}
mkdir -p $OUT

echo "===== [0] 장면 고르기 (CPU)"
$V $BASE/scripts/judge_ab.py pick --out $OUT/scenes.json --episodes 8 || exit 1

echo
echo "===== [1] Qwen 네 단계 시험"
$V $BASE/scripts/judge_smoke_model.py --model "$QWEN"
RC=$?
if [ $RC -ne 0 ]; then
  echo
  echo "!! Qwen 이 네 단계를 못 넘었다. 여기서 멈춘다 -- 형식대로 못 답하는"
  echo "!! 모델로 A/B 를 돌리면 비교가 아니라 고장 확인이 된다."
  echo "!! 기존 환경을 고치지 말고 새 venv 에 최신 transformers 를 깔 것."
  exit $RC
fi

serve () {   # $1=model $2=port $3=log
  GATE_SYSTEM=aligned $V $BASE/scripts/vlm_gate_cosmos.py --serve \
    --model "$1" --port "$2" > "$3" 2>&1 &
  echo $!
}
wait_ready () {   # $1=log
  for i in $(seq 1 90); do
    sleep 10
    grep -q "JUDGE READY" "$1" && return 0
  done
  tail -25 "$1"; return 1
}

for pair in "cosmos|$COSMOS|8210" "qwen|$QWEN|8211"; do
  IFS='|' read -r TAG MODEL PORT <<< "$pair"
  echo
  echo "===== [2] $TAG 판정 서버 ($MODEL)"
  LOG=$OUT/${TAG}_serve.log
  PID=$(serve "$MODEL" "$PORT" "$LOG")
  if ! wait_ready "$LOG"; then
    echo "!! $TAG 서버가 안 떴다. 건너뛴다."
    kill $PID 2>/dev/null; continue
  fi
  $V $BASE/scripts/judge_ab.py run --port "$PORT" --tag "$TAG" \
     --scenes $OUT/scenes.json --out $OUT/${TAG}.jsonl
  kill $PID 2>/dev/null
  sleep 20          # 다음 모델을 올리기 전에 GPU 를 비운다
done

echo
echo "===== [3] 나란히 놓기"
$V $BASE/scripts/judge_ab.py cmp $OUT/cosmos.jsonl $OUT/qwen.jsonl
echo
echo "산출물: $OUT"
