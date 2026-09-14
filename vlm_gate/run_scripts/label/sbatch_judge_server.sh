#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=persistent_cosmos_judge_server_for_prompt_iteration_rounds
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=debug
#SBATCH --time=2:55:00
#SBATCH --output=out/%j-judgesrv.out
#SBATCH --error=out/%j-judgesrv.err
set -u
# **판정기를 띄워 놓고 유지한다.** 문항을 깎는 단계는 라벨->채점->수정->재라벨의
# 짧은 반복인데, 라운드마다 sbatch 를 던지면 큐 대기 20~60분 + 모델 로딩 2~5분이
# 매번 붙는다. 실제 GPU 작업은 5~20분이다. 서버를 붙잡아 두면 그 둘이 한 번으로
# 끝난다. 전량 라벨링처럼 규모가 큰 것만 배열 잡으로 따로 던진다.
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
cd "$HOME/quantization_agent_workspace/vlm_gate" || exit 1
PORT=$((27000 + SLURM_JOB_ID % 900))
INFO=_tmp/judge_server.json
LOG="_tmp/judge_server_${SLURM_JOB_ID}.log"
"$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python" -u \
    scripts/vlm_gate_cosmos.py --serve --port "$PORT" > "$LOG" 2>&1 &
J=$!
trap 'kill $J 2>/dev/null; rm -f '"$INFO"'' EXIT
for _ in $(seq 1 120); do grep -q "JUDGE READY" "$LOG" && break; sleep 10; done
grep -q "JUDGE READY" "$LOG" || { echo "판정기 안 뜸"; tail -20 "$LOG"; exit 1; }
printf '{"port": %d, "host": "%s", "job": "%s", "log": "%s"}\n' \
    "$PORT" "$(hostname)" "$SLURM_JOB_ID" "$LOG" > "$INFO"
echo "[judge] READY host=$(hostname) port=$PORT -> $INFO"
# 시간이 끝날 때까지 살아 있는다. 클라이언트는 이 포트로 붙는다.
while kill -0 $J 2>/dev/null; do sleep 30; done
