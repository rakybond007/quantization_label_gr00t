#!/bin/bash
# 라벨링에 실제로 쓴 프롬프트를 논문 디렉터리로 다시 복사한다.
# 논문 쪽을 손으로 고치면 실제 라벨과 갈라진다 -- 항상 이 스크립트로 맞춘다.
set -euo pipefail
V=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate
D="$(cd "$(dirname "$0")" && pwd)/prompts"
mkdir -p "$D"
for pair in robocasa_v22:robocasa_v22 libero_v3d:libero_v3d humandata_v3:humandata_v3 openarm_v2:openarm_v2; do
  name=${pair%%:*}; dir=${pair#*:}
  src="$V/output/$dir/PROMPT.txt"
  [ -f "$src" ] || { echo "[ERR] 없다: $src"; exit 1; }
  cp "$src" "$D/$name.txt"
  printf "  %-16s <- %s (%s줄)\n" "$name.txt" "$src" "$(wc -l < "$src")"
done
