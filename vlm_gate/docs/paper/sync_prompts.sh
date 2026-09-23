#!/bin/bash
# 라벨링에 실제로 쓴 프롬프트를 논문 디렉터리로 다시 가져온다. 논문 쪽을 손으로 고치면
# 실제 라벨과 갈라진다 -- 항상 이 스크립트로 맞춘다.
#
# 넷은 이 서버의 vlm_gate/output 에서, 둘(ALLEX·DexJoCo)은 다른 서버에서 라벨링해
# HF 배달본이 정본이므로 HF 에서 받는다.
set -euo pipefail
V=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate
D="$(cd "$(dirname "$0")" && pwd)/prompts"
mkdir -p "$D"
for pair in robocasa_v22:robocasa_v22 libero_v3d:libero_v3d humandata_v3:humandata_v3 openarm_v2:openarm_v2; do
  name=${pair%%:*}; dir=${pair#*:}; src="$V/output/$dir/PROMPT.txt"
  [ -f "$src" ] || { echo "[ERR] 없다: $src"; exit 1; }
  cp "$src" "$D/$name.txt"; printf "  %-24s <- %s (%s줄)\n" "$name.txt" "$src" "$(wc -l < "$src")"
done
"$HOME/miniconda3/envs/quant_gate/bin/python" - "$D" <<'PY'
import sys, shutil
from huggingface_hub import hf_hub_download
D = sys.argv[1]
tok = open("/sjw_alinlab/home/hojin2/quantization_agent_workspace/hf_token.txt").read().strip()
for repo, path, name in (("prehj/allex-ratio-labels-v1v2v3v4", "prompts/allex.txt", "allex_v1v2v3v4.txt"),
                         ("prehj/dexjoco-vlm-labels-Astra", "prompts/dexjoco_v3_FULL.txt", "dexjoco_v3.txt")):
    p = hf_hub_download(repo, path, repo_type="dataset", token=tok, force_download=True)
    shutil.copy(p, f"{D}/{name}")
    print(f"  {name:24s} <- {repo}/{path} ({sum(1 for _ in open(p))}줄)")
PY
