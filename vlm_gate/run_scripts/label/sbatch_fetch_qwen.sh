#!/bin/bash
#SBATCH --job-name=fetch_qwen3_8_27b_weights_for_judge_ab_comparison_no_gpu
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.err
set -u
# 가중치만 받는다. 실제로 GPU 는 안 쓰지만, 이 클러스터는
# --cpus-per-task 를 막아 두었고(DefCpuPerGPU 가 자동 적용) GPU 를
# 잡아야 CPU 가 붙는다. 한 장만 잡는다.
#
# 로그인 노드에서 nohup 으로 걸었더니 세션이 끊길 때 같이 죽었다(3.9/55.6 GB 에서).
# 잡으로 걸면 안 죽고, 재큐돼도 이미 받은 파일은 건너뛴다.
V=$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python
$V - <<'PY'
import os
from huggingface_hub import snapshot_download
tok = None
p = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(p):
    tok = open(p).read().strip()
print("DONE", snapshot_download("Qwen/Qwen3.8-27B", max_workers=16, token=tok))
PY
