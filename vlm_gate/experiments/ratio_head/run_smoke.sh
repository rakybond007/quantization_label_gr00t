#!/bin/bash
# 배속 머리 스모크. **srun 안에서 돌린다** -- 산출물이 실제로 나오는지 본다.
#
#   srun --wckey=project-short-name:sub_fast -p debug --gpus=1 --time=00:40:00 \
#     --exclude=worker-node100,worker-node1,worker-node104,worker-node3 \
#     bash vlm_gate/experiments/ratio_head/run_smoke.sh
set -eu
W=$HOME/quantization_agent_workspace
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN=${RUN:-$W/vlm_gate/experiments/ratio_head/artifacts/smoke}
LAB=${LAB:-$W/vlm_gate/output/_gate_distill/robocasa_contact_ratio.parquet}

$HOME/miniconda3/envs/quant_gate/bin/python "$HERE/train.py" \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$LAB" \
  --cache-dir "$W/assets/frame_cache_robocasa" \
  --task-emb "$W/assets/robocasa_task_embeddings.npz" \
  --out-dir "$RUN" \
  --max-steps 60 --bs 32 --max-tasks 0 --preload \
  --episodes-per-task 2 --frames-per-episode 8 "$@"

echo "--- 산출물"
ls -la "$RUN" 2>/dev/null
