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
# **run_full.sh 와 같은 파일이어야 한다.** 다르면 스모크가 본학습과 다른 것을
# 검사한다. 옛 파일에는 instruction 열이 없어 text_col 이 task 로 떨어지고,
# 임베딩은 지시문 334종이라 태스크 24개가 전부 안 찾아져 죽는다.
LAB=${LAB:-$W/vlm_gate/output/_gate_distill/robocasa_contact_ratio_instr.parquet}

# 스모크는 매번 새로 시작한다. 학습기가 out-dir 을 exist_ok=False 로 만들어서,
# 앞 판이 빈 디렉터리만 남기고 죽으면 다음 판이 FileExistsError 로 죽는다.
# 본학습(run_full.sh)에서는 안 지운다 -- 거기서는 덮어쓰기가 사고다.
rm -rf "$RUN"

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
