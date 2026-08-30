#!/bin/bash
#SBATCH --job-name=robocasa_dense_full_frame_labeling_phase6_guidance_v5_every_timestep
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# 전 프레임 라벨링. 타일을 굽지 않고 영상에서 바로 디코딩한다 — 간격 1 이면 타일이
# 207만 개가 되어 이 파일시스템에서 감당이 안 된다. 이유는 스크립트 상단 주석에.
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8900+S))
LOG=output/_gate_distill/dense_judge$S.log
$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u \
  scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" $LOG && break; done
grep -q "JUDGE READY" $LOG || { tail -20 $LOG; kill $JP 2>/dev/null; exit 1; }

GUIDANCE_FILE=robocasa_guidance_phase_v5.txt TAG=dense_phase6 LABEL_STRIDE=1 \
  $HOME/miniconda3/envs/quant_gate_eval/bin/python -u \
  scripts/cosmos_label_dense.py $PORT $S 16
rc=$?
kill $JP 2>/dev/null
exit $rc      # 라벨러의 종료코드를 잡의 종료코드로 — kill 의 것이 아니라
