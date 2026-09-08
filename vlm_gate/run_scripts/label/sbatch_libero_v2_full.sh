#!/bin/bash
#SBATCH --job-name=libero_v2_ratio_label_five_graded_checks_cosmos_judge_batch8_16shard
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# job-name 이 50자 미만이면 클러스터가 제출을 거부한다.
# libero v2 문항으로 전 프레임 라벨링. robocasa phase9 와 같은 모양이다 --
# 샤드마다 제 판정 서버를 띄우고, 배치 8 로 부르고, 재개 가능하다.
#
#   데이터셋   1,693 에피소드 · 273,465 프레임 (robocasa 204만의 1/7.5)
#   타일       stride 1 (전 프레임). 간격마다만 라벨하면 그 시점들이 gate_valid=0
#              으로 손실에서 빠져 감독이 성겨진다.
#   배치       8. **배치는 라벨의 일부다** -- phase9 에서 단건과 배치8 이 32칸 중
#              2칸 달랐다. 값은 <TAG>_s16_<shard>_meta.json 에 적힌다.
#
# 먼저 타일이 있어야 한다:
#   sbatch --array=0-15 vlm_gate/run_scripts/label/sbatch_libero_tiles.sh
#   python vlm_gate/scripts/gen_libero_tiles_shard.py merge 16
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8700+S))
LOG=output/_gate_distill/libero_v2_judge$S.log
GATE_SYSTEM=aligned $HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u \
  scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" $LOG && break; done
grep -q "JUDGE READY" $LOG || { tail -20 $LOG; kill $JP 2>/dev/null; exit 1; }
TAG=${TAG:-libero_v2_full} PROMPT_VER=v2 LIBERO_BATCH=${LIBERO_BATCH:-8} \
MANIFEST=$BASE/output/_gate_distill/libero_tiles_manifest.txt \
TILES=$BASE/output/_gate_distill/libero_full/tiles \
  $HOME/miniconda3/envs/quant_gate_eval/bin/python -u \
  scripts/libero_label_chunks.py $PORT $S 16
kill $JP 2>/dev/null
