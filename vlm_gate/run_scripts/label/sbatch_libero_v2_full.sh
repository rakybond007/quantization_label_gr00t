#!/bin/bash
#SBATCH --job-name=libero_v2_ratio_label_five_graded_checks_cosmos_judge_batch8_16shard
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# job-name 이 50자 미만이면 클러스터가 제출을 거부한다.
#
# **array 는 4 다. 16 이 아니다.** background 는 남는 자리를 쓰는 파티션이라
# 16샤드를 물어도 되지만(robocasa phase9 가 그랬다), sjw_alinlab 은 학습용이라
# 그만큼 물면 남의 학습을 밀어낸다. allex 라벨링도 alinlab 계열로 갈 때는
# array 0-1 로 줄였고, phase9 는 array 대신 GPU 4장에 일꾼 4개로 돌렸다.
# background 로 되돌릴 때만 16 으로 올린다.
# libero v2 문항으로 전 프레임 라벨링. robocasa phase9 와 같은 모양이다 --
# 샤드마다 제 판정 서버를 띄우고, 배치 8 로 부르고, 재개 가능하다.
#
#   데이터셋   1,693 에피소드 · 273,465 프레임 (robocasa 204만의 1/7.5)
#   샤드       4. sjw_alinlab 에서는 그 이상 물지 않는다(위 주석).
#   타일       stride 1 (전 프레임). 간격마다만 라벨하면 그 시점들이 gate_valid=0
#              으로 손실에서 빠져 감독이 성겨진다.
#   배치       8. **배치는 라벨의 일부다** -- phase9 에서 단건과 배치8 이 32칸 중
#              2칸 달랐다. 값은 <TAG>_s16_<shard>_meta.json 에 적힌다.
#
# 먼저 타일이 있어야 한다:
#   sbatch vlm_gate/run_scripts/label/sbatch_libero_tiles.sh
#   python vlm_gate/scripts/gen_libero_tiles_shard.py merge 4
#
# merge 뒤의 숫자는 **타일 잡의 array 크기와 같아야 한다.** 다르면 샤드
# 매니페스트를 못 찾아 조용히 일부만 합쳐진다.
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
NSH=$((SLURM_ARRAY_TASK_MAX+1))   # 샤드 수는 array 크기에서 받는다
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
  scripts/libero_label_chunks.py $PORT $S $NSH
kill $JP 2>/dev/null
