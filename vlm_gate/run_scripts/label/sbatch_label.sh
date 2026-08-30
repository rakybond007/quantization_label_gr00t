#!/bin/bash
#SBATCH --job-name=dense_full_frame_labeling_unified_driver_across_benchmarks
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
# 벤치마크는 인자 하나로만 갈린다: BENCH=robocasa|libero|dexjoco
# 나머지 차이(데이터 경로·뷰·기술자·프롬프트·에피소드 키)는 label_chunks.py 의
# BENCHMARKS 표에 선언돼 있다. 스크립트를 갈라 쓰면 재개 버그를 세 군데 고쳐야 한다.
BENCH="${BENCH:?BENCH=robocasa|libero|dexjoco 를 지정하세요}"
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8900+S))
LOG=output/_gate_distill/${BENCH}_judge$S.log
$HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u \
  scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" $LOG && break; done
grep -q "JUDGE READY" $LOG || { tail -20 $LOG; kill $JP 2>/dev/null; exit 1; }

TAG="${TAG:-${BENCH}_dense}" LABEL_STRIDE="${LABEL_STRIDE:-1}" \
  $HOME/miniconda3/envs/quant_gate_eval/bin/python -u \
  scripts/label_chunks.py "$BENCH" $PORT $S 16
rc=$?
kill $JP 2>/dev/null
# 라벨러의 종료코드를 잡의 종료코드로. 예전 잡들은 마지막이 kill 이라 무슨 일이
# 있었든 0 으로 끝났고, 그래서 잡 상태가 증거가 되지 못했다.
exit $rc
