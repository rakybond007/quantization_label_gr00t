#!/bin/bash
# 배속 머리 본학습. **파티션은 여기서 정한다.**
#
#   sbatch vlm_gate/experiments/ratio_head/sbatch_ratio_head.sh
#
# 전에는 이 파일이 없었다. run_full.sh 머리에 이 이름을 적어 두고 만들지
# 않아서, 제출하는 사람이 파티션을 매번 정하게 됐고 한 번은 background 로
# 나갔다. 파티션은 사람이 기억할 것이 아니라 파일에 적혀 있어야 한다.
#
#   학습          sjw_alinlab      <- 이 잡
#   라벨링 · 평가  background
#   시험          debug
#
# sjw_alinlab 은 남의 학습이 서는 자리다. array 를 쓰지 않고 GPU 도 2장까지만
# 쓴다. sjw_alinlab_premium 은 쿼터를 많이 먹어 따로 허락을 받는다.
#SBATCH --job-name=ratio_head_robocasa_contact_30k_bs256
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p sjw_alinlab
#SBATCH --gpus=2
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.err
set -eu
# 제출 환경에 있어야 하고 이 경로로 시작해야 한다. 없으면 슬럼이 거절한다.
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/ratio_head
bash $HOME/quantization_agent_workspace/vlm_gate/experiments/ratio_head/run_full.sh "$@"
