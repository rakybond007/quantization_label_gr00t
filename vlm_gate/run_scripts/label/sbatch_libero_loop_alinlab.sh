#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=libero_question_design_loop_phase_scenes_and_v2_diagnosis_overnight
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=out/%j-libloop.out
#SBATCH --error=out/%j-libloop.err
#SBATCH --comment="LIBERO question-design loop: wait for tiles, build phase-balanced scenes, label with v2 checks, score."
# **sjw_alinlab 을 쓰는 이유.** background 는 tier 1 이라 선점당한다 -- 이 루프가
# 라벨 160 행에서 선점돼 재큐됐고, 큐가 혼잡해 재시작이 오래 걸린다. 판정기는 GPU 가
# 필수라 cpu 파티션으로 못 옮기고, batch 는 권한이 없다. GPU 1 대만 쓰므로 학습을
# 밀어내지 않는다(사용자가 정한 선은 4 대).
set -u
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
: "${SLURM_ARRAY_TASK_ID:=0}"
export SLURM_ARRAY_TASK_ID
bash $HOME/quantization_agent_workspace/vlm_gate/_tmp/libero/loop.sh
