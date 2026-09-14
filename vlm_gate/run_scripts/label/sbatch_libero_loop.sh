#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=libero_question_design_loop_phase_scenes_and_v2_diagnosis_overnight
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=out/%j-libloop.out
#SBATCH --error=out/%j-libloop.err
#SBATCH --comment="LIBERO question-design loop: wait for tiles, build phase-balanced scenes, label with v2 checks, score."
set -u
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
: "${SLURM_ARRAY_TASK_ID:=0}"
export SLURM_ARRAY_TASK_ID
bash $HOME/quantization_agent_workspace/vlm_gate/_tmp/libero/loop.sh
