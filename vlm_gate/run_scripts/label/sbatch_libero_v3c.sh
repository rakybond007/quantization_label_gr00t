#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=libero_question_variant_v3c_single_change_diagnosis_phase_balanced_scenes
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=background
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --output=out/%j-libv3c.out
#SBATCH --error=out/%j-libv3c.err
set -u
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase9_labels
bash $HOME/quantization_agent_workspace/vlm_gate/_tmp/libero/run_v3c.sh
