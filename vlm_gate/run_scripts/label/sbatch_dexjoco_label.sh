#!/bin/bash
#SBATCH --job-name=dexjoco_v1_chunk_labelling_five_axes_guidance_absolute_actions
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH -p background
#SBATCH --gpus=1
#SBATCH --array=0-15
#SBATCH --requeue
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --comment=MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/dexjoco_v1_labels
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A_%a-%x.err
set -u
# LAUNCH:
#   MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/dexjoco_v1 \
#     sbatch run_scripts/label/sbatch_dexjoco_label.sh
# VERIFY:  ~/quantization_agent_workspace/bin/qgate labels dexjoco_v1_s16 -v
# Each array task owns its own judge server on port 8900+shard and labels the
# global episodes with ep % 16 == shard (ep = 1000*task_id + local episode, see
# scripts/dexjoco_label_common.py -- the six tasks number their episodes
# independently, so the key has to be global or they merge).
# Resumable: the labeller skips every (ep, f) already in its shard file, so a
# preempted+requeued task re-emits nothing.
# PREREQ: sbatch_dexjoco_tiles.sh, then dexjoco_tiles_manifest.py.
BASE=$HOME/quantization_agent_workspace/vlm_gate; cd $BASE
S=$SLURM_ARRAY_TASK_ID; PORT=$((8900+S))
LOG=output/_gate_distill/dexjoco_v1_judge$S.log
GATE_SYSTEM=aligned $HOME/quantization_agent_workspace/cosmos_judge_venv/bin/python -u \
  scripts/vlm_gate_cosmos.py --serve --port $PORT > $LOG 2>&1 &
JP=$!
for i in $(seq 1 120); do sleep 20; grep -q "JUDGE READY" $LOG && break; done
grep -q "JUDGE READY" $LOG || { tail -20 $LOG; kill $JP 2>/dev/null; exit 1; }
TAG=dexjoco_v1 $HOME/miniconda3/envs/quant_gate_eval/bin/python -u \
  scripts/dexjoco_label_chunks.py $PORT $S 16
kill $JP 2>/dev/null
