#!/bin/bash
#SBATCH --job-name=dump_raw_action_chunk_samples_for_offline_entropy_ablation
#SBATCH --qos=eval
#SBATCH --gres=gpu:1
#SBATCH --time=3:45:00
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/dump_%A_%a.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/dump_%A_%a.out
set -uo pipefail
# One GPU pass over 96 RoboCasa episodes (stride 300 = 4 per task across all 24 tasks),
# saving raw action-chunk samples so the DemoSpeedup entropy knobs -- bandwidth, which
# action dims, ensemble depth -- can be compared on CPU without re-running the model.
BENCH=${BENCH:-robocasa}
case "$BENCH" in
  robocasa) DATA=/s3data/robocasa_mg_gr00t_300/v1; CFG=single_panda_gripper; TAG=new_embodiment
            CKREPO=models--prehj--GR00T-N1.5-robocasa-baseline; STRIDE=${STRIDE:-300}; DIMS=${DIMS:-0:12} ;;
  libero)   DATA=/s3data/libero_gr00t_delta/v1;    CFG=libero;               TAG=libero
            # bs32 is the correct recipe and the control we compare against; the
            # bs64 checkpoint of the same name was trained at the wrong batch size.
            CKREPO=models--prehj--GR00T-N1.5-libero-baseline-bs32-60k; STRIDE=${STRIDE:-70}; DIMS=${DIMS:-0:7} ;;
  *) echo "unknown BENCH=$BENCH"; exit 1 ;;
esac
OUT=/ckpt/$USER/baselines/action_samples_${BENCH}${OUTSUF:-}
mkdir -p "$OUT"
SHARD=${SLURM_ARRAY_TASK_ID:-0}
CKPT=$(find "$HOME/.cache/huggingface/hub/$CKREPO/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate quant_gate
export HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
cd "$HOME/quant_label_workspace/quantization_label_gr00t"
echo "[$(date '+%T')] node=$(hostname) bench=$BENCH shard=$SHARD data=$DATA cfg=$CFG stride=$STRIDE"
python -u baselines/dump_action_samples.py \
  --dataset-path "$DATA" --model-path "$CKPT" \
  --data-config "$CFG" --embodiment-tag "$TAG" \
  --episodes ${EP:-24} --episode-offset $SHARD --episode-stride $STRIDE \
  --n-samples ${N_SAMPLES:-10} --frame-batch ${FRAME_BATCH:-8} \
  --emit ${EMIT:-samples} --dims "$DIMS" --bandwidth ${BANDWIDTH:-1} --ensemble-k ${ENSEMBLE_K:-8} \
  --out-dir "$OUT"
echo "[$(date '+%T')] files: $(ls "$OUT"/ep*.np* 2>/dev/null | wc -l)"
