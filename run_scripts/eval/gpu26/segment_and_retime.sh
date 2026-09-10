#!/bin/bash
#SBATCH --job-name=demospeedup_segment_entropy_and_verify_retiming_rule
#SBATCH --qos=eval
#SBATCH --time=3:30:00
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --output=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/segment_%j.out
#SBATCH --error=/fsx/home/hojin/quant_label_workspace/setup_jobs/logs/segment_%j.out
set -uo pipefail
BENCH=${BENCH:-robocasa}
case "$BENCH" in
  robocasa) SRC=/s3data/robocasa_mg_gr00t_300/v1;  DELTA=5:11; DISC=4,11 ;;
  libero)   SRC=/s3data/libero_gr00t_delta/v1;     DELTA=0:6;  DISC=6    ;;
esac
OUT=/ckpt/$USER/baselines/demospeedup
mkdir -p "$OUT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate quant_gate
export NO_ALBUMENTATIONS_UPDATE=1
cd "$HOME/quant_label_workspace/quantization_label_gr00t/baselines"

echo "[$(date '+%T')] === segment $BENCH ==="
python -u segment_entropy.py \
  --entropy-glob "/s3ckpt/$USER/baselines/action_samples_${BENCH}_full/ep*_entropy.npz" \
  --out "$OUT/labels_$BENCH.npz" --rule ${RULE:-reference}

echo "[$(date '+%T')] === verify retiming rule on $BENCH (parquet action deltas) ==="
python -u retime_dataset.py --src "$SRC" --labels "$OUT/labels_$BENCH.npz" \
  --dst "$OUT/retimed_$BENCH" --delta-dims "$DELTA" --discrete-dims "$DISC" \
  --limit ${LIMIT:-200} --verify-only
echo "[$(date '+%T')] done"
