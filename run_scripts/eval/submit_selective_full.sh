#!/bin/bash
# Submit all 6 selective compression eval sbatch jobs (2 ckpts × 3 score modes).
# Uses environment-driven eval_robocasa_selective.sh template.
#
# Set TAU values via env (one per setting). After smoke calibration picks τ per
# (ckpt × score), set them here or override via env.
#
# Usage:
#   bash run_scripts/eval/submit_selective_full.sh
#
# Override:
#   TAU_MH_SELF=0.10 TAU_MH_ENT=0.005 TAU_MH_HYB=0.40 \
#       TAU_MOE_SELF=0.10 TAU_MOE_ENT=0.005 TAU_MOE_HYB=0.40 \
#       bash run_scripts/eval/submit_selective_full.sh

set -u

TAU_MH_SELF=${TAU_MH_SELF:-0.10}
TAU_MH_ENT=${TAU_MH_ENT:-0.005}
TAU_MH_HYB=${TAU_MH_HYB:-0.40}
TAU_MOE_SELF=${TAU_MOE_SELF:-0.10}
TAU_MOE_ENT=${TAU_MOE_ENT:-0.005}
TAU_MOE_HYB=${TAU_MOE_HYB:-0.40}

ALPHA=${ALPHA:-0.5}
N_ENT=${N_ENT:-10}

cd "$HOME/multigpu_workspace/Isaac-GR00T"

submit() {
    local CKPT_TAG=$1 SCORE=$2 N=$3 TAU=$4
    echo "[submit] CKPT_TAG=$CKPT_TAG SCORE=$SCORE N=$N TAU=$TAU"
    CKPT_TAG=$CKPT_TAG SCORE_MODE=$SCORE N_SAMPLES=$N TAU_VALUE=$TAU ALPHA=$ALPHA \
        sbatch --job-name=ev_${CKPT_TAG}_${SCORE}_t${TAU//./p} \
        run_scripts/eval/eval_robocasa_selective.sh
}

submit mh_m8         self_agree 1      "$TAU_MH_SELF"
submit mh_m8         entropy    "$N_ENT" "$TAU_MH_ENT"
submit mh_m8         hybrid     "$N_ENT" "$TAU_MH_HYB"
submit per_expert_moe self_agree 1      "$TAU_MOE_SELF"
submit per_expert_moe entropy    "$N_ENT" "$TAU_MOE_ENT"
submit per_expert_moe hybrid     "$N_ENT" "$TAU_MOE_HYB"

echo "[done] all 6 selective sbatch submitted"
squeue -u hojin2 -o "%.10i %.30j %.4t %R" | grep -E "ev_|JOBID"
