#!/bin/bash
# Submit 12 selective AAC evals: 2 ckpt × 3 score × 2 AAC rule.
# AAC algorithm is threshold-free on score; ξ (cliff floor) and K (binary cutoff)
# are integer hyperparameters with defaults.
#
# Usage: bash run_scripts/eval/submit_selective_aac.sh
# Override:
#   AAC_XI_CLIFF=1 AAC_XI_BIN=4 N_ENT=10 bash ...
set -u

AAC_XI_CLIFF=${AAC_XI_CLIFF:-1}
AAC_XI_BIN=${AAC_XI_BIN:-4}
N_ENT=${N_ENT:-10}

cd "$HOME/multigpu_workspace/Isaac-GR00T"

submit() {
    local CKPT_TAG=$1 SCORE=$2 N=$3 RULE=$4 XI=$5
    echo "[submit] $CKPT_TAG / $SCORE / N=$N / rule=$RULE / xi=$XI"
    CKPT_TAG=$CKPT_TAG SCORE_MODE=$SCORE N_SAMPLES=$N \
    DECISION_RULE=$RULE AAC_XI=$XI \
        sbatch --job-name=aac_${CKPT_TAG}_${SCORE}_${RULE} \
        run_scripts/eval/eval_robocasa_selective_aac.sh
}

for CKPT in mh_m8 per_expert_moe; do
    # self_agree: N=1
    submit "$CKPT" self_agree 1 aac_cliff "$AAC_XI_CLIFF"
    submit "$CKPT" self_agree 1 aac_chunk_binary "$AAC_XI_BIN"
    # entropy: N=10 (paper exact, step-level AAC)
    submit "$CKPT" entropy "$N_ENT" aac_cliff "$AAC_XI_CLIFF"
    submit "$CKPT" entropy "$N_ENT" aac_chunk_binary "$AAC_XI_BIN"
    # hybrid: N=10
    submit "$CKPT" hybrid "$N_ENT" aac_cliff "$AAC_XI_CLIFF"
    submit "$CKPT" hybrid "$N_ENT" aac_chunk_binary "$AAC_XI_BIN"
done

echo "[done] 12 AAC sbatch submitted"
squeue -u hojin2 -o "%.10i %.30j %.4t %R" | grep -E "aac_|JOBID"
