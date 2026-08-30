#!/bin/bash
# Submit 4 libero MoE per_expert eval sbatch (one per decision mode).
# Each sbatch sets its own --job-name (≥50 chars) via env override.
set -u
cd "$HOME/multigpu_workspace/Isaac-GR00T"

submit() {
    local MODE=$1 NAME=$2
    echo "[submit] mode=$MODE  name=$NAME (${#NAME} chars)"
    MODE=$MODE sbatch --job-name="$NAME" run_scripts/eval/eval_libero_moe4_per_expert.sh
}

submit as_is            "eval_libero_moe4_per_expert_router_pick_as_is_no_compression_60k_chkpt"
submit aac_cliff        "eval_libero_moe4_per_expert_with_aac_cliff_prefix_compression_xi1_60k"
submit aac_chunk_binary "eval_libero_moe4_per_expert_with_aac_chunk_binary_K4_compression_60k"
submit self_agree       "eval_libero_moe4_per_expert_with_self_agreement_chunk_binary_tau05_60k"

squeue -u hojin2 -o "%.10i %.70j %.4t %R" | grep -E "eval_libero|JOBID"
