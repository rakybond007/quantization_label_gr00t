#!/bin/bash
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --job-name=verify_newhome_migration_e2e_smoke_gemma4_cosmos3_x_libero_robocasa_gr00t_n1_5_ttl_gate
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=background
#SBATCH --exclude=worker-node100,worker-node1
#SBATCH --time=04:00:00
#SBATCH --output=out/%j-newhome_e2e_smoke.out
#SBATCH --error=out/%j-newhome_e2e_smoke.err
#SBATCH --comment="NEW-HOME migration end-to-end verification: run all 4 judge x bench smoke combos (2ep, TTL) fully from /sjw_alinlab home."
set -u
export HOME=/sjw_alinlab/home/hojin2
cd "$HOME/quantization_agent_workspace/vlm_gate"
for C in "libero gemma" "libero cosmos" "robocasa gemma" "robocasa cosmos"; do
  set -- $C
  echo "===== START $1/$2 $(date +%H:%M) ====="
  bash run_scripts/eval/_smoke_newhome.sh "$1" "$2"
done
echo "NEWHOME_E2E_SMOKE_DONE"
