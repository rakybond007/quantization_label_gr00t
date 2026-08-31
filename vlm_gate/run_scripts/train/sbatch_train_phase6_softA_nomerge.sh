#!/bin/bash
# phase6 labels with the controller-clip flag dropped from the aggregation.
#
# infeasible_merge fires when merging two steps would pass the simulator's action
# clip. Releasing that clip changes nothing measurable: naive K=2 scores 0.5983
# with it and 0.5950 with it scaled x3, at 214.0 and 213.6 steps. So the flag was
# blocking 18.7% of chunks on a property of the harness, not of the motion.
#
# Dropping it sharpens the labels where the gate actually decides. Per chunk,
# against whether the gripper transitions in the window:
#   phase5 0.690   phase6 0.758   phase6-nomerge 0.811
# and with each task's mean removed, 0.782 / 0.831 / 0.901. It scores WORSE on
# labelcheck's per-task axis (-0.045), which is the same disagreement phase6
# already shows — the closed loop after this settles it.
#
# Everything else is matched to sbatch_train_phase6_softA.sh so the aggregation
# is the only variable.
#SBATCH --job-name=train_gate_module_A_phase6_softA_without_the_controller_clip_flag_matched_control
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --requeue
#SBATCH --comment=MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase6_softA_nomerge
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.err
set -u
WS="$HOME/quantization_agent_workspace"; cd "$WS/vlm_gate"
OUT="$WS/assets/modules_A/robocasa_module_A_phase6_softA_nomerge"; mkdir -p "$OUT"
# phase5 softA 와 계산층·집계식·학습설정을 전부 맞췄다. 유일한 차이는 프롬프트다
# (가이던스 v5 다섯 축 + 5문항 대 phase5 의 4문항). 10에폭은 phase5 와 동일.
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/v6b_phase6_softA_nomerge.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$OUT" --epochs 10 --bs 256 --lr 3e-4 --num-workers 8 \
  --wandb "gateA_phase6_softA_nomerge"
python - "$OUT" <<'PY' > "$OUT/summary.json"
import json,os,sys; O=sys.argv[1]
print(json.dumps({"out":O,"best":os.path.exists(f"{O}/gate_module_best.pt"),
                  "final":os.path.exists(f"{O}/gate_module.pt")},ensure_ascii=False))
PY
