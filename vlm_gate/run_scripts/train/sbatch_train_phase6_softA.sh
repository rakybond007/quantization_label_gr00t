#!/bin/bash
#SBATCH --job-name=train_gate_module_A_phase6_five_axis_prompt_softA_matched_to_phase5_control
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --requeue
#SBATCH --comment=MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/gate_modules/phase6_softA
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%A-%x.err
set -u
WS="$HOME/quantization_agent_workspace"; cd "$WS/vlm_gate"
OUT="$WS/assets/modules_A/robocasa_module_A_phase6_softA"; mkdir -p "$OUT"
# phase5 softA 와 계산층·집계식·학습설정을 전부 맞췄다. 유일한 차이는 프롬프트다
# (가이던스 v5 다섯 축 + 5문항 대 phase5 의 4문항). 10에폭은 phase5 와 동일.
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/v6b_phase6_softA.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$OUT" --epochs 10 --bs 256 --lr 3e-4 --num-workers 8 \
  --wandb "gateA_phase6_softA"
python - "$OUT" <<'PY' > "$OUT/summary.json"
import json,os,sys; O=sys.argv[1]
print(json.dumps({"out":O,"best":os.path.exists(f"{O}/gate_module_best.pt"),
                  "final":os.path.exists(f"{O}/gate_module.pt")},ensure_ascii=False))
PY
