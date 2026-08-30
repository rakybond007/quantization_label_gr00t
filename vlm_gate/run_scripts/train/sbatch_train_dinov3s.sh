#!/bin/bash
#SBATCH --job-name=train_gate_module_dinov3s_frozen_encoder_attention_pooling_phase5_labels
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.err
set -u
WS="$HOME/quantization_agent_workspace"; cd "$WS/vlm_gate"
OUT="$WS/assets/modules_A/robocasa_module_A_phase5_dinov3s"; mkdir -p "$OUT"
# DINOv3 는 transformers 4.56+ 가 필요하다. 평가 스택의 4.51.3 고정을 깨지 않도록
# 별도 디렉터리 오버레이를 얹는다 (numpy 는 환경 것을 그대로 쓴다).
export PYTHONPATH="$WS/pylibs/tf4573${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --encoder dinov3s \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/v6b_phase5_1call_full.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$OUT" --epochs 10 --bs 128 --lr 3e-4 --num-workers 8 \
  --wandb gateA_phase5_dinov3s
python - "$OUT" <<'PY' > "$OUT/summary.json"
import json,os,sys; O=sys.argv[1]
print(json.dumps({"out":O,"best":os.path.exists(f"{O}/gate_module_best.pt"),
                  "final":os.path.exists(f"{O}/gate_module.pt")},ensure_ascii=False))
PY
