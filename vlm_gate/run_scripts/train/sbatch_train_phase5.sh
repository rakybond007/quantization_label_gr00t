#!/bin/bash
#SBATCH --job-name=train_gate_module_A_phase5_labels_full_distillation_robocasa_247k_chunks
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node100,worker-node1,worker-node104,worker-node3
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --requeue
#SBATCH --output=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.out
#SBATCH --error=/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/out/%j-%x.err
set -u
WS="$HOME/quantization_agent_workspace"
cd "$WS/vlm_gate"
OUT="$WS/assets/modules_A/robocasa_module_A_phase5"
mkdir -p "$OUT"
"$HOME/miniconda3/envs/quant_gate_eval/bin/python" -u scripts/train_gate_module.py \
  --dataset-path /sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300 \
  --labels "$WS/assets/labels/robocasa/v6b_phase5_1call_full.parquet" \
  --cache-dir "$WS/assets/frame_cache_robocasa" \
  --task-emb "$WS/assets/robocasa_task_embeddings.npz" \
  --out-dir "$OUT" \
  --epochs 30 --bs 256 --lr 3e-4 --num-workers 8 \
  --wandb gateA_phase5_full
# 끝날 때 요약 JSON — 로컬 에이전트가 ls 없이 확인할 수 있도록
python - <<'PY' >> "$OUT/summary.json"
import json,os,glob,re
OUT=os.path.expanduser("~/quantization_agent_workspace/assets/modules_A/robocasa_module_A_phase5")
logs=sorted(glob.glob(os.path.expanduser("~/quantization_agent_workspace/vlm_gate/out/*-train_gate_module_A_phase5*.out")))
last={}
if logs:
    for l in open(logs[-1]):
        m=re.search(r"epoch (\d+)/(\d+) loss=([\d.]+) val_AUC=([\d.]+) val_agree@tau=([\d.]+)",l)
        if m: last=dict(epoch=int(m.group(1)),total=int(m.group(2)),loss=float(m.group(3)),
                        val_auc=float(m.group(4)),val_agree=float(m.group(5)))
print(json.dumps({"out":OUT,"ckpt_exists":os.path.exists(f"{OUT}/gate_module.pt"),"last":last},ensure_ascii=False))
PY
