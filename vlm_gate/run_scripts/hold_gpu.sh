#!/bin/bash
# 할당을 붙들고 시킬 일을 이어 받는다. **일이 없다고 반납하지 않는다** --
# 다음에 뭘 시킬지는 여기서 정하는 게 아니고, 큐를 다시 서는 값이 노는 값보다 크다.
set -u
W=$HOME/quantization_agent_workspace
Q=$W/vlm_gate/_tmp/gpu_queue
mkdir -p "$Q/done"
echo "[hold] $(hostname)  GPU=${CUDA_VISIBLE_DEVICES:-?}  $(date +%H:%M:%S)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -2
while :; do
    shopt -s nullglob
    for f in "$Q"/*.sh; do
        n=$(basename "$f")
        echo "[hold] 실행 $n  $(date +%H:%M:%S)"
        bash "$f" > "$Q/done/${n%.sh}.log" 2>&1
        echo "[hold] $n exit=$?"
        mv "$f" "$Q/done/$n"
    done
    [ -f "$Q/STOP" ] && { echo "[hold] STOP"; break; }
    sleep 10
done
