#!/bin/bash
# usage: luna_label_streaming.sh <worker_id> <n_workers>
# tiles 디렉토리를 주기적으로 재스캔, ep%n==id 인 미라벨 타일을 즉시 라벨링 (타일 생성과 병행).
set -u
WID=$1; NW=$2
BASE=$HOME/quantization_agent_workspace/vlm_gate
DIR=$BASE/output/_gate_distill/luna_robocasa_full
LABELS=$DIR/labels_luna_w$WID.jsonl
BATCH=6
GUIDE=$(cat $BASE/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt)
touch "$LABELS"
idle=0
while [ $idle -lt 6 ]; do
  mapfile -t TILES < <(W=$WID N=$NW D=$DIR ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import os, json, glob
done=set()
for f in glob.glob(os.environ['D']+'/labels_luna_w*.jsonl'):
    for l in open(f):
        try: r=json.loads(l); done.add((r['ep'],r['f']))
        except Exception: pass
w=int(os.environ['W']); n=int(os.environ['N'])
out=[]
for p in sorted(glob.glob(os.environ['D']+'/tiles/ep*.png')):
    b=os.path.basename(p); ep=int(b[2:6]); fi=int(b[8:11])
    if ep%n==w and (ep,fi) not in done: out.append(p)
print('\n'.join(out[:600]))")
  if [ ${#TILES[@]} -eq 0 ] || [ -z "${TILES[0]:-}" ]; then
    idle=$((idle+1)); sleep 600; continue
  fi
  idle=0
  n=${#TILES[@]}
  for ((i=0; i<n; i+=BATCH)); do
    chunk=("${TILES[@]:i:BATCH}")
    IFLAGS=(); for p in "${chunk[@]}"; do IFLAGS+=(-i "$p"); done
    k=${#chunk[@]}
    PROMPT="You are judging $k frames (attached, in order) from RoboCasa simulated kitchen manipulation demos (task visible from scene). Panels: left cam, right cam, wrist cam.
For EACH frame in order: p_yes in [0,1] = probability the next ~1 second of robot motion can be compressed (half control rate) without changing the outcome.
Guidance:
$GUIDE
Output ONLY a JSON array of $k numbers."
    resp=$(codex exec --skip-git-repo-check -m gpt-5.6-luna "${IFLAGS[@]}" -- "$PROMPT" 2>>$DIR/w$WID.err | tail -3)
    printf '%s\n' "${chunk[@]}" | RESP="$resp" ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import sys, os, json, re
paths=[l.strip() for l in sys.stdin if l.strip()]
m=re.search(r'\[[-0-9.,eE\s]+\]', os.environ['RESP'])
if m:
    try:
        vals=json.loads(m.group(0))
        if len(vals)==len(paths):
            for p,v in zip(paths,vals):
                b=os.path.basename(p)
                print(json.dumps({'ep':int(b[2:6]),'f':int(b[8:11]),'p_yes':float(v)}))
    except Exception: pass
" >> "$LABELS"
  done
done
echo "[w$WID] no new tiles for 1h - exiting. total $(wc -l < $LABELS)" >&2
