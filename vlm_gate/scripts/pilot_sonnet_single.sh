#!/bin/bash
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate
OUT=$BASE/output/_gate_distill/sonnet_pilot
GUIDE=$(cat $BASE/output/_gate_distill/real_gripper_patched_guidance.txt)
LABELS=$OUT/labels_sonnet_single.jsonl
: > "$LABELS"
mapfile -t TILES < <(~/miniconda3/envs/quant_gate_eval/bin/python -c "
import json; m=json.load(open('$OUT/manifest.json')); print('\n'.join(t['path'] for t in m))")
n=${#TILES[@]}; done=0
for p in "${TILES[@]}"; do
  b=$(basename "$p")
  resp=$(claude -p --model sonnet --allowedTools "Read" --max-turns 4 "Read the image $p — one frame from a robot teleop episode (task: 'Pick up the doll and put it into the plate'). LEFT half = external camera, RIGHT half = wrist camera.
Can the next ~1 second of robot motion be compressed (half control rate) without changing the outcome?
Guidance:
$GUIDE
Output ONLY: {\"p_yes\": 0.xx}" 2>>$OUT/driver_sonnet_single.err)
  echo "$resp" | B="$b" ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import sys,os,json,re
m=re.search(r'\{[^{}]*p_yes[^{}]*\}', sys.stdin.read())
if m:
    try:
        v=float(json.loads(m.group(0))['p_yes'])
        mm=re.match(r'ep(\d+)_f(\d+)', os.environ['B'])
        print(json.dumps({'ep':int(mm.group(1)),'f':int(mm.group(2)),'p_yes':v}))
    except Exception: pass
" >> "$LABELS"
  done=$((done+1))
  [ $((done % 25)) -eq 0 ] && echo "[sonnet-single] $done/$n ($(wc -l < $LABELS))" >&2
done
echo "[sonnet-single] finished: $(wc -l < $LABELS) -> $LABELS" >&2
