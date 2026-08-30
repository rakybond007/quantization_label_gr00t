#!/bin/bash
# Label pilot tiles with subscription Claude (headless `claude -p`).
# Batches of 6 tiles per call; model via $PILOT_MODEL (default sonnet).
# Output: one JSONL line per frame {ep, f, p_yes} in $OUT/labels_${PILOT_MODEL}.jsonl
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate
OUT=$BASE/output/_gate_distill/sonnet_pilot
MODEL="${PILOT_MODEL:-sonnet}"
BATCH=6
GUIDE=$(cat $BASE/output/_gate_distill/real_gripper_patched_guidance.txt)
LABELS=$OUT/labels_${MODEL}.jsonl
: > "$LABELS"

mapfile -t TILES < <(~/miniconda3/envs/quant_gate_eval/bin/python -c "
import json
m=json.load(open('$OUT/manifest.json'))
print('\n'.join(t['path'] for t in m))")

n=${#TILES[@]}
for ((i=0; i<n; i+=BATCH)); do
  chunk=("${TILES[@]:i:BATCH}")
  files=$(printf '%s\n' "${chunk[@]}")
  prompt="You are judging frames from a robot teleoperation episode (task: 'Pick up the doll and put it into the plate').
Each image file below is one frame: LEFT half = external camera, RIGHT half = wrist (eye-in-hand) camera.

Decision: for EACH frame, can the next ~1 second of robot motion be compressed (run at half control rate) WITHOUT changing the outcome? Answer with a probability p_yes in [0,1] (1.0 = definitely safe to compress).

Guidance:
$GUIDE

Read each image file listed below, judge it, and output ONLY a JSON array (no prose), one object per file in the same order: [{\"file\": \"<basename>\", \"p_yes\": 0.xx}, ...]

Files:
$files"
  resp=$(claude -p --model "$MODEL" --allowedTools "Read" --max-turns 20 "$prompt" 2>>$OUT/driver_${MODEL}.err)
  echo "$resp" | ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import sys, json, re
raw=sys.stdin.read()
m=re.search(r'\[.*\]', raw, re.S)
if not m: sys.exit(0)
try: arr=json.loads(m.group(0))
except Exception: sys.exit(0)
for o in arr:
    b=o.get('file','')
    mm=re.match(r'ep(\d+)_f(\d+)', b)
    if mm: print(json.dumps({'ep':int(mm.group(1)),'f':int(mm.group(2)),'p_yes':float(o['p_yes'])}))
" >> "$LABELS"
  echo "[pilot-$MODEL] $((i+BATCH>n?n:i+BATCH))/$n done ($(wc -l < $LABELS) labels)" >&2
done
echo "[pilot-$MODEL] finished: $(wc -l < $LABELS) labels -> $LABELS" >&2
