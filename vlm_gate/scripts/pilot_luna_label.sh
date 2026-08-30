#!/bin/bash
# Label pilot tiles with gpt-5.6-luna via Codex CLI (ChatGPT subscription).
set -u
BASE=$HOME/quantization_agent_workspace/vlm_gate
OUT=$BASE/output/_gate_distill/sonnet_pilot
BATCH=6
GUIDE=$(cat $BASE/output/_gate_distill/real_gripper_patched_guidance.txt)
LABELS=$OUT/labels_luna.jsonl
: > "$LABELS"

mapfile -t TILES < <(~/miniconda3/envs/quant_gate_eval/bin/python -c "
import json
m=json.load(open('$OUT/manifest.json'))
print('\n'.join(t['path'] for t in m))")

n=${#TILES[@]}
for ((i=0; i<n; i+=BATCH)); do
  chunk=("${TILES[@]:i:BATCH}")
  IFLAGS=(); for p in "${chunk[@]}"; do IFLAGS+=(-i "$p"); done
  k=${#chunk[@]}
  PROMPT="You are judging $k frames (attached images, in order) from a robot teleoperation episode (task: 'Pick up the doll and put it into the plate'). Each image: LEFT half = external camera, RIGHT half = wrist camera.
For EACH frame in order, output p_yes in [0,1]: probability that the next ~1 second of robot motion can be compressed (half control rate) without changing the outcome.
Guidance:
$GUIDE
Output ONLY a JSON array of $k numbers."
  resp=$(codex exec --skip-git-repo-check -m gpt-5.6-luna "${IFLAGS[@]}" -- "$PROMPT" 2>>$OUT/driver_luna.err | tail -3)
  printf '%s\n' "${chunk[@]}" | RESP="$resp" ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import sys, os, json, re
paths=[l.strip() for l in sys.stdin if l.strip()]
m=re.search(r'\[[-0-9.,eE\s]+\]', os.environ['RESP'])
if not m: sys.exit(0)
try: vals=json.loads(m.group(0))
except Exception: sys.exit(0)
if len(vals)!=len(paths): sys.exit(0)
for p,v in zip(paths,vals):
    b=os.path.basename(p); mm=re.match(r'ep(\d+)_f(\d+)', b)
    if mm: print(json.dumps({'ep':int(mm.group(1)),'f':int(mm.group(2)),'p_yes':float(v)}))
" >> "$LABELS"
  echo "[pilot-luna] $((i+BATCH>n?n:i+BATCH))/$n ($(wc -l < $LABELS) labels)" >&2
done
echo "[pilot-luna] finished: $(wc -l < $LABELS) -> $LABELS" >&2
