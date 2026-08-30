#!/bin/bash
# usage: luna_label_resumable.sh <manifest.json> <out_jsonl>
set -u
MAN=$1; LABELS=$2
BASE=$HOME/quantization_agent_workspace/vlm_gate
BATCH=6
GUIDE=$(cat $BASE/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt)
touch "$LABELS"
mapfile -t TILES < <(~/miniconda3/envs/quant_gate_eval/bin/python -c "
import json
done=set()
for l in open('$LABELS'):
    try:
        r=json.loads(l); done.add((r['ep'],r['f']))
    except Exception: pass
m=json.load(open('$MAN'))
print('\n'.join(t['path'] for t in m if (t['ep'],t['f']) not in done))")
n=${#TILES[@]}
echo "[luna-full] $n tiles to label (resume-aware)" >&2
for ((i=0; i<n; i+=BATCH)); do
  chunk=("${TILES[@]:i:BATCH}")
  IFLAGS=(); for p in "${chunk[@]}"; do IFLAGS+=(-i "$p"); done
  k=${#chunk[@]}
  PROMPT="You are judging $k frames (attached, in order) from RoboCasa simulated kitchen manipulation demos (task visible from scene). Panels: left cam, right cam, wrist cam.
For EACH frame in order: p_yes in [0,1] = probability the next ~1 second of robot motion can be compressed (half control rate) without changing the outcome.
Guidance:
$GUIDE
Output ONLY a JSON array of $k numbers."
  resp=$(codex exec --skip-git-repo-check -m gpt-5.6-luna "${IFLAGS[@]}" -- "$PROMPT" 2>>${LABELS%.jsonl}.err | tail -3)
  printf '%s\n' "${chunk[@]}" | RESP="$resp" ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import sys, os, json, re
paths=[l.strip() for l in sys.stdin if l.strip()]
m=re.search(r'\[[-0-9.,eE\s]+\]', os.environ['RESP'])
if m:
    try:
        vals=json.loads(m.group(0))
        if len(vals)==len(paths):
            for p,v in zip(paths,vals):
                mm=re.match(r'ep(\d+)_f(\d+)', os.path.basename(p))
                if mm: print(json.dumps({'ep':int(mm.group(1)),'f':int(mm.group(2)),'p_yes':float(v)}))
    except Exception: pass
" >> "$LABELS"
done
echo "[luna-full] shard finished: $(wc -l < $LABELS)" >&2
