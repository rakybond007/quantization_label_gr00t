#!/bin/bash
# usage: luna_label_driver.sh <manifest.json> <guidance_file> <out_jsonl> <task_desc>
set -u
MAN=$1; GFILE=$2; LABELS=$3; TASK=$4
BATCH=6
GUIDE=$(cat "$GFILE")
: > "$LABELS"
mapfile -t TILES < <(~/miniconda3/envs/quant_gate_eval/bin/python -c "
import json; m=json.load(open('$MAN')); print('\n'.join(t['path'] for t in m))")
n=${#TILES[@]}
for ((i=0; i<n; i+=BATCH)); do
  chunk=("${TILES[@]:i:BATCH}")
  IFLAGS=(); for p in "${chunk[@]}"; do IFLAGS+=(-i "$p"); done
  k=${#chunk[@]}
  PROMPT="You are judging $k frames (attached images, in order) from a robot teleoperation/demonstration episode. Task context: $TASK
For EACH frame in order, output p_yes in [0,1]: probability that the next ~1 second of robot motion can be compressed (half control rate) without changing the outcome.
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
                b=os.path.basename(p); mm=re.match(r'(?:ep(\d+)_f(\d+))|(?:e(\d+)_f(\d+)_t(.+))', b[:-4])
                if mm and mm.group(1) is not None:
                    print(json.dumps({'ep':int(mm.group(1)),'f':int(mm.group(2)),'p_yes':float(v)}))
                elif mm:
                    print(json.dumps({'ep':int(mm.group(3)),'f':int(mm.group(4)),'task':mm.group(5),'p_yes':float(v)}))
    except Exception: pass
" >> "$LABELS"
  [ $(( (i/BATCH) % 20 )) -eq 0 ] && echo "[luna] $((i+k))/$n ($(wc -l < $LABELS) labels)" >&2
done
echo "[luna] finished: $(wc -l < $LABELS) -> $LABELS" >&2
