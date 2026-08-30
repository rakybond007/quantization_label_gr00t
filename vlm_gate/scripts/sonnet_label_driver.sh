#!/bin/bash
# usage: sonnet_label_driver.sh <manifest.json> <guidance_file> <out_jsonl> <task_desc>
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
  files=$(printf '%s\n' "${chunk[@]}")
  k=${#chunk[@]}
  PROMPT="You are judging $k frames from a robot manipulation episode. Task context: $TASK
Read each image file listed below. For EACH, output p_yes in [0,1]: probability that the next ~1 second of robot motion can be compressed (half control rate) without changing the outcome.
Guidance:
$GUIDE
Output ONLY a JSON array (no prose), one object per file in the same order: [{\"file\": \"<basename>\", \"p_yes\": 0.xx}, ...]
Files:
$files"
  resp=$(claude -p --model sonnet --allowedTools "Read" --max-turns 20 "$PROMPT" 2>>${LABELS%.jsonl}.err)
  echo "$resp" | ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import sys, json, re
raw=sys.stdin.read()
m=re.search(r'\[.*\]', raw, re.S)
if m:
    try:
        for o in json.loads(m.group(0)):
            mm=re.match(r'ep(\d+)_f(\d+)', o.get('file',''))
            if mm: print(json.dumps({'ep':int(mm.group(1)),'f':int(mm.group(2)),'p_yes':float(o['p_yes'])}))
    except Exception: pass
" >> "$LABELS"
  [ $(( (i/BATCH) % 20 )) -eq 0 ] && echo "[sonnet] $((i+k))/$n ($(wc -l < $LABELS) labels)" >&2
done
echo "[sonnet] finished: $(wc -l < $LABELS) -> $LABELS" >&2
