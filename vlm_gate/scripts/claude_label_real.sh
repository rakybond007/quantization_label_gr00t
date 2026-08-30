#!/bin/bash
# usage: claude_label_real.sh <model> <worker_id> <n_workers>
set -u
M=$1; WID=$2; NW=$3
BASE=$HOME/quantization_agent_workspace/vlm_gate
D=$BASE/output/_gate_distill/luna_real_full
G=$(cat $BASE/output/_gate_distill/real_gripper_patched_guidance.txt)
LB=$D/labels_claude_${M}_w$WID.jsonl
touch "$LB"
# 동일 600프레임 샘플(seed 2) 목록
mapfile -t TILES < <(~/miniconda3/envs/quant_gate_eval/bin/python -c "
import os, random, json
d='$D/tiles'; names=sorted(os.listdir(d))
random.seed(2); s=random.sample(names,600)
done=set()
for l in open('$LB'):
    try: r=json.loads(l); done.add((r['ep'],r['f']))
    except: pass
out=[]
for i,n in enumerate(s):
    if i%$NW!=$WID: continue
    ep=int(n[2:5]); f=int(n[6:9].lstrip('f') or n.split('_f')[1][:3])
    import re; m=re.match(r'ep(\d+)_f(\d+)', n); ep,f=int(m.group(1)),int(m.group(2))
    if (ep,f) not in done: out.append(n)
print('\n'.join(out))")
for n in "${TILES[@]}"; do
  [ -z "$n" ] && continue
  resp=$(printf '%s\n' "Read the image $D/tiles/$n . It is one frame from a real robot teleoperation episode (task: pick-and-place on a table). LEFT half = external camera, RIGHT half = wrist camera.
p_yes in [0,1] = probability that the next ~1 second of robot motion can be compressed (half control rate) without changing the outcome.
Guidance:
$G
Output ONLY JSON: {\"p_yes\": <number>}" | timeout 180 claude -p --model $M --allowedTools "Read" 2>>$D/claude_${M}.err | tail -2)
  N="$n" RESP="$resp" ~/miniconda3/envs/quant_gate_eval/bin/python -c "
import os, re, json
m=re.search(r'p_yes\W+([0-9.]+)', os.environ['RESP'])
if m:
    n=os.environ['N']; mm=re.match(r'ep(\d+)_f(\d+)', n)
    print(json.dumps({'ep':int(mm.group(1)),'f':int(mm.group(2)),'p_yes':float(m.group(1))}))
" >> "$LB"
done
echo "[${M}-w$WID] done $(wc -l < $LB)" >&2
