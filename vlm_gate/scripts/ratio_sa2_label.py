"""sa2 라벨러. 태스크마다 후보 목록이 다르므로 문항을 장면마다 만든다."""
import json, os, sys
import numpy as np
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
from vlm_gate import VLMGate
from robocasa_descriptors import descriptors, facts_text
import phase_scorecard as ps
import ratio_checks_sa2 as C
TILES=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
S=json.load(open(f"{BASE}/_tmp/selfagg_bias/scenes_phase.json"))
SUB=json.load(open(f"{BASE}/analysis/subaction/robocasa_subactions.json"))["tasks"]
OUT=f"{BASE}/analysis/ratio_prompt/ratio_sa2.jsonl"
PORT=sys.argv[1]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["tile"])
        except Exception: pass
gate=VLMGate(f"http://127.0.0.1:{PORT}",timeout=600); out=open(OUT,"a")
n=miss=0
todo=[s for s in S if s["tile"] not in done and s["task"] in SUB]
print(f"{len(todo)} 장면 · sa2", flush=True)
for s in todo:
    st=[x for x in SUB[s["task"]] if x!="approach"]
    st=list(dict.fromkeys(st))[:5]           # 중복 제거, 최대 5
    if not st: miss+=1; continue
    p=f"{TILES}/{s['tile']}.png"
    if not os.path.exists(p): miss+=1; continue
    try:
        a=ps.get(int(s["ep"])); x=descriptors(a,int(s["f"]))
    except Exception: miss+=1; continue
    r=gate.judge([Image.open(p).convert("RGB")], facts_text(x), C.GUIDANCE,
                 question=C.ask(st), n_ask=1, n_grade=len(st), mode="text")
    c=r.get("picks")
    if r.get("error") or not c or c[0] is None or not (1<=int(c[0])<=len(st)):
        miss+=1; continue
    sub=st[int(c[0])-1]
    rec={"tile":s["tile"],"task":s["task"],"ep":s["ep"],"f":s["f"],"phase":s["phase"],
         "cands":st,"pick":int(c[0]),"sub":sub,"ratio":C.LIMITS[sub]}
    gp=r.get("grade_probs")
    if gp: rec["gp"]=[[round(float(v),4) for v in row] for row in gp]
    out.write(json.dumps(rec)+"\n"); n+=1
    if n%160==0: print(f"  {n}/{len(todo)} (miss {miss})", flush=True)
out.close(); print(f"완료 {n} · 실패 {miss} -> {OUT}", flush=True)
