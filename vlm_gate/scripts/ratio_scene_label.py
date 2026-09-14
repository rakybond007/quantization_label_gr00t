"""국면균등 장면에 계산 사실 + 배율 문항. 정답은 태스크별 실측 최대 배율."""
import importlib, json, os, sys
import numpy as np
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
from vlm_gate import VLMGate
from robocasa_descriptors import descriptors, facts_text
import phase_scorecard as ps
CK=importlib.import_module(os.environ.get("CHECKS","ratio_checks_v3"))
TAG=os.environ.get("TAG","ratio_v3")
TILES=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
S=json.load(open(f"{BASE}/_tmp/selfagg_bias/scenes_phase.json"))
OUT=f"{BASE}/analysis/ratio_prompt/{TAG}.jsonl"
BATCH=int(os.environ.get("RATIO_BATCH","8")); PORT=sys.argv[1]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["tile"])
        except Exception: pass
gate=VLMGate(f"http://127.0.0.1:{PORT}",timeout=300); out=open(OUT,"a")
json.dump({"checks":os.environ.get("CHECKS"),"ngrade":CK.NGRADE,
           "ladder":getattr(CK,"LADDER",None),"sign":CK.SIGN,"weight":CK.WEIGHT},
          open(OUT.replace(".jsonl","_meta.json"),"w"),ensure_ascii=False,indent=1)
buf,n,miss=[],0,0
todo=[s for s in S if s["tile"] not in done]
print(f"{len(todo)} 장면 · {TAG}",flush=True)
for s in todo+[None]:
    if s is not None:
        p=f"{TILES}/{s['tile']}.png"
        if not os.path.exists(p): miss+=1; continue
        try:
            a=ps.get(int(s["ep"])); x=descriptors(a,int(s["f"]))
        except Exception: miss+=1; continue
        buf.append((s,([Image.open(p).convert("RGB")], facts_text(x))))
        if len(buf)<BATCH: continue
    if not buf: continue
    try:
        rs=gate.judge_batch([b[1] for b in buf],CK.GUIDANCE,question=CK.ASK,
                            n_ask=len(CK.SIGN),n_grade=CK.NGRADE,mode="text")
    except Exception as e:
        print(f"batch {type(e).__name__}: {e}",flush=True); buf=[]; continue
    for (s,_),res in zip(buf,rs):
        c=res.get("picks"); Q=sorted(CK.SIGN)
        if res.get("error") or not c or len(c)!=len(Q) or any(v is None for v in c):
            miss+=1; continue
        rec={"tile":s["tile"],"task":s["task"],"ep":s["ep"],"f":s["f"],
             "phase":s["phase"], **{q:int(v) for q,v in zip(Q,c)}}
        if hasattr(CK,"LADDER") and len(Q)==1: rec["ratio"]=CK.LADDER[int(c[0])]
        gp=res.get("grade_probs")
        if gp: rec["gp"]=[[round(float(v),4) for v in row] for row in gp]
        out.write(json.dumps(rec)+"\n"); n+=1
    out.flush(); buf=[]
    if n%160<BATCH: print(f"  {n}/{len(todo)}",flush=True)
out.close(); print(f"완료 {n} · 실패 {miss} -> {OUT}")
