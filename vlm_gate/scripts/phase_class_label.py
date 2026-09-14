"""국면 분류 라벨러. 국면균등 장면 -> 예측 국면 + 등급 분포. 정답은 장면 파일에 있다."""
import importlib, json, os, sys
import numpy as np
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
from vlm_gate import VLMGate
CK=importlib.import_module(os.environ.get("CHECKS","phase_class_checks"))
TAG=os.environ.get("TAG","phase_class")
TILES=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
SCENES=f"{BASE}/_tmp/selfagg_bias/scenes_phase.json"
OUT=f"{BASE}/analysis/ratio_prompt/{TAG}.jsonl"
BATCH=int(os.environ.get("RATIO_BATCH","8")); PORT=sys.argv[1]
S=json.load(open(SCENES))
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r=json.loads(l); done.add(r["tile"])
        except Exception: pass
    print(f"이어서 {len(done)}",flush=True)
gate=VLMGate(f"http://127.0.0.1:{PORT}",timeout=300); out=open(OUT,"a")
json.dump({"checks":os.environ.get("CHECKS"),"ngrade":CK.NGRADE,"ladder":CK.LADDER},
          open(OUT.replace(".jsonl","_meta.json"),"w"),ensure_ascii=False,indent=1)
buf,n,miss=[],0,0
todo=[s for s in S if s["tile"] not in done]
print(f"{len(todo)} 장면 · {TAG}",flush=True)
for s in todo+[None]:
    if s is not None:
        p=f"{TILES}/{s['tile']}.png"
        if not os.path.exists(p): miss+=1; continue
        buf.append((s,([Image.open(p).convert("RGB")], s.get("task",""))))
        if len(buf)<BATCH: continue
    if not buf: continue
    try:
        rs=gate.judge_batch([b[1] for b in buf],CK.GUIDANCE,question=CK.ASK,
                            n_ask=1,n_grade=CK.NGRADE,mode="text")
    except Exception as e:
        print(f"batch {type(e).__name__}: {e}",flush=True); buf=[]; continue
    for (s,_),res in zip(buf,rs):
        c=res.get("picks")
        if res.get("error") or not c or c[0] is None: miss+=1; continue
        rec={"tile":s["tile"],"task":s["task"],"true":s["phase"],
             "pred_grade":int(c[0]),"pred":CK.LADDER.get(int(c[0]),"?"),
             "text":res.get("text","")}
        gp=res.get("grade_probs")
        if gp: rec["gp"]=[[round(float(v),4) for v in row] for row in gp]
        out.write(json.dumps(rec)+"\n"); n+=1
    out.flush(); buf=[]
    if n%160<BATCH: print(f"  {n}/{len(todo)}",flush=True)
out.close(); print(f"완료 {n} · 실패 {miss} -> {OUT}")
