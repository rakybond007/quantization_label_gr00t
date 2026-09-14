"""에피소드 시작 장면 라벨러 -- 폐루프 성패가 이미 알려진 에피소드에 장면 문항을 건다.

앞선 배속 라벨러들은 태스크당 한 줄을 냈다. 여기서는 **에피소드당 한 줄**을 낸다.
채점도 태스크 상관이 아니라 태스크 안 성공/실패 분리로 한다(하네스 R1).

    python ratio_ep_scene_label.py <port> <index.json>
"""
import importlib, json, os, sys
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
from vlm_gate import VLMGate
CK=importlib.import_module(os.environ.get("CHECKS","ratio_checks_scene"))
TAG=os.environ.get("TAG","ep_scene")
OUT=f"{BASE}/analysis/ratio_prompt/{TAG}.jsonl"
PORT=sys.argv[1]; IDX=sys.argv[2]
BATCH=int(os.environ.get("RATIO_BATCH","6"))
INSTR={
 "PnPCounterToSink":"pick the object up from the counter and put it in the sink",
 "CloseDoubleDoor":"close both cabinet doors",
 "OpenDoubleDoor":"open both cabinet doors",
 "CoffeeServeMug":"place the mug under the coffee machine dispenser",
}
Q=sorted(CK.SIGN)
rows=json.load(open(IDX))
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
    print(f"이어서 {len(done)}",flush=True)
todo=[r for r in rows if f"{r['task']}_{r['ep']}" not in done]
print(f"{len(todo)} 에피 · {TAG} · 문항 {''.join(Q)}",flush=True)
gate=VLMGate(f"http://127.0.0.1:{PORT}",timeout=300); out=open(OUT,"a")
buf,n,miss=[],0,0
for r in todo+[None]:
    if r is not None:
        if not os.path.exists(r["img"]): miss+=1; continue
        buf.append((r,([Image.open(r["img"]).convert("RGB")],
                       INSTR.get(r["task"], r["task"]))))
        if len(buf)<BATCH: continue
    if not buf: continue
    try:
        rs=gate.judge_batch([b[1] for b in buf],CK.GUIDANCE,question=CK.ASK,
                            n_ask=len(Q),n_grade=CK.NGRADE,mode="text")
    except Exception as e:
        print(f"batch {type(e).__name__}: {e}",flush=True); buf=[]; continue
    for (r,_),res in zip(buf,rs):
        c=res.get("picks")
        if res.get("error") or not c or len(c)<len(Q) or any(x is None for x in c):
            miss+=1; continue
        rec={"key":f"{r['task']}_{r['ep']}","task":r["task"],"ep":r["ep"],
             "success":bool(r["success"]),"text":res.get("text","")}
        for i,q in enumerate(Q): rec[q]=int(c[i])
        gp=res.get("grade_probs")
        if gp: rec["gp"]=[[round(float(v),4) for v in row] for row in gp]
        out.write(json.dumps(rec)+"\n"); n+=1
    out.flush(); buf=[]
    print(f"  {n}/{len(todo)}",flush=True)
out.close(); print(f"완료 {n} · 실패 {miss} -> {OUT}")
