"""모든 폐루프 eval 결과를 출처(체크포인트·τ·clip·K)와 함께 1회 수집 → analysis/results_db.json"""
import json, os, glob, re
import numpy as np
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
OUT=f"{BASE}/output/robocasa"
def parse_pred(d):
    s=n=0; steps=[]; cfg={}
    for f in glob.glob(f"{d}/*/prediction.txt"):
        txt=open(f).read()
        for m in re.finditer(r"is_success: \[\s*(\w+)\] action_steps: (\d+)", txt):
            n+=1; ok=m.group(1)=="True"; s+=ok
            if ok: steps.append(int(m.group(2)))
        for key in ["compress_k","tau","judge_threshold","compensate","clip_scale","vark_bound","gate_k3_threshold"]:
            mm=re.search(rf"^{key}: (\S+)", txt, re.M)
            if mm: cfg[key]=mm.group(1)
    if not n: return None
    return {"success": round(s/n,4), "steps_mean": round(float(np.mean(steps)),1) if steps else None,
            "episodes": n, "tasks": len(glob.glob(f"{d}/*/prediction.txt")), "config": cfg}
def parse_judge(d):
    for f in sorted(glob.glob(f"{d}/judge-*.log")):
        for l in open(f, errors="ignore"):
            m=re.search(r"JUDGE READY \(([^,]+), ([^,)]+)", l)
            if m: return {"backend": m.group(1).strip(), "ckpt": m.group(2).strip()}
    for f in sorted(glob.glob(f"{d}/server-*.log")):
        for l in open(f, errors="ignore"):
            m=re.search(r"model_path[= ]+(\S+)", l)
            if m: return {"backend":"internal/policy","ckpt":m.group(1)}
    return {"backend":"?", "ckpt":"?"}
def parse_gate(d):
    c=[]; q=[]
    for f in glob.glob(f"{d}/*/gate_conf.csv"):
        for l in open(f, errors="ignore"):
            p=l.strip().split(",")
            try:
                c.append(float(p[2]))
                if len(p)>3 and p[3] in "01": q.append(int(p[3]))
            except Exception: pass
    if not c: return None
    c=np.array(c)
    return {"n_calls": len(c), "qrate": round(float(np.mean(q)) if q else float("nan"),3),
            "conf_max": round(float(c.max()),3), "conf_p50": round(float(np.percentile(c,50)),3),
            "conf_p90": round(float(np.percentile(c,90)),3)}
db={}
for d in sorted(glob.glob(f"{OUT}/*")):
    if not os.path.isdir(d): continue
    name=os.path.basename(d)
    r=parse_pred(d)
    if not r: continue
    r["provenance"]=parse_judge(d)
    g=parse_gate(d)
    if g: r["gate"]=g
    r["mtime"]=os.path.getmtime(d)
    db[name]=r
json.dump(db, open(f"{BASE}/analysis/results_db.json","w"), ensure_ascii=False, indent=1)
print(f"수집 완료: {len(db)}개 실행")
for k,v in sorted(db.items(), key=lambda x:-x[1]["success"])[:8]:
    ck=os.path.basename(os.path.dirname(v["provenance"]["ckpt"])) if v["provenance"]["ckpt"]!="?" else "?"
    par=v["provenance"]["ckpt"].split("/")[-3] if v["provenance"]["ckpt"].count("/")>2 else ""
    print(f"  {k:38s} succ={v['success']:.3f} steps={v['steps_mean']} eps={v['episodes']} ← {par}/{ck}")
