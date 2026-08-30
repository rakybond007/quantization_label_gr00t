import json, sys, pandas as pd, numpy as np
src, dst = sys.argv[1], sys.argv[2]
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
seen={}
for l in open(src):
    try:
        r=json.loads(l); seen[(r['ep'],r['f'])]=r['p_yes']
    except Exception: pass
rows=[{"episode_index":ep,"frame_index":f,"task":instr.get(ep,""),"p_yes":float(p),"quantize":bool(p>=0.5)}
      for (ep,f),p in sorted(seen.items())]
pd.DataFrame(rows).to_parquet(dst)
print(f"{dst}: {len(rows)}행 qrate={np.mean([r['quantize'] for r in rows]):.2f}")
