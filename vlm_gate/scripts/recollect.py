"""완료된 배치 결과를 재과금 없이 다시 수거"""
import json, os, sys, re, math, urllib.request
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
H={"Authorization":f"Bearer {KEY}"}
tag=sys.argv[1]; MODE=sys.argv[2]      # bits4 | bits8 | ladder
OUT=f"{BASE}/output/_gate_distill/exp_{tag}"
bids=re.findall(r"batch_[a-z0-9]+", open(f"{OUT}/run.log").read())
res=open(f"{OUT}/labels.jsonl","w"); n=0
for bid in dict.fromkeys(bids):
    s=json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/batches/{bid}", headers=H), timeout=300))
    if s["status"]!="completed" or not s.get("output_file_id"): continue
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/files/{s['output_file_id']}/content", headers=H), timeout=900).read().decode()
    for line in raw.splitlines():
        try:
            r=json.loads(line)
            c=r["response"]["body"]["choices"][0]["message"]["content"]
            ys=[ch for ch in (c or "").strip().upper() if ch in "YN"]
            if MODE=="bits8" and len(ys)>=8:
                ys=ys[:8]; risk=sum(1 for ch in (ys[0],ys[1],ys[2],ys[4],ys[5],ys[6],ys[7]) if ch=="Y")
                v=(7-risk)/7.0*(1.0 if ys[3]=="Y" else 0.7)
            elif MODE=="bits4" and len(ys)>=4:
                v=1.0-sum(1 for ch in ys[:4] if ch=="Y")/4.0
            elif MODE=="ladder" and len(ys)>=4:
                v=sum(1 for ch in ys[:4] if ch=="Y")/4.0
            else: continue
            nm=r["custom_id"]; res.write(json.dumps({"ep":int(nm[2:6]),"f":int(nm[8:11]),"p_yes":v})+"\n"); n+=1
        except Exception: pass
res.close(); print(f"{tag}: {n}건 재수거")
