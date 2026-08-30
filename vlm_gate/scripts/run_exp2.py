"""재개 가능한 배치 러너: 파트별 상태 저장, 인플라이트 배치 재수거"""
import json, os, time, urllib.request, sys, re, math
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
tag=sys.argv[1]; GRADE20 = "grade20" in tag; GRADE10 = "grade10" in tag; BITS4 = "bits4" in tag or "bits4cal" in tag or "act4" in tag; LADDER = "ladder" in tag; BITS8 = "bits8" in tag or "bits8cal" in tag or "seq8" in tag; VOTE = "vote" in tag; OUT=f"{BASE}/output/_gate_distill/exp_{tag}"
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
H={"Authorization":f"Bearer {KEY}"}
ST=f"{OUT}/state.json"
st=json.load(open(ST)) if os.path.exists(ST) else {"inflight":{}, "done":[]}
def save(): json.dump(st, open(ST,"w"))
def api(path, data=None):
    return json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com"+path, data=data, headers={**H,"Content-Type":"application/json"}), timeout=900))
def upload(p):
    b=b"--X\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--X\r\n"
    b+=f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(p)}"\r\nContent-Type: application/json\r\n\r\n'.encode()
    b+=open(p,'rb').read()+b"\r\n--X--\r\n"
    return json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/files", data=b, headers={**H,"Content-Type":"multipart/form-data; boundary=X"}), timeout=1800))["id"]
res=open(f"{OUT}/labels.jsonl","a")
def collect(bid):
    while True:
        s=api(f"/v1/batches/{bid}")
        if s["status"] in ("completed","failed","expired","cancelled"): break
        time.sleep(150)
    print(bid, s["status"], s["request_counts"], flush=True)
    if s["status"]!="completed" or not s.get("output_file_id"): return 0
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/files/{s['output_file_id']}/content", headers=H), timeout=900).read().decode()
    n=0
    for line in raw.splitlines():
        r=json.loads(line)
        try:
            if VOTE:
                chs=r["response"]["body"]["choices"]
                yes=sum(1 for cc in chs if "YES" in (cc["message"]["content"] or "").upper())
                v=yes/max(len(chs),1)
                nm=r["custom_id"]; res.write(json.dumps({"ep":int(nm[2:6]),"f":int(nm[8:11]),"p_yes":v})+"\n"); n+=1
                continue
            ch=r["response"]["body"]["choices"][0]; c=ch["message"]["content"]; lp=ch.get("logprobs"); v=None
            if lp and lp.get("content"):
                if BITS8:
                    txt=(c or "").strip().upper()
                    ys=[ch for ch in txt if ch in "YN"][:8]
                    if len(ys)==8:
                        risk=sum(1 for ch in (ys[0],ys[1],ys[2],ys[4],ys[5],ys[6],ys[7]) if ch=="Y")
                        gross = ys[3]=="Y"
                        v=(7-risk)/7.0*(1.0 if gross else 0.7)
                if LADDER:
                    txt=(c or "").strip().upper()
                    ys=[ch for ch in txt if ch in "YN"][:4]
                    if len(ys)==4:
                        v=sum(1 for ch in ys if ch=="Y")/4.0
                if BITS4:
                    txt=(c or "").strip().upper()
                    ys=[ch for ch in txt if ch in "YN"][:4]
                    if len(ys)==4:
                        risk=sum(1 for ch in ys if ch=="Y")
                        v=1.0-risk/4.0
                SCALE={"CERTAIN":1.0,"LIKELY":0.75,"UNSURE":0.5,"DOUBTFUL":0.25,"IMPOSSIBLE":0.0,
                       "16":1.0,"12":0.75,"8":0.5,"4":0.25,"0":0.0}
                if GRADE10:
                    tot=w=0.0
                    for t in lp["content"][0].get("top_logprobs",[]) or []:
                        tok=t["token"].strip()
                        if tok.isdigit() and 1<=int(tok)<=10:
                            pr=math.exp(t["logprob"]); w+=pr*(int(tok)-1)/9.0; tot+=pr
                    if tot>0: v=w/tot
                if GRADE20:
                    tot=w=0.0
                    for t in lp["content"][0].get("top_logprobs",[]) or []:
                        tok=t["token"].strip()
                        if tok.isdigit() and 1<=int(tok)<=20:
                            pr=math.exp(t["logprob"]); w+=pr*(int(tok)-1)/19.0; tot+=pr
                    if tot>0: v=w/tot
                for ti in lp["content"]:
                    py=pn=0.0; sw=sp_=0.0
                    for t in ti.get("top_logprobs",[]):
                        tok=t["token"].strip().upper(); pr=math.exp(t["logprob"])
                        for k,val in SCALE.items():
                            if tok and k.startswith(tok) and len(tok)>=3: sw+=pr*val; sp_+=pr; break
                        else:
                            if tok.startswith("YES"): py+=pr
                            elif tok.startswith("NO"): pn+=pr
                    if sp_>0.05: v=sw/sp_; break          # 5단계 척도 기대값
                    if py>0 or pn>0: v=py/(py+pn); break  # YES/NO 확률비
            if v is None:
                cu=(c or "").upper()
                if "YES" in cu and "NO" not in cu: v=1.0
                elif "NO" in cu and "YES" not in cu: v=0.0
                else: v=float(json.loads(c)["p_yes"]) if c.strip().startswith("{") else float(re.search(r"[0-9.]+",c).group(0))
            nm=r["custom_id"]; res.write(json.dumps({"ep":int(nm[2:6]),"f":int(nm[8:11]),"p_yes":v})+"\n"); n+=1
        except Exception: pass
    res.flush(); return n
for p,bid in list(st["inflight"].items()):
    print("resume collect", p, bid, flush=True)
    if collect(bid): st["done"].append(p)
    st["inflight"].pop(p, None); save()
for p in json.load(open(f"{OUT}/files.json")):
    if p in st["done"]: continue
    fid=upload(p); b=api("/v1/batches", json.dumps({"input_file_id":fid,"endpoint":"/v1/chat/completions","completion_window":"24h"}).encode())
    st["inflight"][p]=b["id"]; save(); print("submitted",os.path.basename(p),b["id"],flush=True)
    if collect(b["id"]): st["done"].append(p)
    st["inflight"].pop(p, None); save()
print("DONE", tag, flush=True)
