import json, os, time, urllib.request, sys, re
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
OUT=f"{BASE}/output/_gate_distill/exp_{sys.argv[1]}"
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
H={"Authorization":f"Bearer {KEY}"}
def api(path, data=None):
    return json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com"+path, data=data, headers={**H,"Content-Type":"application/json"}), timeout=600))
def upload(p):
    b=b"--X\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--X\r\n"
    b+=f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(p)}"\r\nContent-Type: application/json\r\n\r\n'.encode()
    b+=open(p,'rb').read()+b"\r\n--X--\r\n"
    return json.load(urllib.request.urlopen(urllib.request.Request("https://api.openai.com/v1/files", data=b, headers={**H,"Content-Type":"multipart/form-data; boundary=X"}), timeout=1800))["id"]
res=open(f"{OUT}/labels.jsonl","a"); tin=tout=0
for p in json.load(open(f"{OUT}/files.json")):
    fid=upload(p)
    for attempt in range(40):
        b=api("/v1/batches", json.dumps({"input_file_id":fid,"endpoint":"/v1/chat/completions","completion_window":"24h"}).encode())
        bid=b["id"]; print("submitted",p,bid,flush=True)
        while True:
            time.sleep(150); st=api(f"/v1/batches/{bid}")
            if st["status"] in ("completed","failed","expired","cancelled"): break
        print(bid, st["status"], st["request_counts"], flush=True)
        if st["status"]=="completed": break
        msg=json.dumps((st.get("errors") or {}))
        if "Enqueued token limit" in msg:
            print("한도 대기 후 재시도", flush=True); time.sleep(600); continue
        print("ERR", msg[:200], flush=True); break
    if st["status"]!="completed": continue
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/files/{st['output_file_id']}/content", headers=H), timeout=900).read().decode()
    for line in raw.splitlines():
        r=json.loads(line)
        try:
            ch=r["response"]["body"]["choices"][0]
            c=ch["message"]["content"]
            u=r["response"]["body"].get("usage",{}); tin+=u.get("prompt_tokens",0); tout+=u.get("completion_tokens",0)
            lp=ch.get("logprobs")
            v=None
            if lp and lp.get("content"):
                import math
                for tokinfo in lp["content"]:
                    tops={t["token"].strip().upper(): t["logprob"] for t in tokinfo.get("top_logprobs",[])}
                    if "YES" in tops or "NO" in tops:
                        ly=tops.get("YES", -20.0); ln=tops.get("NO", -20.0)
                        v=math.exp(ly)/(math.exp(ly)+math.exp(ln)); break
            if v is None:
                cs=c.strip()
                if cs.startswith("{"):
                    v=float(json.loads(cs)["p_yes"])
                elif re.search(r"\bYES\b", cs, re.I) and not re.search(r"\bNO\b", cs, re.I): v=1.0
                elif re.search(r"\bNO\b", cs, re.I): v=0.0
                else: v=float(re.search(r"[0-9.]+",cs).group(0))
            n=r["custom_id"]; res.write(json.dumps({"ep":int(n[2:6]),"f":int(n[8:11]),"p_yes":v})+"\n")
        except Exception: pass
    res.flush()
print(f"DONE in={tin} out={tout} cost=${tin/1e6*0.10+tout/1e6*0.60:.2f}", flush=True)
