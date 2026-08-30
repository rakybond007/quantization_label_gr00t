import json, os, time, urllib.request, sys
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
OUT=f"{BASE}/output/_gate_distill/openai_single_med"
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
H={"Authorization":f"Bearer {KEY}"}
def api(path, data=None, method=None, ct="application/json"):
    req=urllib.request.Request("https://api.openai.com"+path, data=data, headers={**H,"Content-Type":ct}, method=method)
    return json.load(urllib.request.urlopen(req, timeout=600))
def upload(p):
    b=b"--X\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n--X\r\n"
    b+=f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(p)}"\r\nContent-Type: application/json\r\n\r\n'.encode()
    b+=open(p,'rb').read()+b"\r\n--X--\r\n"
    req=urllib.request.Request("https://api.openai.com/v1/files", data=b, headers={**H,"Content-Type":"multipart/form-data; boundary=X"})
    return json.load(urllib.request.urlopen(req, timeout=1200))["id"]
res=open(f"{OUT}/labels_single.jsonl","a")
tin=tout=0
for p in json.load(open(f"{OUT}/files.json")):
    fid=upload(p); print("uploaded", p, flush=True)
    b=api("/v1/batches", json.dumps({"input_file_id":fid,"endpoint":"/v1/chat/completions","completion_window":"24h"}).encode())
    bid=b["id"]; print("submitted", bid, flush=True)
    while True:
        time.sleep(120)
        st=api(f"/v1/batches/{bid}")
        if st["status"] in ("completed","failed","expired","cancelled"): break
    print(bid, st["status"], st["request_counts"], flush=True)
    if st["status"]!="completed":
        print("ERR", (st.get("errors") or {}), flush=True); continue
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/files/{st['output_file_id']}/content", headers=H), timeout=600).read().decode()
    import re
    for line in raw.splitlines():
        r=json.loads(line)
        try:
            c=r["response"]["body"]["choices"][0]["message"]["content"]
            u=r["response"]["body"].get("usage",{}); tin+=u.get("prompt_tokens",0); tout+=u.get("completion_tokens",0)
            v=float(json.loads(c)["p_yes"]) if c.strip().startswith("{") else float(re.search(r"[0-9.]+",c).group(0))
            n=r["custom_id"]; res.write(json.dumps({"ep":int(n[2:6]),"f":int(n[8:11]),"p_yes":v})+"\n")
        except Exception: pass
    res.flush()
print(f"DONE tokens in={tin} out={tout} batch_cost=${tin/1e6*0.10+tout/1e6*0.60:.3f}", flush=True)
