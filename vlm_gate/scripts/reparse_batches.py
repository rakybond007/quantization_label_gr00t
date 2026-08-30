import json, os, re, sys, urllib.request, math
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
H={"Authorization":f"Bearer {KEY}"}
tag=sys.argv[1]; ids=sys.argv[2:]
OUT=f"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/output/_gate_distill/exp_{tag}"
res=open(f"{OUT}/labels.jsonl","a"); n=0
for bid in ids:
    st=json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/batches/{bid}", headers=H), timeout=300))
    if st["status"]!="completed" or not st.get("output_file_id"):
        print(bid, st["status"], "(건너뜀)"); continue
    raw=urllib.request.urlopen(urllib.request.Request(f"https://api.openai.com/v1/files/{st['output_file_id']}/content", headers=H), timeout=900).read().decode()
    for line in raw.splitlines():
        r=json.loads(line)
        try:
            ch=r["response"]["body"]["choices"][0]; c=ch["message"]["content"]; v=None
            lp=ch.get("logprobs")
            if lp and lp.get("content"):
                for t in lp["content"]:
                    tops={x["token"].strip().upper(): x["logprob"] for x in t.get("top_logprobs",[])}
                    if "YES" in tops or "NO" in tops:
                        ly=tops.get("YES",-20.0); ln=tops.get("NO",-20.0)
                        v=math.exp(ly)/(math.exp(ly)+math.exp(ln)); break
            if v is None:
                cs=c.strip()
                if cs.startswith("{"): v=float(json.loads(cs)["p_yes"])
                elif re.search(r"\bYES\b",cs,re.I) and not re.search(r"\bNO\b",cs,re.I): v=1.0
                elif re.search(r"\bNO\b",cs,re.I): v=0.0
                else: v=float(re.search(r"[0-9.]+",cs).group(0))
            nm=r["custom_id"]; res.write(json.dumps({"ep":int(nm[2:6]),"f":int(nm[8:11]),"p_yes":v})+"\n"); n+=1
        except Exception: pass
res.close(); print(f"{tag}: 재파싱 {n}건")
