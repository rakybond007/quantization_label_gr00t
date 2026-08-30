import json, base64, os, random, urllib.request, re, sys
KEY=open(os.path.expanduser("~/quantization_agent_workspace/openai_key")).read().strip()
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
G=open(f"{BASE}/output/_gate_distill/real_gripper_patched_guidance.txt").read().strip()
d=f"{BASE}/output/_gate_distill/luna_real_full/tiles"
names=sorted(os.listdir(d))
random.seed(2); sample=random.sample(names, 600)
out=open(f"{BASE}/output/_gate_distill/luna_real_full/labels_api_low.jsonl","w")
tail=("For EACH frame in order: p_yes in [0,1] = probability the next ~1 second of robot motion can be compressed (half control rate) without changing the outcome.\n"
f"Guidance:\n{G}\nOutput ONLY a JSON array of numbers.")
tin=tout=0
for i in range(0,len(sample),6):
    chunk=sample[i:i+6]
    content=[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(open(f"{d}/{n}",'rb').read()).decode()}} for n in chunk]
    content.append({"type":"text","text":f"You are judging {len(chunk)} frames (attached, in order) from a real robot teleop episode (task: 'Pick up the doll and put it into the plate'). Each image: LEFT=external cam, RIGHT=wrist cam.\n"+tail})
    body=json.dumps({"model":"gpt-5.6-luna","max_completion_tokens":2048,"reasoning_effort":"low",
                     "messages":[{"role":"user","content":content}]}).encode()
    req=urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    try:
        r=json.load(urllib.request.urlopen(req, timeout=120))
        u=r.get("usage",{}); tin+=u.get("prompt_tokens",0); tout+=u.get("completion_tokens",0)
        m=re.search(r"\[[-0-9.,eE\s]+\]", r["choices"][0]["message"]["content"])
        vals=json.loads(m.group(0)) if m else []
        if len(vals)==len(chunk):
            for n,v in zip(chunk,vals):
                mm=re.match(r"ep(\d+)_f(\d+)", n)
                out.write(json.dumps({"ep":int(mm.group(1)),"f":int(mm.group(2)),"p_yes":float(v)})+"\n")
    except Exception as e:
        print("err",e, file=sys.stderr)
out.close()
cost=tin/1e6*0.20+tout/1e6*1.20
print(f"tokens in={tin} out={tout}, sync cost=${cost:.3f}")
