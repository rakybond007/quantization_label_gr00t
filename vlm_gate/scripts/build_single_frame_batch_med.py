import json, base64, os, random
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
OUT=f"{BASE}/output/_gate_distill/openai_single_med"
names=sorted(os.listdir(TIL))
byep={}
for n in names:
    ep=int(n[2:6]); byep.setdefault(ep,[]).append(n)
eps=[e for e in sorted(byep) if len(byep[e])>=25]
random.seed(11); sel=random.sample(eps, 40)
chosen=[n for e in sel for n in sorted(byep[e])]
print("에피소드", len(sel), "프레임", len(chosen))
tail=("p_yes in [0,1] = probability the next ~1 second of robot motion can be compressed "
      "(half control rate) without changing the outcome.\n"
      f"Guidance:\n{G}\nOutput ONLY JSON: {{\"p_yes\": <number>}}")
paths=[]
CH=1200
for i in range(0, len(chosen), CH):
    p=f"{OUT}/single_{i//CH:02d}.jsonl"; paths.append(p)
    with open(p,"w") as f:
        for n in chosen[i:i+CH]:
            b=base64.b64encode(open(f"{TIL}/{n}","rb").read()).decode()
            req={"custom_id": n[:-4], "method":"POST","url":"/v1/chat/completions",
                 "body":{"model":"gpt-5.6-luna","max_completion_tokens":2048,"reasoning_effort":"medium",
                         "messages":[{"role":"user","content":[
                            {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b}"}},
                            {"type":"text","text":"You are judging one frame from a RoboCasa simulated kitchen manipulation demo. Panels: left cam, right cam, wrist cam.\n"+tail}]}]}}
            f.write(json.dumps(req)+"\n")
    print(p, round(os.path.getsize(p)/1e6,1), "MB")
json.dump(paths, open(f"{OUT}/files.json","w"))
