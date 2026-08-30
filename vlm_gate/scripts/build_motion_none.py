import json, base64, os, random
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
OUT=f"{BASE}/output/_gate_distill/openai_motion_none"
have=set(os.listdir(TIL))
byep={}
for n in sorted(have):
    byep.setdefault(int(n[2:6]),[]).append(n)
eps=[e for e in sorted(byep) if len(byep[e])>=25]
random.seed(11); sel=random.sample(eps, 40)   # 이전 실험과 동일 에피소드
reqs=[]
for e in sel:
    fr=sorted(int(n[8:11]) for n in byep[e])
    for f in fr:
        nxt=[f+8, f+16]
        if not all(f"ep{e:04d}_f{x:03d}.png" in have for x in nxt): continue
        reqs.append((e,f,[f]+nxt))
print("모션 요청 수:", len(reqs))
tail=("The three images are CONSECUTIVE moments of the SAME episode, ~0.4s apart (t, t+0.4s, t+0.8s), "
      "so you can see how the motion is evolving.\n"
      "Judge the FIRST image (time t): p_yes in [0,1] = probability the next ~1 second of robot motion "
      "can be compressed (executed at half control rate, merging pairs of actions) without changing the outcome.\n"
      f"Guidance:\n{G}\nOutput ONLY JSON: {{\"p_yes\": <number>}}")
paths=[]; CH=400
for i in range(0, len(reqs), CH):
    p=f"{OUT}/motion_{i//CH:02d}.jsonl"; paths.append(p)
    with open(p,"w") as fo:
        for e,f,fs in reqs[i:i+CH]:
            content=[]
            for x in fs:
                b=base64.b64encode(open(f"{TIL}/ep{e:04d}_f{x:03d}.png","rb").read()).decode()
                content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{b}"}})
            content.append({"type":"text","text":"You are judging a RoboCasa simulated kitchen manipulation demo. Each image has 3 panels: left cam, right cam, wrist cam.\n"+tail})
            fo.write(json.dumps({"custom_id":f"ep{e:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions",
                "body":{"model":"gpt-5.6-luna","max_completion_tokens":64,"reasoning_effort":"none",
                        "messages":[{"role":"user","content":content}]}})+"\n")
    print(p, round(os.path.getsize(p)/1e6,1),"MB")
json.dump(paths, open(f"{OUT}/files.json","w"))
