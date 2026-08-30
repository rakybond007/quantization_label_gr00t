"""액션 전용 호출 — 이미지 없이 숫자만 (2회 분리 호출의 두 번째)"""
import json, os, sys, random, re
import numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
OUT=f"{BASE}/output/_gate_distill/exp_cp_s_act4"; os.makedirs(OUT, exist_ok=True)
info=json.load(open(f"{DS}/meta/info.json"))
# s_vis4와 정확히 같은 프레임 사용
frames=[]
for l in open(f"{BASE}/output/_gate_distill/exp_cp_s_vis4/labels.jsonl"):
    r=json.loads(l); frames.append((r['ep'],r['f']))
if not frames:   # labels 없으면 part 파일에서
    for l in open(f"{BASE}/output/_gate_distill/exp_cp_s_vis4/part_00.jsonl"):
        nm=json.loads(l)["custom_id"]; frames.append((int(nm[2:6]), int(nm[8:11])))
print("프레임", len(frames))
SYS=("You are analysing a robot arm's planned motion. You are given the planned action sequence for the next "
     "16 control steps: 7 numbers per step - end-effector delta x, y, z, then rotation x, y, z, then the "
     "GRIPPER command (last number, 0 = open, 1 = closed).\n"
     "SCALE REFERENCE for this dataset: a step magnitude |d| is typically 0.34; 0.12 is slow (10th percentile), "
     "0.73 is fast (90th percentile); consecutive steps turn by about 11 degrees on average and a turn beyond "
     "90 degrees occurs in only ~1% of steps; while carrying an object |d| is still about 0.33.")
ASK=("Decide these four, reading only the numbers:\n"
     "  E) Does the gripper command change value within the window?\n"
     "  F) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning by more "
     "than 90 degrees?\n"
     "  G) Is the gripper closed while |d| stays below 0.12 for most of the window?\n"
     "  H) Do the magnitudes decrease steadily and end below 0.15?\n"
     "Answer with EXACTLY four characters, one per check in order E,F,G,H, each Y or N. No other text.")
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
    return acts[ep]
rows=[]
for ep,f in frames:
    a=A(ep)
    if a is None or f>=len(a)-4: continue
    txt="Planned actions:\n"+json.dumps(np.round(a[f:f+16,5:],2).tolist())+"\n"+ASK
    rows.append({"custom_id":f"ep{ep:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions",
      "body":{"model":"gpt-5.6-luna","max_completion_tokens":24,"reasoning_effort":"none",
              "messages":[{"role":"system","content":[{"type":"text","text":SYS}]},
                          {"role":"user","content":[{"type":"text","text":txt}]}]}})
paths=[]
for i in range(0,len(rows),1000):
    p=f"{OUT}/part_{i//1000:02d}.jsonl"
    with open(p,"w") as fo:
        for r in rows[i:i+1000]: fo.write(json.dumps(r)+"\n")
    paths.append(p); print(p, round(os.path.getsize(p)/1e6,2),"MB")
json.dump(paths, open(f"{OUT}/files.json","w")); print("요청", len(rows))
