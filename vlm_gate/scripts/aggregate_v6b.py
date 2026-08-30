"""v6b(1콜) 라벨 집계 — 계산 위험 플래그 + VLM 장면 문항을 noisy-OR 후 순위정규화.

2콜과 다른 점: 액션 쪽은 VLM 답이 아니라 계산값이 그대로 위험 플래그로 들어간다.
VLM은 계산으로 알 수 없는 장면 정보만 담당한다.
  위험: 계산(grip_transition, reversal, precise_hold, infeasible_merge)
        + VLM(B 좁은 수용부, C 문 당김)
  안전: VLM(A 고정 기구 — 가이던스상 압축 가능, D 빈 공간)
"""
import glob, json, os, numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
TAG=os.environ.get("TAG","v6b")
OUTP=os.path.expanduser(os.environ.get("OUTP",f"~/quantization_agent_workspace/assets/labels/robocasa/{TAG}_1call_full.parquet"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
rows={}
for p in sorted(glob.glob(f"{BASE}/output/_gate_distill/{TAG}_s16_*.jsonl")):
    for l in open(p):
        try:
            r=json.loads(l)
            if all(k in r for k in ("A","B","C","D","grip_transition")): rows[(r["ep"],r["f"])]=r
        except Exception: pass
print(f"수집 {len(rows)}프레임")
keys=sorted(rows)
CR=np.array([[rows[k][c] for c in ("grip_transition","reversal","precise_hold","infeasible_merge")] for k in keys])
V =np.array([[rows[k][q] for q in "ABCD"] for k in keys])
risk=1-np.prod(1-np.column_stack([CR, V[:,1], V[:,2], V[:,3]]), axis=1)   # 계산 4 + VLM B(정렬중),C(기계버튼)
safe=0.5+0.5*V[:,0]                              # VLM A(고정기구), D(빈공간)
raw=(1-risk)*safe
rank=(np.argsort(np.argsort(raw))/(len(raw)-1)).astype(np.float64)
df=pd.DataFrame({"episode_index":[k[0] for k in keys],"frame_index":[k[1] for k in keys],
                 "task":[instr.get(k[0],"") for k in keys],"p_yes":rank,"p_raw":raw,
                 "quantize":(rank>=0.5).astype(int)})
for i,c in enumerate(("grip_transition","reversal","precise_hold","infeasible_merge")): df[f"c_{c}"]=CR[:,i]
for i,q in enumerate("ABCD"): df[f"q_{q}"]=V[:,i]
os.makedirs(os.path.dirname(OUTP),exist_ok=True); df.to_parquet(OUTP,index=False)
print(f"저장 {OUTP}  {len(df)}행  태스크 {df.task.nunique()}종")
print("계산 플래그 발생률: "+"  ".join(f"{c}={CR[:,i].mean():.1%}" for i,c in enumerate(("grip","rev","hold","merge"))))
print("VLM 문항 평균: "+"  ".join(f"{q}={V[:,i].mean():.3f}" for i,q in enumerate("ABCD")))
print(f"raw conf 평균 {raw.mean():.3f} std {raw.std():.3f}")
