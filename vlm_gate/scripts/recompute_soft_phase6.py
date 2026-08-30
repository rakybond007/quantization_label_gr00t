"""phase6(5문항) 라벨의 계산 플래그를 연속값으로 다시 계산해 재집계한다. VLM 재호출 없음.

phase5 용 recompute_soft_labels.py 와 같은 처리(softA: 네 플래그 모두 연속)이고,
문항 역할만 phase6 배치를 따른다 — A 안전, B·C·D·E 위험.
이렇게 두면 phase5-softA 와 계산층·집계식이 같아지고 프롬프트만 달라진다."""
import json, os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from robocasa_descriptors_soft import soft_risk

WS=os.path.expanduser("~/quantization_agent_workspace")
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
SRC=f"{WS}/assets/labels/robocasa/v6b_phase6_full.parquet"
OUT=os.environ.get("OUT", f"{WS}/assets/labels/robocasa/v6b_phase6_softA.parquet")

d=pd.read_parquet(SRC)
cs=json.load(open(f"{DS}/meta/info.json"))["chunks_size"]
S=np.zeros((len(d),4)); ok=np.zeros(len(d),bool)
t0=time.time()
for ep, idx in d.groupby("episode_index").groups.items():
    try: a=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ep//cs:03d}/episode_{ep:06d}.parquet")["action"].values)
    except Exception: continue
    pos=d.index.get_indexer(idx)
    for p, f in zip(pos, d.loc[idx,"frame_index"].values):
        if f < len(a)-4:
            r=soft_risk(a, int(f)); S[p]=[r["grip_transition"],r["reversal"],r["precise_hold"],r["infeasible_merge"]]; ok[p]=True
print(f"재계산 {ok.sum()}/{len(d)} ({time.time()-t0:.0f}초)")

C=["c_grip_transition","c_reversal","c_precise_hold","c_infeasible_merge"]
for i,c in enumerate(C): d[c+"_soft"]=S[:,i]
V=d[["q_A","q_B","q_C","q_D","q_E"]].values
# 집계식은 그대로. 계산 플래그만 이진 -> 연속으로 교체.
risk=1-np.prod(1-np.column_stack([S, V[:,1], V[:,2], V[:,3], V[:,4]]), axis=1)  # B,C,D,E 위험
safe=0.5+0.5*V[:,0]
raw=(1-risk)*safe
d["p_raw_soft"]=raw
d["p_yes_soft"]=(np.argsort(np.argsort(raw))/(len(raw)-1))
d.loc[~ok, ["p_raw_soft","p_yes_soft"]]=np.nan
print(f"집계 완료 (저장은 진단 뒤)")
print(f"\n연속 플래그 평균: "+"  ".join(f"{c.replace('c_','')}={S[:,i].mean():.4f}" for i,c in enumerate(C)))
print(f"p_raw==0 비율: 기존 {(d.p_raw==0).mean():.4f} -> 연속 {(d.p_raw_soft==0).mean():.4f}")
print(f"p_raw 분포(연속): 25%={np.nanpercentile(raw,25):.3f} 50%={np.nanpercentile(raw,50):.3f} 75%={np.nanpercentile(raw,75):.3f}")
sp=d[["p_yes","p_yes_soft"]].dropna().corr(method="spearman").iloc[0,1]
print(f"기존 순위와의 스피어만 상관: {sp:.4f}")

# train_gate_module.py 는 p_yes 를 읽는다. phase5 softA 파케이도 같은 방식으로
# p_yes 를 연속값으로 덮어써 뒀다 — 안 맞추면 이진 라벨로 학습된다.
d["p_raw"]=d["p_raw_soft"]; d["p_yes"]=d["p_yes_soft"]
d=d[d.p_yes.notna()].reset_index(drop=True)
d.to_parquet(OUT, index=False)
print(f"저장 {OUT}  {len(d)}행")
