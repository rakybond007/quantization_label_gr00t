"""2콜 문항별 confidence -> noisy-OR 종합 -> 순위정규화 -> 학습용 parquet.

noisy-OR: 위험 문항(A,B,C,E,F,G) 중 하나라도 강하면 위험. 가중평균은 단일 강증거를
희석시켜 검출률을 무너뜨린다(4.5% vs 71.6%). 안전 문항(D,H)은 이동 구간의 confidence를
밀어 올리는 용도로만 쓴다.

순위정규화: 판정기마다 confidence 캘리브레이션이 달라 같은 τ가 다른 동작점을 뜻한다.
순위로 바꾸면 τ가 곧 차단율이 되어 (티처×아키텍처) 간 τ 비교가 가능해진다.
"""
import glob, json, os, re, numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
OUTP=os.path.expanduser("~/quantization_agent_workspace/assets/labels/robocasa/cosmos_2call_full.parquet")

instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l)
    c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""

rows={}
for p in sorted(glob.glob(f"{BASE}/output/_gate_distill/cosmos_fast_s16_*.jsonl")):
    for l in open(p):
        try:
            r=json.loads(l)
            if all(k in r for k in "ABCDEFGH"): rows[(r["ep"],r["f"])]=r
        except Exception: pass
print(f"수집 {len(rows)}프레임 / 샤드 {len(glob.glob(f'{BASE}/output/_gate_distill/cosmos_fast_s16_*.jsonl'))}개")

keys=sorted(rows)
M=np.array([[rows[k][q] for q in "ABCDEFGH"] for k in keys])       # (N,8)
risk=1-np.prod(1-M[:,[0,1,2,4,5,6]],axis=1)                        # noisy-OR
raw=(1-risk)*(0.5+0.5*M[:,[3,7]].mean(axis=1))
rank=(np.argsort(np.argsort(raw))/(len(raw)-1)).astype(np.float64)  # τ = 차단율

df=pd.DataFrame({"episode_index":[k[0] for k in keys],
                 "frame_index":[k[1] for k in keys],
                 "task":[instr.get(k[0],"") for k in keys],
                 "p_yes":rank, "p_raw":raw, "quantize":(rank>=0.5).astype(int)})
for q,i in zip("ABCDEFGH",range(8)): df[f"q_{q}"]=M[:,i]
os.makedirs(os.path.dirname(OUTP),exist_ok=True)
df.to_parquet(OUTP,index=False)
print(f"저장 {OUTP}  행수 {len(df)}  태스크 {df.task.nunique()}종")
print(f"\n문항별 confidence 평균: "+"  ".join(f"{q}={M[:,i].mean():.3f}" for i,q in enumerate("ABCDEFGH")))
print(f"raw conf 평균={raw.mean():.3f} std={raw.std():.3f}  분위 "
      +" ".join(f"p{int(x*100)}={np.quantile(raw,x):.3f}" for x in (0.1,0.25,0.5,0.75,0.9)))
print("\nτ(순위정규화) 대비 압축률 qrate:")
for t in (0.20,0.30,0.35,0.40,0.50,0.60,0.65,0.70,0.80):
    print(f"  τ={t:.2f} -> qrate {np.mean(rank>=t):.3f}  (차단 {np.mean(rank<t):.1%})")
