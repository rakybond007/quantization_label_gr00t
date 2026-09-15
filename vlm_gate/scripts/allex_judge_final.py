"""판정기 비교 최종 채점. **네 범주 정답지**에서, 각자 자기 가중으로.

정답지는 프레임을 직접 보고 표시했다(`_tmp/eye_labels.json`):

    2 양손상자  실측 0.533   A(PINCH_RIGID) 의 표적
    1 한손봉투       0.767   B(POUCH) 의 표적
    0 한손상자       0.930
    3 Pass           0.98    C(FREE_END) 의 표적

**가중은 판정기마다 따로 잡는다**(하네스 R5). 판정기가 바뀌면 라벨이 바뀌므로
한 가중을 셋에 쓰면 그 가중을 맞춘 판정기만 유리하다 -- 실제로 Cosmos 가중으로
재면 Sonnet 의 순위상관이 +0.50, 자기 가중으로는 +1.00 이었다.

    python allex_judge_final.py
"""
import json, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
BASE=os.path.dirname(HERE)
import allex_v4c_checks as C
from scipy.stats import spearmanr

EYE={tuple(int(x) for x in k.split("_")):v for k,v in
     json.load(open(f"{BASE}/_tmp/eye_labels.json")).items() if v>=0}
GT={2:0.533, 1:0.767, 0:0.930, 3:0.980}
ORDER=[2,1,0,3]
NAME={2:"양손상자",1:"한손봉투",0:"한손상자",3:"Pass"}
SRC={"cosmos":"output/allex_cosmos_cmp/records.jsonl",
     "sonnet":"output/allex_api_sonnet.jsonl",
     "gemini3.8f":"output/allex_api_gemini38flash.jsonl"}
Q=sorted(C.SIGN)

def load(p):
    p=f"{BASE}/{p}"
    return {(r["ep"],r["f"]):r for r in (json.loads(l) for l in open(p))} if os.path.exists(p) else {}

print(f"정답지 {len(EYE)}청크 · " + " · ".join(
    f"{NAME[c]} {sum(1 for v in EYE.values() if v==c)}" for c in ORDER))
print()
print(f"{'판정기':11s} {'n':>4s} {'A차':>6s} {'B차':>6s} {'C차':>6s} | "
      f"{'ρ':>5s} {'A':>4s} {'B':>4s} {'C':>4s} {'D':>4s} | " +
      " ".join(f"{NAME[c]:>8s}" for c in ORDER) + f" {'간격':>6s} {'sd':>6s}")
for tag,p in SRC.items():
    L=load(p); ks=[k for k in EYE if k in L]
    if len(ks) < 40:
        print(f"  {tag:9s} {len(ks):4d}  (라벨 부족)"); continue
    cell=np.array([EYE[k] for k in ks])
    G=np.array([[L[k][q] for q in Q] for k in ks], float); g=(G-1)/4
    # 문항별 표적 분리도: 표적 범주 vs 나머지
    tgt={"A":2,"B":1,"C":3}
    dif={q: G[cell==tgt[q], Q.index(q)].mean() - G[cell!=tgt[q], Q.index(q)].mean()
         for q in ("A","B","C") if (cell==tgt[q]).any()}
    gtv=[GT[c] for c in ORDER]
    best=None
    for wa in np.arange(0.0,1.01,0.05):
      for wc in np.arange(0.0,1.01,0.05):
        W=np.array([-wa,-(1-wa),wc,1-wc]); cf=0.5*(1+g@W)
        m=[cf[cell==c].mean() for c in ORDER]
        r=spearmanr(gtv,m).statistic
        key=(round(r,3), m[-1]-m[0])
        if best is None or key>best[0]: best=(key,wa,wc,m,cf)
    (r,gap),wa,wc,m,cf=best
    print(f"  {tag:9s} {len(ks):4d} {dif.get('A',float('nan')):+6.2f} "
          f"{dif.get('B',float('nan')):+6.2f} {dif.get('C',float('nan')):+6.2f} | "
          f"{r:+5.2f} {wa:4.2f} {1-wa:4.2f} {wc:4.2f} {1-wc:4.2f} | " +
          " ".join(f"{x:8.3f}" for x in m) + f" {gap:6.3f} {cf.std():6.3f}")
print("\n  A차/B차/C차 = 그 문항이 자기 표적 범주에서 나머지보다 얼마나 높은가 (클수록 좋다)")
print("  ρ = conf 가 실측 네 셀 순서를 재현하는 정도 · 간격 = Pass conf - 양손상자 conf")
