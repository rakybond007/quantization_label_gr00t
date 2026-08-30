"""phase6 문항 자격 판정 — 전량 라벨링 전에 파일럿으로 거른다.

문항이 슬롯을 얻는 조건 (전부 통과해야 함)
  ① 발화율 0.05 ~ 0.60
  ② 계산 플래그·다른 문항과의 |상관| <= 0.50
  ③ 최소 두 동작군에서 0.10 이상 발화
  ④ 그 문항을 빼도 tau 에서 차단 집합이 10% 이상 바뀔 것
     — 점수 기여가 아니라 결정이 바뀌는지를 본다. 옛 D 문항은 "집계 기여 2.55%p"
       였는데, 옳은 통계는 이것이었다.
"""
import glob, json, os, re
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
rows = []
for f in sorted(glob.glob(f"{BASE}/output/_gate_distill/v6b_phase6_s4_*.jsonl")):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
d = pd.DataFrame(rows)
print(f"파일럿 {len(d)} 청크\n")

SLOTS = ["A","B","C","D","E"]
NAME = {"A":"자유 이동(안전)", "B":"한 방향 부품을 잡음", "C":"경로가 꺾여야 함",
        "D":"무게를 받거나 놓음", "E":"베이스가 함께 움직여야 함"}
COMP = ["grip_transition","reversal","precise_hold","infeasible_merge"]

# 동작군
fam = {"누르기": r"press|button|microwave", "돌리기": r"faucet|spout|knob|burner|stove",
       "열기": r"\bopen\b", "닫기": r"\bclose\b", "집어옮기기": r"pick|place|put|move"}
lab = pd.read_parquet(os.path.expanduser(
    "~/quantization_agent_workspace/assets/labels/robocasa/v6b_phase5_1call_full.parquet"),
    columns=["episode_index","frame_index","task"])
d = d.merge(lab, left_on=["ep","f"], right_on=["episode_index","frame_index"], how="left")

def score(cols):
    """주어진 위험 문항들로 p_raw 를 만든다. A 는 안전 항."""
    risk_src = [d[c].values for c in COMP] + [d[c].values for c in cols]
    risk = 1 - np.prod([1 - np.clip(x,0,1) for x in risk_src], axis=0)
    return (1 - risk) * (0.5 + 0.5 * d["A"].values)

RISK = ["B","C","D","E"]
full = score(RISK)
rank_full = np.argsort(np.argsort(full)) / (len(full) - 1)
blocked_full = rank_full < 0.5

print(f"{'문항':4s} {'발화율':>7s} {'>0.5':>6s} {'표준편차':>8s} {'최대상관':>8s} "
      f"{'발화군':>6s} {'차단변화':>8s}  판정")
for q in SLOTS:
    v = d[q].values
    fire, hi, sd = v.mean(), (v>0.5).mean(), v.std()
    others = [c for c in SLOTS if c != q] + COMP
    mx = max(abs(np.corrcoef(v, d[c].values)[0,1]) for c in others)
    nfam = sum(1 for pat in fam.values()
               if d.loc[d.task.str.contains(pat, case=False, regex=True, na=False), q].mean() >= 0.10)
    if q == "A":
        alt = (1 - (1 - np.prod([1 - np.clip(d[c].values,0,1) for c in COMP+RISK], axis=0))) * 1.0
        jac = np.mean((np.argsort(np.argsort(alt))/(len(alt)-1) < 0.5) != blocked_full)
    else:
        alt = score([c for c in RISK if c != q])
        jac = np.mean((np.argsort(np.argsort(alt))/(len(alt)-1) < 0.5) != blocked_full)
    ok = (0.05 <= fire <= 0.60) and mx <= 0.50 and nfam >= 2 and jac >= 0.10
    print(f"{q:4s} {fire:7.3f} {hi:6.3f} {sd:8.3f} {mx:8.3f} {nfam:6d} {jac:8.3f}  "
          f"{'통과' if ok else '탈락'}   {NAME[q]}")
print("\n기준: 발화율 0.05~0.60 · 최대상관<=0.50 · 발화군>=2 · 차단변화>=0.10")
