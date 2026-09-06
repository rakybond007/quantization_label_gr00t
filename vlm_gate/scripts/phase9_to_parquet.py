"""phase9 의 다섯 등급을 학습용 p_yes 로 옮긴다.

라벨링은 등급만 적는다. 그 등급을 하나의 값으로 만드는 규칙은 `PROMPT_METHOD.md`
5단계에서 나온다 -- 부호는 어느 풀에서 뽑힌 문항인지가 정하고, 가중치는 그 문항이
덮는 태스크 수이며 각 변의 합이 1 이 되게 정규화한다. 그 값들이
`phase9_checks.py` 의 SIGN·WEIGHT 다.

    등급 -> 값     (g - 1) / 4        등급표는 한 판단의 강도이므로 눈금이 균등하다
    값 -> 확신     ( 1 + 가점 가중합 - 감점 가중합 ) / 2
    확신 -> p_yes  순위 정규화

순위 정규화는 phase6 라벨(v6b_phase6_softA)이 쓴 것과 같다. p_yes 는 확률이
아니라 순서이고, 그래서 정확히 절반이 0.5 위에 온다. 기존 라벨과 같은 척도를
유지해야 학생 모델 비교가 이어진다. 원값은 p_raw 로 같이 남기므로 다른 척도가
필요하면 거기서 다시 만들면 된다.

**순위는 실제로 학습에 쓸 행에서 매긴다.** 전량에서 순위를 매기고 나중에 부분만
잘라 쓰면 그 부분은 더 이상 균등하지 않다. --cache-dir 을 주면 이미지 캐시에
있는 행만 남기고 그 위에서 순위를 매긴다.

    python phase9_to_parquet.py <labels.jsonl> <out.parquet> [--cache-dir DIR]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase9_checks import NGRADE, SIGN, WEIGHT  # noqa: E402

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
Q = "ABCDE"

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--cache-dir", default="")
ap.add_argument("--ds", default=DS)
a = ap.parse_args()

# 무게는 각 변의 합이 1 이어야 한다. phase9_checks 의 값이 그렇게 돼 있는지 본다.
risk_q = [q for q in Q if SIGN[q] < 0]
safe_q = [q for q in Q if SIGN[q] > 0]
for side, qs in (("감점", risk_q), ("가점", safe_q)):
    s = sum(WEIGHT[q] for q in qs)
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"{side} 무게 합이 {s:.4f} 다. 1 이어야 한다")
print(f"감점 {risk_q}  가점 {safe_q}")
print("무게 " + "  ".join(f"{q}={WEIGHT[q]:.3f}" for q in Q))

keep = None
if a.cache_dir:
    keep = set()
    for p in sorted(glob.glob(f"{a.cache_dir}/index_shard*.parquet")):
        d = pd.read_parquet(p)
        keep |= set(zip(d.episode_index.tolist(), d.frame_index.tolist()))
    print(f"이미지 캐시 {len(keep):,} 프레임으로 제한")

rows = {}
nbad = 0
for line in open(a.src):
    try:
        r = json.loads(line)
    except Exception:
        nbad += 1
        continue
    k = (int(r["ep"]), int(r["f"]))
    if keep is not None and k not in keep:
        continue
    g = [r.get(q) for q in Q]
    if any(x is None or not (1 <= int(x) <= NGRADE) for x in g):
        nbad += 1
        continue
    rows[k] = [int(x) for x in g]
if nbad:
    print(f"버린 줄 {nbad:,}")
if not rows:
    raise SystemExit("남은 행이 없다")

keys = sorted(rows)
V = np.array([rows[k] for k in keys], dtype=np.float64)
G = (V - 1.0) / (NGRADE - 1.0)                      # 등급 -> 0~1

risk = sum(WEIGHT[q] * G[:, Q.index(q)] for q in risk_q)
safe = sum(WEIGHT[q] * G[:, Q.index(q)] for q in safe_q)
raw = (1.0 + safe - risk) / 2.0
raw = np.clip(raw, 0.0, 1.0)

# 동점은 같은 값으로 묶는다. 등급이 정수 다섯 단계이고 무게가 고정이라 확신은
# 이산적이다 -- 25만 행에 서로 다른 값이 백 가지 남짓뿐이다. 여기에 단순
# argsort 순위를 매기면 같은 답을 낸 청크끼리도 순서가 갈리고, 학생은 없는
# 순서를 배운다. phase6 주석이 경고한 자리인데 거기서는 계산 플래그가 연속값이라
# 동점이 드물어 드러나지 않았다.
_u, _inv, _cnt = np.unique(raw, return_inverse=True, return_counts=True)
_start = np.concatenate(([0], np.cumsum(_cnt)[:-1]))
_mid = _start + (_cnt - 1) / 2.0              # 그 값이 차지하는 순위의 가운데
p_yes = (_mid[_inv] / max(1, len(raw) - 1)).astype(np.float64)

instr = {}
for line in open(f"{a.ds}/meta/episodes.jsonl"):
    d = json.loads(line)
    c = [t for t in d.get("tasks", []) if isinstance(t, str)
         and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""
missing = {k[0] for k in keys} - set(instr)
if missing:
    raise SystemExit(f"지시문 없는 에피소드 {len(missing)}개: {sorted(missing)[:5]}")

df = pd.DataFrame({
    "episode_index": [k[0] for k in keys],
    "frame_index": [k[1] for k in keys],
    "task": [instr[k[0]] for k in keys],
    "p_yes": p_yes,
    "p_raw": raw,
    "quantize": (p_yes >= 0.5).astype(int),
})
for i, q in enumerate(Q):
    df[f"q_{q}"] = V[:, i].astype(np.int8)
os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
df.to_parquet(a.dst, index=False)

print(f"\n저장 {a.dst}  {len(df):,}행  에피소드 {df.episode_index.nunique():,}  "
      f"태스크 {df.task.nunique()}종")
print("등급 평균: " + "  ".join(f"{q}={V[:, i].mean():.2f}" for i, q in enumerate(Q)))
print(f"p_raw 평균 {raw.mean():.3f} 표준편차 {raw.std():.3f} "
      f"[{raw.min():.3f}, {raw.max():.3f}]")
tie = float((raw == raw.max()).mean())
print(f"최댓값 동점 {tie:.2%}   서로 다른 raw 값 {len(np.unique(raw)):,}가지")
print(f"p_yes 서로 다른 값 {len(np.unique(p_yes)):,}가지  평균 {p_yes.mean():.3f}  "
      f"0.5 이상 {100*(p_yes >= 0.5).mean():.1f}%")
