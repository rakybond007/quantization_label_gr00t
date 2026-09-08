"""libero 최종 라벨 -- 매 시점에 **몇 배속으로 갈지**를 하나의 값으로 적는다.

지금까지는 두 갈래였다. F_level 이 접촉 가드로 level 1 을 강제하고, 가드가 서지
않으면 movement head 가 K 를 골랐다. 우리는 따로 confidence 를 만들었다.
그것을 하나로 합친다 (2026-09-08 사용자 지시).

    접촉이 잡히면            -> 1.0배 고정
    접촉이 아니면            -> band_place( VLM 신뢰도, 그 태스크의 [lo, hi] )

**배속은 연속값으로 적지 않는다.** 디코더가 이산이라 연속값을 적으면 결국
반올림해서 쓰게 되고, 그러면 결정이 반올림에서 일어난다. 처음부터 디코더가 가진
눈금으로 적는다. 그 눈금에서는 배속이 정확히 떨어진다
(`fractional_blocks.block_sizes` + `replan_rows`, `RATIOS_ARE_NOT_K.md`).

    GRID = (1.0, 1.5, 2.0, 2.5)

상한표는 **클립만 푼 사다리**에서 나온다. 구동부를 3배로 연 실행
(`released_*`)은 1.429배에서 이미 0.616 으로 무너져 압축이 아니라 제어기 손상을
재고 있으므로 상한표에 쓰지 않는다.

    python vlm_gate/scripts/libero_ratio_label.py <labels.jsonl> [out.jsonl]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SLOTS = "ABCDE"
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1}
WEIGHT = {"A": 0.636, "B": 0.364, "C": 0.533, "D": 0.267, "E": 0.200}

# 디코더가 가진 눈금. F_level 의 level_ks 와 같아야 한다.
GRID = (1.0, 1.5, 2.0, 2.5)

# 태스크별 상한. `derive_libero_ceilings.py` 가 클립만 푼 사다리에서 만든다.
# 1.5배 칸이 들어오기 전까지는 비어 있고, 없는 태스크는 DEFAULT 를 쓴다.
CEIL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "analysis", "libero_task_ceilings.json")
DEFAULT_BAND = (1.0, 2.0)


def confidence(picks):
    """(1 + Σw·가점 − Σw·감점) / 2. 등급은 (g−1)/4 로 0~1 에 옮긴다."""
    risk = sum(WEIGHT[k] * (picks[k] - 1) / 4.0 for k in SLOTS if SIGN[k] < 0)
    safe = sum(WEIGHT[k] * (picks[k] - 1) / 4.0 for k in SLOTS if SIGN[k] > 0)
    return float(min(1.0, max(0.0, (1.0 + safe - risk) / 2.0)))


def band_place(confs, lo, hi):
    """신뢰도를 그 태스크의 띠에 앉힌다. 칸 안 **분위수**로 앉힌다.

    신뢰도의 절대 크기는 등급표 눈금이 정하는 임의값이라 그대로 띠에 곱하면
    안 된다 -- 눈금을 늘리기만 해도 배속이 바뀐다. 순서만 뜻이 있다.
    같은 답을 낸 시점은 같은 분위수를 받아 같은 값으로 묶인다.

    `allex_v3_checks.band_place` 와 같은 규칙이다.
    """
    c = np.asarray(confs, dtype=float)
    n = c.size
    if n == 0:
        return c
    if n == 1:
        return np.array([lo + 0.5 * (hi - lo)])
    order = c.argsort(kind="mergesort")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    for v in np.unique(c):
        m = c == v
        rank[m] = rank[m].mean()
    return lo + (rank / (n - 1)) * (hi - lo)


def snap(x):
    """눈금 위의 가장 가까운 값으로. 라벨은 디코더가 실제로 낼 수 있는 값이어야 한다."""
    g = np.asarray(GRID, dtype=float)
    return float(g[int(np.argmin(np.abs(g - float(x))))])


def contact(rec):
    """이 시점이 접촉인가. 계산으로 끝난 것만 쓴다 -- VLM 에게 묻지 않는다.

    `libero_descriptors.computed_risk` 의 `grip_transition` 이 그리퍼가 이 창에서
    열리거나 닫힌다는 뜻이고, 그것이 곧 무게가 옮겨 가는 순간이다. 여기서
    몇 밀리미터가 잡혔는지 떨어졌는지를 가른다.

    `precise_hold`(들고 기어가는 중)도 접촉으로 센다 -- 쥔 채로 느리게 가는 것은
    놓을 자리를 맞추는 중이다.
    """
    return bool(rec.get("grip_transition", 0) > 0
                or rec.get("precise_hold", 0) > 0.5)


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.replace(".jsonl", "_ratio.jsonl")
    rows = [json.loads(l) for l in open(src) if l.strip()]
    rows = [r for r in rows if all(k in r for k in SLOTS)]
    print(f"라벨 {len(rows)}행  <- {src}")

    ceil = {}
    if os.path.exists(CEIL_PATH):
        ceil = json.load(open(CEIL_PATH))
        print(f"상한표 {len(ceil)}태스크  <- {os.path.normpath(CEIL_PATH)}")
    else:
        print(f"[!] 상한표가 없다 ({os.path.normpath(CEIL_PATH)}). "
              f"모든 태스크에 기본 띠 {DEFAULT_BAND} 를 쓴다 -- "
              f"클립만 푼 1.5배 결과가 들어오면 derive_libero_ceilings.py 로 만든다.")

    from libero_v2_verify import _ep_to_task
    ep2t = _ep_to_task()

    # 신뢰도는 **태스크 안에서** 분위수로 앉힌다. 태스크를 섞으면 어려운
    # 태스크의 쉬운 순간이 쉬운 태스크의 어려운 순간보다 높은 배속을 받는다.
    by_task = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[ep2t.get(r["ep"])].append(i)

    ratio = [None] * len(rows)
    nfix = 0
    for tk, idx in by_task.items():
        lo, hi = ceil.get(tk, DEFAULT_BAND) if tk else DEFAULT_BAND
        free = [i for i in idx if not contact(rows[i])]
        for i in idx:
            if i not in free:
                ratio[i] = 1.0
                nfix += 1
        if free:
            vals = band_place([confidence(rows[i]) for i in free], lo, hi)
            for i, v in zip(free, vals):
                ratio[i] = snap(v)

    with open(out, "w") as f:
        for r, v in zip(rows, ratio):
            f.write(json.dumps({**r, "ratio": v,
                                "task": ep2t.get(r["ep"]),
                                "conf": round(confidence(r), 4)}) + "\n")

    import collections
    c = collections.Counter(ratio)
    n = len(rows) or 1
    print(f"\n접촉으로 1.0배 고정  {nfix}/{n} = {nfix / n:.1%}")
    print("배속 분포  " + "  ".join(f"{g}x:{c.get(g, 0) / n:5.1%}" for g in GRID))
    print(f"평균 배속  {np.mean([v for v in ratio if v]):.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
