"""최종 라벨 -- 매 시점에 **몇 배속으로 갈지**를 하나의 값으로 적는다.

벤치마크를 인자로 받는다. **문항과 가중치는 벤치마크마다 다르고, 각자의 checks
모듈에서 읽는다** -- robocasa 는 `phase9_checks`, libero 는 `libero_v2_checks`.
여기에 박아 두면 robocasa 라벨을 libero 가중치로 계산하는 일이 조용히 일어난다.

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

    python vlm_gate/scripts/ratio_label.py <bench> <labels.jsonl> [out.jsonl]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SLOTS = "ABCDE"

# 디코더가 가진 눈금. F_level 의 level_ks 와 같아야 한다.
GRID = (1.0, 1.5, 2.0, 2.5)
DEFAULT_BAND = (1.0, 2.0)

_HERE = os.path.dirname(os.path.abspath(__file__))
# 벤치마크 -> (checks 모듈, 에피소드->태스크 함수를 가진 모듈)
CHECKS = {"libero": "libero_v2_checks", "robocasa": "phase9_checks"}


def load_bench(bench):
    """그 벤치마크의 SIGN·WEIGHT·NGRADE 와 상한표를 가져온다."""
    if bench not in CHECKS:
        raise SystemExit(f"모르는 벤치마크 {bench}. 아는 것: {sorted(CHECKS)}")
    mod = __import__(CHECKS[bench])
    sign, weight = dict(mod.SIGN), dict(mod.WEIGHT)
    ngrade = int(getattr(mod, "NGRADE", 5))
    # 각 변의 합이 1 이어야 한다. 아니면 신뢰도가 0~1 을 벗어난다.
    for side, sel in (("감점", -1), ("가점", +1)):
        t = sum(weight[k] for k in weight if sign[k] == sel)
        if abs(t - 1.0) > 1e-6:
            raise SystemExit(f"{bench} {side} 가중치 합이 {t:.4f} 다. 1 이어야 한다")
    ceil_path = os.path.join(_HERE, "..", "analysis",
                             f"{bench}_task_ceilings.json")
    ceil = json.load(open(ceil_path)) if os.path.exists(ceil_path) else {}
    return sign, weight, ngrade, ceil, ceil_path


def confidence(picks, sign, weight, ngrade=5):
    """(1 + Σw·가점 − Σw·감점) / 2. 등급은 (g−1)/(n−1) 로 0~1 에 옮긴다."""
    d = float(ngrade - 1)
    risk = sum(weight[k] * (picks[k] - 1) / d for k in SLOTS if sign[k] < 0)
    safe = sum(weight[k] * (picks[k] - 1) / d for k in SLOTS if sign[k] > 0)
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
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    bench, src = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else src.replace(".jsonl", "_ratio.jsonl")
    SIGN, WEIGHT, NGRADE, ceil, ceil_path = load_bench(bench)
    print(f"[{bench}]  감점 " + " ".join(
        f"{k}={WEIGHT[k]:.3f}" for k in SLOTS if SIGN[k] < 0)
        + "  가점 " + " ".join(f"{k}={WEIGHT[k]:.3f}" for k in SLOTS if SIGN[k] > 0))
    rows = [json.loads(l) for l in open(src) if l.strip()]
    rows = [r for r in rows if all(k in r for k in SLOTS)]
    print(f"라벨 {len(rows)}행  <- {src}")

    if ceil:
        print(f"상한표 {len(ceil)}태스크  <- {os.path.normpath(ceil_path)}")
    else:
        print(f"[!] 상한표가 없다 ({os.path.normpath(ceil_path)}). 모든 태스크에 "
              f"기본 띠 {DEFAULT_BAND} 를 쓴다 -- 사다리가 차면 "
              f"derive_libero_ceilings.py 로 만든다.")

    if bench == "libero":
        from libero_v2_verify import _ep_to_task
        ep2t = _ep_to_task()
    else:
        # robocasa 는 라벨 줄이 태스크를 들고 있다(phase9_to_parquet 참고).
        ep2t = {}

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
            vals = band_place([confidence(rows[i], SIGN, WEIGHT, NGRADE) for i in free], lo, hi)
            for i, v in zip(free, vals):
                ratio[i] = snap(v)

    with open(out, "w") as f:
        for r, v in zip(rows, ratio):
            f.write(json.dumps({**r, "ratio": v,
                                "task": ep2t.get(r["ep"]),
                                "conf": round(confidence(r, SIGN, WEIGHT, NGRADE), 4)}) + "\n")

    # 배포용 parquet 도 같이 낸다. `apply_ratio_labels.py` 가 읽는 형식이고,
    # 다른 기계에서는 이것 하나만 받으면 된다 -- 등급도 jsonl 도 필요 없다.
    try:
        import pandas as pd
        pq = out.replace(".jsonl", ".parquet")
        pd.DataFrame({
            "episode_index": [int(r["ep"]) for r in rows],
            "frame_index": [int(r["f"]) for r in rows],
            "task": [ep2t.get(r["ep"]) for r in rows],
            "ratio": np.asarray(ratio, dtype=np.float32),
            "conf": np.array([confidence(r, SIGN, WEIGHT, NGRADE) for r in rows], dtype=np.float32),
            "fixed": np.array([1 if contact(r) else 0 for r in rows], dtype=np.int8),
        }).sort_values(["episode_index", "frame_index"]).to_parquet(pq, index=False)
        print(f"-> {pq}  (배포용. docs/RATIO_LABELS.md 참고)")
    except ImportError:
        print("[!] pandas 가 없어 parquet 은 안 만들었다. jsonl 은 나왔다.")

    import collections
    c = collections.Counter(ratio)
    n = len(rows) or 1
    print(f"\n접촉으로 1.0배 고정  {nfix}/{n} = {nfix / n:.1%}")
    print("배속 분포  " + "  ".join(f"{g}x:{c.get(g, 0) / n:5.1%}" for g in GRID))
    print(f"평균 배속  {np.mean([v for v in ratio if v]):.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
