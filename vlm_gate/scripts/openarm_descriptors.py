"""openarm 계산 사실. 양팔(각 7관절) + 양손(각 6관절), **전부 절대 관절**.

dexjoco v3 의 계산 사실 문체를 따른다 -- 숫자 대신 정성 서술이고, 마지막에
"이것은 명령의 요구이지 로봇의 한계나 측정된 접촉이 아니다" 를 붙인다.

**문턱은 전부 `analysis/openarm/quantiles.json` 에서 읽는다**(태스크군별 3분위).
손으로 적은 상수를 쓰면 그것이 곧 추측한 한계가 된다.

절대 관절이므로 병합은 **블록-라스트**다(합산이 아니다). 그래서 "병합 점프" 는
K 스텝 떨어진 두 명령 사이의 관절공간 거리로 잰다.
"""
import json
import os

import numpy as np

_QP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "analysis", "openarm", "quantiles.json")
try:
    _Q = json.load(open(_QP))
except Exception:                                    # noqa: BLE001
    _Q = {}

SL = {"left_arm": (0, 7), "right_arm": (7, 14),
      "left_hand": (14, 20), "right_hand": (20, 26)}


def _q(group):
    return _Q.get(group or "", {}) or {}


def descriptors(a, f=0, n=16, group=None):
    """창 a[f:f+n] -> dict. a 는 (T, 26) 절대 관절."""
    a = np.asarray(a, dtype=float)
    q = _q(group)
    w = a[f:f + n]
    out = {"group": group}
    for k, (lo, hi) in SL.items():
        seg = w[:, lo:hi]
        st = np.linalg.norm(np.diff(seg, axis=0), axis=1) if len(seg) > 1 else np.zeros(1)
        out[f"{k}_speed"] = float(st.mean())
        t = q.get(f"{k}_tertile")
        out[f"{k}_band"] = (None if not t else
                            "slow" if st.mean() < t[0] else
                            "moderate" if st.mean() < t[1] else "fast")
    for k in ("left_hand", "right_hand"):
        lo, hi = SL[k]
        g = w[:, lo:hi].mean(1)
        th = q.get(f"{k}_closed")
        out[f"{k}_closed_frac"] = float((g > th).mean()) if th is not None else None
        out[f"{k}_closing"] = bool(g[-1] > g[0] + 0.05)
        out[f"{k}_opening"] = bool(g[-1] < g[0] - 0.05)
        rng = q.get(f"{k}_range")
        # 그 손이 이 태스크군에서 애초에 거의 안 움직이면 개폐를 말하지 않는다.
        out[f"{k}_active"] = bool(rng and (rng[1] - rng[0]) > 0.4)
    arms = w[:, 0:14]
    st = np.linalg.norm(np.diff(arms, axis=0), axis=1) if len(arms) > 1 else np.zeros(1)
    h = max(1, len(st) // 2)
    v0 = st[:h].mean(); v1 = st[h:].mean() if len(st) > h else v0
    dq = q.get("decel_q25")
    out["decel"] = bool(v1 < v0 * dq) if dq else False
    # 방향 반전: 창 전반/후반 관절 변위 벡터의 코사인
    if len(arms) > 3:
        d0 = arms[h] - arms[0]; d1 = arms[-1] - arms[h]
        nn = np.linalg.norm(d0) * np.linalg.norm(d1)
        out["reversal"] = bool(nn > 1e-9 and float(d0 @ d1) / nn < -0.2)
    else:
        out["reversal"] = False
    for K in (2, 3):
        out[f"merge_k{K}"] = (float(np.linalg.norm(arms[K:] - arms[:-K], axis=1).max())
                              if len(arms) > K else 0.0)
    return out


def _band_words(b):
    return {"slow": "slowly", "moderate": "at a moderate rate", "fast": "quickly"}.get(b, "")


def facts_text(x):
    """v3 문체의 정성 서술. 숫자를 쓰지 않는다."""
    p = []
    for side in ("left", "right"):
        arm = x.get(f"{side}_arm_band")
        p.append(f"the {side} arm is moving {_band_words(arm)}" if arm
                 else f"the {side} arm is moving")
    hands = []
    for side in ("left", "right"):
        k = f"{side}_hand"
        if not x.get(f"{k}_active"):
            # 이 태스크군에서 그 손은 개폐가 거의 없다 -- 없는 사건을 말하지 않는다.
            hands.append(f"the {side} hand keeps its shape")
            continue
        if x.get(f"{k}_closing"):
            hands.append(f"the {side} hand is closing")
        elif x.get(f"{k}_opening"):
            hands.append(f"the {side} hand is opening")
        else:
            cf = x.get(f"{k}_closed_frac")
            hands.append(f"the {side} hand stays closed" if (cf or 0) > 0.5
                         else f"the {side} hand stays open")
    line2 = ("The planned arm path reverses direction part-way through"
             if x.get("reversal") else
             "The planned arm path has no substantial direction reversal")
    line2 += ("; its later steps slow markedly" if x.get("decel")
              else "; its later steps do not show marked slowing")
    return ("MEASURED FROM THE PLANNED MOTION over the chunk ahead (these are computed "
            "facts, not estimates):\n"
            + "; ".join(p + hands) + ".\n"
            + line2 + ".\n"
            "These comparisons describe command demands, not robot limits or "
            "measured contact.")
