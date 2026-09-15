"""human_data (pnp_task · long_horizon_task) 의 계산 사실.

**libero_descriptors 를 그대로 못 쓴다.** libero 는 델타 EEF 액션이라
`clip_excess` 가 델타의 합인데, 여기 액션은 `joint_pos_abs` -- 7개 절대 관절
타깃 + 연속 그리퍼 1차원이다. 절대 타깃에서 K배 병합은 **합이 아니라 블록의
마지막 타깃**이고, 한 틱에 요구되는 변위는 `max‖a[i+K] - a[i]‖` 다
(`prompts/FORMAT.md` §3).

그리퍼도 다르다. libero 는 이진 차원을 쓰지만 여기는 0~1 연속이다. 실측 분포가
0.1 미만 57~62% · 0.4 초과 35~40% 로 갈라져 있어 0.4 를 문턱으로 잡는다.

문장 모양은 libero·robocasa·allex 와 같게 간다: 상태는 말로, 압축 요구만
숫자와 분위대로. 문장 수는 조건과 무관하게 항상 같다.
"""
import json
import os

import numpy as np

N_JOINT = 7

# **문턱을 손으로 적지 않는다.** 처음에 속도 3단을 0.004 / 0.012 로, 그리퍼를 0.4 로
# 적어 두었는데 전부 내가 지어낸 값이었다. 데이터에서 뽑으니 속도 3분위가 pnp
# 0.0152 / 0.0251 · long 0.0123 / 0.0200 이고 그리퍼 골은 pnp 0.70 · long 0.38 이다
# -- 내 숫자로는 거의 모든 창이 "moving fast" 로 나갔다. 데이터셋마다 다르므로
# 하나로 묶을 수도 없다. 전부 humandata_quantiles.py 가 낸다.
GRIP_FALLBACK = 0.4

# 분위대 경계는 **데이터셋마다 자기 분포에서** 낸다. 남의 숫자를 쓰면 안 된다.
# scripts/humandata_quantiles.py 가 만든다.
_QP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "human_data", "quantiles.json")
try:
    _Q = json.load(open(_QP))
except Exception:
    _Q = {}


def _th(dataset):
    q = _Q.get(dataset or "", {}) or {}
    return (q.get("speed_tertile") or [None, None],
            q.get("decel_q25"),
            q.get("grip_closed", GRIP_FALLBACK))


def descriptors(a, f=0, n=16, k=2, dataset=None):
    """창 a[f:f+n] 에서 나오는 값들. a 는 (T, 8) joint_pos_abs + gripper."""
    a = np.asarray(a, dtype=float)
    (s_lo, s_hi), dq, GRIP_CLOSED = _th(dataset)
    w = a[f:f + n]
    J, g = w[:, :N_JOINT], w[:, N_JOINT]
    step = np.linalg.norm(np.diff(J, axis=0), axis=1) if len(J) > 1 else np.zeros(1)

    def merged(K):
        return float(np.linalg.norm(J[K:] - J[:-K], axis=1).max()) if len(J) > K else 0.0

    closed = g > GRIP_CLOSED
    h = max(1, len(step) // 2)
    v0, v1 = step[:h].mean(), step[h:].mean() if len(step) > h else step[:h].mean()
    return {
        "dataset": dataset,
        "grip_closed_frac": float(closed.mean()),
        "grip_change": float(abs(g[-1] - g[0])),
        "grip_closing": bool(g[-1] > g[0] + 0.1),
        "grip_opening": bool(g[-1] < g[0] - 0.1),
        "speed_mean": float(step.mean()), "speed_max": float(step.max()),
        # 감속 판정도 분포에서: 이 데이터셋 창들의 후반/전반 비율 하위 25% 보다 작으면.
        "decel": bool(v1 < v0 * dq) if dq else False,
        # "무언가 든 채 기어감" = 닫힌 채이면서 속도가 이 데이터셋 하위 3분위 아래.
        "closed_slow": bool(closed.mean() > 0.5 and s_lo is not None
                            and step.mean() < s_lo),
        "speed_lo": s_lo, "speed_hi": s_hi,
        "merge_k2": merged(2), "merge_k3": merged(3),
    }


def _band(v, q):
    if not q:
        return "larger than most" if v > 0 else "about average"
    return ("much smaller than most moments in this recording" if v < q[0] else
            "smaller than most" if v < q[1] else
            "about average" if v < q[2] else
            "larger than most" if v < q[3] else
            "much larger than most moments in this recording")


def facts_text(x):
    """문장 수 고정. 상태는 말로, 압축 요구만 숫자 + 분위대."""
    ds = x.get("dataset") or ""
    q2 = (_Q.get(ds, {}) or {}).get("k2")
    q3 = (_Q.get(ds, {}) or {}).get("k3")
    p = []
    cf = x["grip_closed_frac"]
    p.append("the hand stays closed on something throughout" if cf > 0.9 else
             "the hand stays open throughout" if cf < 0.1 else
             "the hand closes or opens part-way through this window")
    p.append("and it is closing" if x["grip_closing"] else
             "and it is opening" if x["grip_opening"] else
             "and its opening barely changes")
    sp, lo, hi = x["speed_mean"], x.get("speed_lo"), x.get("speed_hi")
    p.append("the arm is barely moving" if lo is not None and sp < lo else
             "the arm is moving at a normal pace" if hi is not None and sp < hi else
             "the arm is moving fast")
    p.append("it is decelerating to a near stop" if x["decel"]
             else "it keeps its pace to the end of the window")
    p.append("it is creeping along with something held" if x["closed_slow"]
             else "it is not creeping along with something held")
    p.append(f"at 2x this stretch would move {x['merge_k2']:.3f} rad in one step, "
             f"{_band(x['merge_k2'], q2)}")
    p.append(f"at 3x {x['merge_k3']:.3f} rad, {_band(x['merge_k3'], q3)}")
    return ("MEASURED FROM THE PLANNED MOTION over the chunk ahead (these are computed "
            "facts, not estimates): " + "; ".join(p) + ".")
