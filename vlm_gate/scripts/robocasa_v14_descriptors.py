"""robocasa v14 계산 사실 = v8 + **압축 요구 두 문장.**

allex 와 humandata 에는 있고 robocasa 에만 없던 구성요소다. 포맷을 맞춘다 --
내용(단위·값·분위)은 벤치마크마다 다르지만 **채워 넣는 자리는 같아야 한다.**

robocasa 는 액션이 델타 EEF 라 병합이 **K 개 델타의 합**이다(절대 타깃인 allex·
dexjoco 의 블록-라스트와 다르다). 분위대 경계는 이 데이터셋 자기 분포에서 낸다
(창 5,131개, `configs/robocasa_merge_quantiles.json`).

한계 주장은 하지 않는다 -- allex 규칙대로 "이 녹화본 대비 큰가 작은가" 만 말한다.
"""
import json
import os

import numpy as np

from robocasa_v8_descriptors import *  # noqa: F401,F403
from robocasa_v8_descriptors import facts_text as _v8_facts

_Q = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "configs", "robocasa_merge_quantiles.json")))


def _band(v, q):
    return ("much smaller than most moments in this recording" if v < q[0] else
            "smaller than most" if v < q[1] else
            "about average" if v < q[2] else
            "larger than most" if v < q[3] else
            "much larger than most moments in this recording")


def merge_demand(a, f, n=16):
    """K 배로 합칠 때 컨트롤러가 한 틱에 요구받는 최대 변위. 델타 공간이라 **합**이다."""
    w = np.asarray(a, dtype=float)[f:f + n, 5:11]
    out = {}
    for K in (2, 3):
        m = len(w) // K * K
        out[K] = (float(np.abs(np.add.reduceat(w[:m], np.arange(0, m, K))).max())
                  if m >= K else 0.0)
    return out


def facts_text(x, a=None, f=0, n=16):
    """v8 문장 + 압축 요구 둘. `a`/`f` 를 안 주면 v8 그대로다(하위호환)."""
    s = _v8_facts(x)
    if a is None:
        return s
    d = merge_demand(a, f, n)
    s = s.rstrip(".")
    s += (f"; at 2x this stretch would ask for {d[2]:.2f} in one step, "
          f"{_band(d[2], _Q['k2'])}")
    s += f"; at 3x {d[3]:.2f}, {_band(d[3], _Q['k3'])}."
    return s
