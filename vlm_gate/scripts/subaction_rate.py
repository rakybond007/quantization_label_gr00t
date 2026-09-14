"""청크 -> 배속. 지시문이 목록을 주고 액션이 그중 어디인지 정한다. VLM 불필요.

**왜 이 조합인가.** 액션은 접근/파지/운반/해제를 정확히 주지만 놓기의 종류
(접시냐 받아 주는 곳이냐)를 모른다 -- 액션 공간에서는 셋 다 '해제' 다. 그 구분은
지시문에 있다("place it on the plate" 대 "place it in the sink"). 반대로 지시문은
지금이 그중 어느 단계인지 모른다. 둘을 합치면 24 태스크 전부에서 서브액션이
유일하게 정해진다(한 국면 부류에 서브액션이 둘 이상인 태스크가 0개).

VLM 은 이 경로에 안 들어간다. 놓기 종류를 화면에서 가르는지 재어 봤더니 접시류
2.16 대 받아 주는 곳 2.00 으로 0.17 등급 차이였고(p=0.010), 1.25 와 2.10 을
가르기에는 폭이 없다.
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
SUB = json.load(open(f"{_BASE}/analysis/subaction/robocasa_subactions.json"))["tasks"]
LIM = json.load(open(f"{_BASE}/analysis/subaction/robocasa_limits_pinned.json"))
CLASS = {"approach": "접근", "grasp_open": "파지", "grasp_confined": "파지",
         "grasp_handle": "파지", "carry": "운반", "pull_swing": "운반",
         "place_precise": "해제", "place_catcher": "해제", "place_open": "해제",
         "contact_coarse": "접촉", "contact_knob": "접촉"}


def phase_of_chunk(chunk=None, grip=None):
    """접근/파지/운반/해제. 그리퍼 값만 있으면 된다.

    chunk 를 주면 마지막 열을 그리퍼로 읽고, grip 을 주면 그대로 쓴다.
    """
    g = np.asarray(grip) if grip is not None else np.asarray(chunk)[:, -1]
    dg = np.diff(g)
    if np.any(dg > 0.5):
        return "파지"
    if np.any(dg < -0.5):
        return "해제"
    return "운반" if g.mean() > 0.5 else "접근"


def rate_for(task, chunk=None, grip=None, default=2.0):
    """이 태스크의 서브액션 목록에서 지금 국면에 해당하는 것을 골라 배속을 낸다."""
    cands = SUB.get(task)
    if not cands:
        return default
    ph = phase_of_chunk(chunk, grip)
    # 접촉형 태스크는 파지·해제가 없다. 접근이 아니면 접촉이다.
    hit = [s for s in cands if CLASS.get(s) == ph]
    if not hit and ph in ("파지", "해제", "운반"):
        hit = [s for s in cands if CLASS.get(s) == "접촉"]
    if not hit:
        hit = [s for s in cands if CLASS.get(s) == "접근"] or cands
    return float(min(LIM.get(s, default) for s in hit))


# ---------------------------------------------------------------------------
# VLM 이 후보 중에서 고르는 경로.
#
# 문항은 **전 태스크 동일**하다(ratio_checks_sa1 의 속성 5문항). 태스크마다
# 달라지는 것은 지시문에서 자동으로 뽑은 **후보 목록**뿐이다. 태스크별로 문구를
# 다르게 주면 태스크마다 배속을 손으로 정해 주는 것과 같아져서 일반화가 없다.
#
# 좁힌 목록에서 액션이 국면을 정하면 후보가 1개로 떨어지는 칸이 많아 VLM 이
# 고를 것이 없었다(96칸 중 0칸). 넓힌 목록은 에피소드마다 갈릴 수 있는 짝을
# 남긴다 -- 같은 파지라도 물체가 트인 데 있으면 grasp_open, 싱크 안쪽 벽에
# 붙어 있으면 grasp_confined. 71칸 중 23칸(32%)이 열린다.
#
# 등급은 **표준화해서** 비교한다. 절대 역치는 앞서 실패했다(3.5 로 자르니
# 72.7% 가 기본값으로 떨어지고 상관이 -0.430 이 됐다).
_WIDE = json.load(open(f"{_BASE}/analysis/subaction/robocasa_subactions_wide.json"))["tasks"]
_NORM = {"A": (2.29, 0.41), "B": (1.86, 0.34), "C": (2.01, 0.47),
         "D": (3.01, 0.56), "E": (2.38, 0.35)}
# 후보를 특징짓는 속성. 같은 국면 안의 후보끼리 다른 항만 적는다.
_CHAR = {
    "grasp_open":     {"E": -1.0},
    "grasp_confined": {"E": +1.0},
    "place_precise":  {"B": +1.0, "C": -1.0},
    "place_catcher":  {"C": +1.0, "B": -1.0},
    "place_open":     {"B": -1.0, "C": -0.5},
}


def candidates(task, chunk=None, grip=None):
    """이 순간의 후보 목록. 액션이 국면을 정하고 지시문이 목록을 정한다."""
    cands = _WIDE.get(task) or SUB.get(task)
    if not cands:
        return []
    ph = phase_of_chunk(chunk, grip)
    hit = [s for s in cands if CLASS.get(s) == ph]
    if not hit and ph in ("파지", "해제", "운반"):
        hit = [s for s in cands if CLASS.get(s) == "접촉"]
    if not hit:
        hit = [s for s in cands if CLASS.get(s) == "접근"] or cands
    return hit


def pick_from(cands, grades, default=2.0):
    """후보 + 균일 문항 등급 -> (서브액션, 배속). 등급이 없으면 가장 보수적인 후보."""
    if not cands:
        return None, default
    if len(cands) == 1:
        return cands[0], float(LIM.get(cands[0], default))
    if not grades:
        s = min(cands, key=lambda c: LIM.get(c, default))
        return s, float(LIM.get(s, default))
    z = {q: (float(g) - _NORM[q][0]) / _NORM[q][1]
         for q, g in grades.items() if q in _NORM}
    best = max(cands, key=lambda c: sum(w * z.get(q, 0.0)
                                        for q, w in _CHAR.get(c, {}).items()))
    return best, float(LIM.get(best, default))
