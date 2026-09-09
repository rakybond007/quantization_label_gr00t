"""온라인 게이트가 쓰는 두 조각: **등급표로 압축 여부**, **현오차로 배속**.

평가 도중 VLM 서버를 같이 띄워 놓고 청크마다 부르는 경로에 쓴다
(`robocasa_service_compress.py --judge-url ...`).

## 왜 따로 두나

라벨을 만들 때 쓴 식과 **한 글자도 다르면 안 된다.** 평가에서 다른 식을 쓰면
"라벨이 좋은가" 를 묻는 것이 아니라 "이 평가용 식이 좋은가" 를 묻게 된다.
그래서 부호·가중치·신뢰도 식을 `ratio_label` 에서 그대로 가져다 쓴다.

## 둘로 나눈 이유

    1단계  등급표 답변 -> 신뢰도 -> 압축할까 말까      VLM 이 정한다
    2단계  압축한다면 1.5 / 2.0 / 2.5 중 무엇          현오차가 정한다

시점마다 손상이 갈리는 것은 2배속 이상에서만 보였고(1.5배속에서는 정확히 0),
1.5배속 자체는 균일로 돌려도 공짜다(0.6567 -> 0.6467, 0.7SE). 그래서 두
결정의 성격이 다르다 -- 앞엣것은 화면을 봐야 알고, 뒤엣것은 액션으로 계산된다.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from chord_error import chunk_error_at          # noqa: E402
from ratio_label import confidence, expected_grades  # noqa: E402
from robocasa_descriptors import descriptors, facts_text  # noqa: E402

SLOTS = "ABCDE"
POS = slice(5, 8)          # 로보카사 액션 12차원 중 EE 델타 xyz


def load_checks(name="phase9_checks"):
    """문항·등급표·부호·가중치를 한 모듈에서 가져온다. 사본을 만들지 않는다."""
    m = __import__(name, fromlist=["SIGN", "WEIGHT", "NGRADE", "ASK", "GUIDANCE"])
    return m


def build_instruction(instruction, arr):
    """지시문 + **계산 사실**. 라벨을 만들 때 보낸 것과 같은 모양이어야 한다.

        phase9_two_sided.py:132   ins = f"{instr}\n{facts_text(x)}"
        judge_ab.py               f"{instr}\n{facts_text(x)}"

    빼먹으면 안 된다. 문항 첫 줄이 "The measurements above are stated as fact --
    do not re-estimate or repeat them" 이고 GUIDANCE 도 "read together with
    those measurements" 라고 말하는데, 그 measurements 가 없으면 모델은 다른
    질문에 답하는 셈이 된다.

    실제로 이것을 빼고 온라인 스모크를 돌렸더니 A 가 오프라인 3등급 94% 에서
    온라인 1등급 92% 로 뒤집혔다. C·D 는 100% 그대로였다 -- 배선은 맞고 이
    한 줄이 없어서였다.
    """
    return f"{instruction}\n{facts_text(descriptors(arr, 0))}"


def ask_gate(gate, views, instruction, mod, arr=None):
    """등급표로 묻고 신뢰도를 낸다. **강제 슬롯 로짓을 읽지 않는다.**

    `mode="text"` 로 모델이 답을 쓰게 하고 서버가 그것을 파싱한다. `n_grade` 만
    주고 `mode` 를 빼면 강제된 YES/NO 슬롯의 로짓을 읽는 옛 경로로 떨어진다 --
    CLAUDE.md 1번이 금지하는 것이고, libero 라벨링에서 한 번 그렇게 나갔다.

    `arr` 은 계획된 액션 청크 (T, 12). **반드시 준다** -- 없이 물으면 라벨을
    만들 때와 다른 것을 묻게 된다. 없으면 여기서 죽는다. 조용히 지시문만
    보내면 그 실행이 무엇을 잰 것인지 나중에 알 수 없다.

    돌려주는 것: (신뢰도, 등급 다섯, 원문). 실패하면 (None, None, 사유).
    """
    if arr is None:
        raise ValueError(
            "계획된 액션 청크가 필요하다. 라벨은 지시문 + 계산 사실로 물었고, "
            "계산 사실 없이 물으면 다른 질문이 된다")
    res = gate.judge(views, build_instruction(instruction, arr), mod.GUIDANCE,
                     question=mod.ASK, n_ask=len(SLOTS), n_grade=mod.NGRADE,
                     mode="text")
    picks = res.get("picks")
    if not picks or len(picks) != len(SLOTS) or any(p is None for p in picks):
        return None, None, res.get("error") or res.get("text", "")[:60]
    rec = {k: int(v) for k, v in zip(SLOTS, picks)}
    gp = res.get("grade_probs")
    if gp:
        rec["gp"] = gp
    return (confidence(rec, mod.SIGN, mod.WEIGHT, mod.NGRADE),
            picks, res.get("text", ""))


def flat_actions(sub, keys_sorted):
    """청크 dict -> (T, D) 배열. **판정기에 보내는 것과 같은 순서**로 편다.

    `robocasa_service_compress` 가 judge 프롬프트를 만들 때 쓰는 것과 같은
    `sorted(sub.keys())` 순서다. 두 곳이 다른 순서를 쓰면 5:8 이 EE 델타가
    아니게 되고, 그러면 조용히 엉뚱한 축의 오차를 잰다.
    """
    return np.concatenate(
        [np.asarray(sub[k], dtype=float).reshape(len(sub[k]), -1)
         for k in keys_sorted], axis=1)


def pick_rate(arr, theta, grid=(1.5, 2.0, 2.5), carry=0.0):
    """현오차가 theta 를 넘지 않는 **가장 큰** 배속. 없으면 grid 의 최솟값.

    바닥이 1.0 이 아니라 grid 최솟값인 이유: 균일 1.5 배속이 실측으로 공짜라
    (0.7SE) 그 아래로 내리는 것은 측정된 여유를 버리는 것이다.

    위상(carry)은 실제 실행 상태를 그대로 넘겨받는다 -- 평가에서는 위상이
    정해져 있으므로 라벨 만들 때처럼 훑지 않는다.
    """
    if arr.shape[1] < 8:
        raise ValueError(f"액션이 {arr.shape[1]}차원이라 5:8 을 EE 델타로 "
                         f"볼 수 없다. flatten 순서를 확인할 것")
    pad = np.zeros((16, arr.shape[1]), dtype=float)
    n = min(16, len(arr))
    pad[:n] = arr[:n]
    best = min(grid)
    for r in sorted(grid):
        e = chunk_error_at(pad, 0, r, carry)[0]
        if e <= theta:
            best = r
    return best


def chord_at(arr, r, carry=0.0):
    """진단용: 그 배속에서 실제로 얼마나 질러가나."""
    pad = np.zeros((16, arr.shape[1]), dtype=float)
    n = min(16, len(arr))
    pad[:n] = arr[:n]
    return chunk_error_at(pad, 0, r, carry)[0]
