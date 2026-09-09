"""온라인 게이트가 보내는 프롬프트가 **라벨을 만들 때와 같은가.** 모델 없이 잰다.

    python vlm_gate/scripts/_selfcheck_gate_prompt.py

계산 사실 한 줄이 빠져서 A 가 오프라인 3등급 94% 에서 온라인 1등급 92% 로
뒤집혔다. 배선은 맞았고(C·D 는 100% 그대로) 프롬프트가 달랐다. 사람이 두 코드
경로를 눈으로 견주는 대신 문자열을 맞춰 본다.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from chord_rate import build_instruction                     # noqa: E402
from robocasa_descriptors import descriptors, facts_text     # noqa: E402
import phase9_checks as P                                    # noqa: E402


def main():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.2, (16, 12))
    a[:, 11] = 0.0
    instr = "close the drawer"

    # 라벨 경로가 만드는 것 (phase9_two_sided.py:132 · judge_ab.py 와 같은 식)
    want = f"{instr}\n{facts_text(descriptors(a, 0))}"
    got = build_instruction(instr, a)

    ok = want == got
    print("== 지시문 ==")
    print(f"  {'ok' if ok else '틀림'}  라벨 경로와 {'같다' if ok else '다르다'}")
    if not ok:
        print(f"    라벨: {want[:160]!r}")
        print(f"    온라인: {got[:160]!r}")
    print(f"\n  실제로 가는 것:\n    " + got.replace("\n", "\n    "))

    print("\n== 문항·등급표·부호 ==")
    print(f"  문항 {len(P.SIGN)}개 · 등급 {P.NGRADE}")
    print(f"  부호 {P.SIGN}")
    print(f"  가중 감점 합 {sum(v for k, v in P.WEIGHT.items() if P.SIGN[k] < 0):.3f}"
          f" · 가점 합 {sum(v for k, v in P.WEIGHT.items() if P.SIGN[k] > 0):.3f}")
    print(f"  ASK 첫 줄: {P.ASK.splitlines()[0][:90]}")

    # 문항이 measurements 를 참조하는데 지시문에 그것이 없으면 안 된다
    refs = "measurements above" in P.ASK
    has = "MEASURED FROM THE PLANNED MOTION" in got
    print(f"\n  문항이 measurements 를 참조하는가: {refs}")
    print(f"  지시문에 그것이 들어 있는가:       {has}")
    if refs and not has:
        print("  틀림 -- 참조는 하는데 없다. 이것이 A 를 뒤집은 자리다.")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
