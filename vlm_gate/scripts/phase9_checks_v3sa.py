"""v3 문항 그대로, **GUIDANCE 만 균형을 잡은 판.** 자체집계의 작동점을 고치려는 것.

v3 GUIDANCE 로 자체집계를 시키면 등급이 무너진다 -- 94% 가 Z=2·3 에 몰리고 "압축해도
된다"(Z>=4)가 5% 뿐이다. 순서는 맞는데(K3 +0.458) 작동점이 쓸 수 없는 자리에 있고,
5단계가 곧 답이라 되돌릴 역치가 없다.

원인은 GUIDANCE 가 위험만 말한다는 것이다. v3 는 "압축이 지우는 것은 도중 보정
기회" 로 시작해 무엇이 위험한지 세 문장을 쓰고, 무엇이 안전한지는 마지막 한 구절로
끝낸다. 산수 집계에서는 그 기울기가 가중합에 흡수되고 역치가 되돌리지만, 모델이
최종 판정을 직접 내리면 그대로 보수적으로 나온다.

그래서 **내용은 그대로 두고 균형만 맞춘다.** 바뀐 것 셋.

  1. 대부분의 순간은 견딘다는 사실을 먼저 말한다. 이것은 추정이 아니라 실측이다 --
     K=2 로 전 구간 압축했을 때 24 태스크 평균 성공 유지율이 0.916 이고, 균일 1.5배는
     실측으로 공짜였다. 모델에게 기저율을 알려 주지 않으면 위험 서술만 보고 기운다.
  2. 위험 쪽과 안전 쪽을 같은 분량·같은 어조로 쓴다.
  3. 다섯 칸을 다 쓰라고 명시한다. 가운데로 몰리는 것이 지금 문제다.

태스크별 순서는 넣지 않았다 -- 그것을 프롬프트에 심으면 시험집합에 맞추는 것이 된다.
문항·부호·가중은 phase9_checks_v3 를 그대로 가져온다.
"""
from phase9_checks_v3 import (  # noqa: F401
    NGRADE, SIGN, WEIGHT, SCALE, _AXES, ASK,
)

GUIDANCE = (
    "You are judging one second of a kitchen robot's motion, to decide how much of it "
    "could be thinned out -- how many of its commanded poses could be dropped, letting "
    "the arm travel further between the ones that remain, without changing the outcome.\n\n"
    "Most moments tolerate this well. Across full runs, dropping every second pose left "
    "the task succeeding about as often as before. Treat thinning as workable by default "
    "and look for the specific reason it would not be.\n\n"
    "It is workable where the arm only pushes or turns something that is already held in "
    "place by its own hinge, rail or mount, and where an object is picked up or set down "
    "in the open with room around it. The mechanism or the open space absorbs a small "
    "error.\n\n"
    "It is not workable where the gripper has to close on one exact spot and that spot is "
    "hemmed in -- reached into a cabinet or an appliance, or lined up with a slot or a "
    "handle. There the arm needs its chance to correct itself on the way.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "both kinds from one second to the next. Use the whole range of the answer scale -- "
    "clearly safe and clearly unsafe moments both occur, and answering in the middle "
    "throughout says nothing."
)
