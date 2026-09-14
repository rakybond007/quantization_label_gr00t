"""allex v4b -- v4 에서 **등급 척도만** 바꾼다. 문항 문구·부호·가중은 그대로.

v4 를 15에피 2,696청크에 돌린 결과 **네 문항 전부 4등급 이상이 0.0%** 였다.
표본을 2에피에서 15에피로 7배 늘려도 그대로였으니 우연이 아니다.

    A PINCH_RIGID    1:164  2:315  3:1209        최빈 3등급 72%
    B SWINGING_LOAD  1:934  2:727  3:27          사실상 1~2
    C FREE_END       1:581  2:727  3:379  4:1
    D EMPTY          1:1565 2:122  3:1           93% 가 1등급

원인은 척도의 뜻이다. v7 에서 가져온 5단은

    5 = 지금 일어나는 중 · 4 = 한 동작 앞 · 3 = 향해 가는 중

으로 **시간적 근접성**을 잰다. robocasa v7 의 문항("그리퍼가 한 지점에 닫혀야
하나")은 다가가는 과정이 있어 이 척도가 맞았다. v4 의 문항은 "지금 무엇이 이것을
붙들고 있나" 라는 **상태**를 묻는데, 상태에는 "한 동작 앞" 이 없다. 모델이 갈 곳이
3등급뿐이었다.

그래서 척도를 상태형으로 바꾼다. **문항 문구는 한 글자도 안 건드린다** -- 그래야
등급이 살아난 것이 척도 덕인지 문구 덕인지 갈린다(하네스 R2).

D(EMPTY) 는 아직 빼지 않는다. 93% 가 1등급인 것이 척도 탓일 수 있다 -- "손에
아무것도 없다" 는 상태인데 옛 척도의 5등급("지금 일어나는 중")은 말이 맞지 않는다.
이 판에서도 죽어 있으면 그때 뺀다.

비교 대상: `output/allex_v4_probe/records.jsonl` (같은 에피 0~14, v4 척도).
"""
from allex_v4_checks import (ACTIVE, GUIDANCE, NAME, NGRADE,  # noqa: F401
                             SIGN, WEIGHT, confidence, expected_grades,
                             facts_v3, ratio_for, snap)

SCALE = (
    "  5 = clearly true of this moment\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
)

_AXES = (
    "A) Is something STIFF being held only by one hand pressing against the other --\n"
    "   nothing closed around it, no fingers wrapped through -- so that the hold is\n"
    "   the balance between the hands and the thing cannot squash to stay caught?\n"
    "B) Is what the arms are moving something that DOES NOT HOLD ITS OWN SHAPE --\n"
    "   hanging, sagging, drooping over the hand, or swinging as the arms travel --\n"
    "   rather than a firm body that arrives the same way it set off?\n"
    "C) Is the thing only being SENT ACROSS -- slid or passed over to the other side,\n"
    "   or shoved on its way -- with no particular spot it has to come to rest on?\n"
    "D) Are the hands ON NOTHING AT ALL -- moving through open space or standing\n"
    "   still, with nothing in them and nothing being touched?\n"
)

ASK = (
    "The measurements above are stated as fact -- do not re-estimate or repeat them. "
    "Answer each check from what the cameras show about the MOMENT in front of you, "
    "read together with those measurements.\n"
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one digit "
    "from 1 to 5 per check, rating how far that check describes this moment:\n"
    + SCALE +
    "A grade refers only to the check on that line.\n"
    + _AXES + "Answer:"
)
