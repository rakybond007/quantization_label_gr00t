"""배속 문항 v5 -- VLM 에게 **액션이 모르는 것만** 묻는다.

액션에서 나오는 것(그리퍼 개폐·속도·회전·합쳤을 때의 변위)은 계산 사실로 넘긴다.
VLM 에게 배율이나 국면을 묻지 않는다 -- 둘 다 액션이 이미 안다.

VLM 이 유일하게 아는 것은 **무엇을 상대하고 있고 그것이 얼마나 관대한가**다.
그리고 그게 정확히 계산이 거꾸로 잡는 것이다: 파지·놓기는 천천히 움직여서 변위가
작고, 그래서 요구량 기준으로는 "압축해도 된다" 가 나온다(파지 2.11 > 접근 1.76).
정밀도 요구는 변위가 아니라 상대의 성질에서 온다.

    A  상대가 좁은가        감점
    B  물체가 다루기 어려운가  감점
    C  받아 주는 곳인가       가점
    D  통째로 미는 일인가     가점
"""
NGRADE = 5
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1}
WEIGHT = {"A": 0.60, "B": 0.40, "C": 0.45, "D": 0.55}
NAME = {"A": "TIGHT_TARGET", "B": "AWKWARD_OBJECT", "C": "CATCHER", "D": "BULK_PUSH"}

GUIDANCE = (
    "You are judging one moment of a robot arm working in a kitchen. The measurements "
    "below are computed from the planned motion and are facts -- they already tell you "
    "whether the gripper closes or opens in this stretch, how fast the arm goes, and "
    "how far a merged step would jump. Do not re-estimate any of that, and do not "
    "answer about it.\n\n"
    "Answer only about what the picture shows and the measurements cannot: **what the "
    "hand is dealing with, and how forgiving that thing is.**\n\n"
    "This matters because a careful moment and a coarse one can move the same distance. "
    "A hand closing on a small knob and a hand crossing empty space can travel equally "
    "far in a second; what separates them is how much the outcome changes if the hand "
    "arrives a centimetre off. That tolerance is a property of the object and the place "
    "it must reach, which is in the picture and nowhere else."
)

ASK = (
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one digit "
    "from 1 to 5 per check:\n"
    "  5 = clearly true of what is in front of the hand right now\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
    "A) Is the place the hand must reach SMALL AND EXACT -- a knob, a handle, a narrow\n"
    "   support, a spot something has to sit squarely on?\n"
    "B) Is the object AWKWARD -- thin, lying flat, tall and tippy, or pressed up against\n"
    "   other things -- rather than solid and standing clear?\n"
    "C) Is the destination something that CATCHES -- a basin, a bin, a wide surface --\n"
    "   where landing off-centre changes nothing?\n"
    "D) Is the hand working against something LARGE AND SOLID -- a drawer face, a door,\n"
    "   a whole appliance -- that moves as one piece however it is struck?\n"
    "Answer:"
)
