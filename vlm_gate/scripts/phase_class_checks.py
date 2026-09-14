"""국면 분류 문항 -- VLM 은 분류만 하고 배율은 표에서 온다 (검토에서 나온 구조).

등급 판단이 아니라 **분류**를 시킨다. 정답지가 공짜로 있다: 국면은 액션에서
유도되므로(`phase_scorecard.ph6`) 분류 정확도를 새 평가 실행 없이 잰다.

배율은 이 분류에 붙는다. 청크마다의 운동학 요구량에 국면별 한계를 걸어 적합한
값이 이미 있다 -- 놓기 1.19 < 해제 1.21 < 파지 1.36 < 운반 1.50 < 접근 1.84,
홀드아웃 순위상관 +0.485.
"""
NGRADE = 4
SIGN = {"A": +1}
WEIGHT = {"A": 1.0}
NAME = {"A": "PHASE"}
# 등급 -> 국면. 위험한 쪽을 낮은 번호로 둔다(등급이 곧 관대함의 순서가 되게).
LADDER = {1: "파지", 2: "해제", 3: "운반", 4: "접근"}

GUIDANCE = (
    "You are looking at one moment of a robot arm working in a kitchen. The arm has a "
    "two-finger gripper. Say which stage of the manipulation this moment belongs to.\n\n"
    "The four stages differ in what the hand is doing with the object:\n"
    "  - it is on its way to something, holding nothing yet\n"
    "  - it is closing its fingers on something, taking hold\n"
    "  - it is holding something and moving it\n"
    "  - it is opening its fingers, letting go of what it held\n\n"
    "Read the fingers and what is between them. A gripper that is open with nothing "
    "between the fingers is travelling. A gripper whose fingers are around an object "
    "and shut on it is holding. The two moments where the fingers change state are the "
    "ones to separate carefully: taking hold, and letting go."
)

ASK = (
    "Answer with one line, \"A) 3\", nothing else -- one digit from 1 to 4:\n"
    "  1 = TAKING HOLD. The fingers are closing on the object right now, or have just\n"
    "      shut on it. This is the moment the grip is formed.\n"
    "  2 = LETTING GO. The fingers are opening off the object, or have just released\n"
    "      it. The object stays and the hand leaves.\n"
    "  3 = CARRYING. The object is held between the closed fingers and is being moved.\n"
    "      The grip was formed earlier and is not changing now.\n"
    "  4 = REACHING. Nothing is held. The hand is travelling toward something, or\n"
    "      withdrawing, with the fingers empty.\n"
    "A) Which stage is this moment?\n"
    "Answer:"
)
