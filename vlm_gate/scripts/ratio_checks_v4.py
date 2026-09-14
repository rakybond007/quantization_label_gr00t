"""배속 문항 v4 -- 창의 시작과 끝 두 프레임을 준다.

앞 판들이 실패한 이유는 볼 수 없는 것을 물었기 때문이다. 정지 화면 한 장에는
그리퍼 개폐가 안 보인다(하네스 4절). robocasa 프롬프트가 그리퍼 방향을 계산
사실로 넣는 이유가 그것이다. 여기서도 같은 사실을 주고, 그 위에서 **화면으로만
알 수 있는 것**을 배율 판단에 얹게 한다.
"""
NGRADE = 4
SIGN = {"A": +1}
WEIGHT = {"A": 1.0}
NAME = {"A": "RATIO"}
LADDER = {1: 1.0, 2: 2.0, 3: 2.67, 4: 4.0}

GUIDANCE = (
    "You are given TWO pictures of a robot arm working in a kitchen: the LEFT is the "
    "start of a one-second stretch of motion, the RIGHT is where that stretch ends. "
    "Decide how "
    "coarsely the next second of its motion could be run -- how many of its commanded "
    "poses could be dropped, letting the arm travel further between the ones that "
    "remain, without changing the outcome.\n\n"
    "The measurements given below are computed from the planned motion and are facts. "
    "They state the arm speed and how far one merged step would jump. What the two "
    "pictures show you is the rest, including whether the gripper changes state: the "
    "LEFT picture is the start of this stretch and the RIGHT picture is where it ends.\n\n"
    "Compare the two pictures. Whether the gripper closes on something, opens off it, "
    "or stays as it was is visible in the change between them -- a single still cannot "
    "show it. Also read what the hand is near, how much room it "
    "has, whether the thing it is going for is small and exact or large and forgiving, "
    "and whether anything would be struck by a wider swing.\n\n"
    "Forming a grip and setting an object down are the moments that punish a long step. "
    "Travelling with nothing held, or shoving a large surface, does not."
)

ASK = (
    "Answer with one line, \"A) 3\", nothing else -- one digit from 1 to 4:\n"
    "  1 = every commanded pose is needed here. A longer step would put the hand\n"
    "      somewhere it must not be.\n"
    "  2 = every 2nd pose could be dropped. Real accuracy is demanded, but double the\n"
    "      step still arrives close enough.\n"
    "  3 = every 3rd pose could be dropped. Mostly coarse travel, little that needs care.\n"
    "  4 = every 4th pose could be dropped. Nothing here needs the hand to arrive at a\n"
    "      particular place.\n"
    "A) How coarsely could this stretch be run?\n"
    "Answer:"
)
