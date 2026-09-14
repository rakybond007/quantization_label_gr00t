"""배속 문항 v1b -- VLM 에게 배율을 직접 분류시킨다 (GPT 제안 판).

v1a 와 같은 관측·같은 지도 자료를 쓰고 묻는 방식만 다르다. 문항으로 재서 뒤에서
사상하는 대신, 사다리 네 칸을 구체적으로 서술해 주고 **어느 칸인지 고르게** 한다.
VLM 의 강점이 등급 판단이 아니라 분류라면 이쪽이 나아야 한다.

한 문항뿐이지만 형식은 같다(등급 1~4 를 배율 칸으로 읽는다).
"""
NGRADE = 4
SIGN = {"A": +1}
WEIGHT = {"A": 1.0}
NAME = {"A": "RATIO"}
# 등급 -> 배율
LADDER = {1: 1.0, 2: 2.0, 3: 2.67, 4: 4.0}

GUIDANCE = (
    "You are looking at the opening frame of a robot episode in a kitchen, together "
    "with the instruction the robot was given. The robot will execute a sequence of "
    "commanded poses. We want to know how many of those poses could be dropped -- "
    "executing only every 2nd, 3rd or 4th one, letting the arm travel further between "
    "them -- before this job would start failing.\n\n"
    "What decides it is how much positional accuracy the job demands at the moment the "
    "hand meets something. Shoving a drawer shut or pressing a button needs almost "
    "none. Closing a grip around a handle or a small object needs a lot. Setting an "
    "object down on a spot it has to sit squarely on is the same problem again.\n\n"
    "The object matters too: round and solid and standing clear is forgiving; thin, "
    "flat, tippy or wedged is not.\n\n"
    "Judge this episode, not the category of task."
)

ASK = (
    "Answer with one line, \"A) 3\", nothing else -- one digit from 1 to 4 choosing how "
    "coarsely this job could be run:\n"
    "  1 = every commanded pose is needed. Dropping even every 2nd one would put the\n"
    "      hand somewhere it must not be -- a grip that has to close on one exact\n"
    "      place, or something set down on a spot it must sit squarely on.\n"
    "  2 = every 2nd pose could be dropped. There is real accuracy demanded somewhere,\n"
    "      but a step of double the length still arrives close enough.\n"
    "  3 = every 3rd pose could be dropped. The job is mostly coarse travel with only\n"
    "      a brief moment that needs care.\n"
    "  4 = every 4th pose could be dropped. Nothing here needs the hand to arrive at a\n"
    "      particular place -- a shove, a press, or a wide forgiving target throughout.\n"
    "A) How coarsely could this job be run?\n"
    "Answer:"
)
