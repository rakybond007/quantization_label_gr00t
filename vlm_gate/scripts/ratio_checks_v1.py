"""배속 문항 v1a -- 문항 다섯을 등급으로 받아 뒤에서 배율로 사상한다.

목표는 압축 여부가 아니라 **몇 배속까지 가능한가**다. 지도 자료는 평가 롤아웃
788에피의 실측 최대 안전 배율(1.0/2.0/2.67/4.0)이다.

문항은 실측 태스크 순위에서 읽히는 의미 패턴을 겨눈다.

    높은 배율  CloseDrawer 3.97 · TurnOffMicrowave 3.86 · CoffeePressButton 3.50
    낮은 배율  CoffeeSetupMug 1.28 · PnPMicrowaveToCounter 1.38 · OpenDrawer 1.62

미는 일과 누르는 일은 관대하고, 정확히 쥐거나 정해진 자리에 놓는 일은 빡빡하다.
에피마다 갈리는 것은 대상 물체의 모양과 놓인 자리다.
"""
NGRADE = 5
SIGN = {"A": -1, "B": -1, "C": -1, "D": +1, "E": +1}
# 감점 셋 합 1, 가점 둘 합 1
WEIGHT = {"A": 0.40, "B": 0.35, "C": 0.25, "D": 0.55, "E": 0.45}
NAME = {"A": "GRASP_FORM", "B": "TIGHT_TARGET", "C": "AWKWARD_OBJECT",
        "D": "COARSE_PUSH", "E": "OPEN_PATH"}

GUIDANCE = (
    "You are looking at the opening frame of a robot episode in a kitchen, together "
    "with the instruction the robot was given. Judge how coarsely this whole job "
    "could be executed -- how many of the robot's commanded poses could be dropped, "
    "letting the arm travel further between the ones that remain, before the job "
    "would start failing.\n\n"
    "What decides it is how much positional accuracy the job demands at the moment "
    "the hand meets something. Shoving a drawer shut or pressing a button needs "
    "almost none: the target is large, one coarse contact does it, and a longer step "
    "changes nothing. Closing a grip around a handle or a small object needs a lot -- "
    "the fingers have to arrive at one place, and a longer step arrives somewhere "
    "else. Setting an object down on a spot it has to sit squarely on is the same "
    "problem again at the far end.\n\n"
    "The object itself matters. Something round and solid sitting clear on a surface "
    "can be taken with a coarse approach. Something thin, lying flat, tall and tippy, "
    "or wedged against other things leaves one line in.\n\n"
    "Judge this episode, not the category of task."
)

ASK = (
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one "
    "digit from 1 to 5 per check, rating how far that check describes this episode:\n"
    "  5 = strongly true of this job\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
    "A) Does the hand have to CLOSE A GRIP on something -- fingers arriving around a\n"
    "   handle, knob or object -- rather than only pushing or pressing a surface?\n"
    "B) Does something have to END UP ON A PARTICULAR SPOT it must sit squarely on --\n"
    "   a small support, a rack, under a machine -- rather than anywhere in a basin\n"
    "   or a wide area?\n"
    "C) Is the object AWKWARD TO TAKE -- thin, lying flat, tall and tippy, or pressed\n"
    "   up against other things -- rather than round and solid and standing clear?\n"
    "D) Is this job ONE COARSE SHOVE OR PRESS against a large target -- a drawer face,\n"
    "   a door, a button, a lever -- that works even if the hand lands off-centre?\n"
    "E) Is the way to the target OPEN -- nothing to thread between, nothing that would\n"
    "   be struck by a wider swing?\n"
    "Answer:"
)
