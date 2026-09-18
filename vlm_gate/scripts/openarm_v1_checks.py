"""openarm 문항 v1 = **dexjoco v3 문항 그대로**, VIEW 만 이 로봇에 맞췄다.

바탕: `quantization_label_gr00t` 의 `dexjoco-v3-prompt` 브랜치,
`vlm_gate/scripts/dexjoco_v3_checks.py` (커밋 82d060b). 문항 A~D·부호·가중을
한 글자도 안 바꿨다 -- 먼저 **그대로 썼을 때 얼마나 나오는지** 재기 위해서다.

## VIEW 를 바꿔야 했던 이유

v3 VIEW 는 "image 2 is a wrist (eye-in-hand) close-up" 이라고 못 박는데 **openarm 에는
손목 카메라가 없다.** `ego_left`·`ego_right` 둘 다 각도만 다른 장면 뷰다(직접 열어
확인했다). 그대로 보내면 판정기에게 없는 화면을 보라고 하는 것이다.

## 이 데이터셋이 dexjoco 와 다른 점 (예측은 analysis/openarm/PREDICTION.md)

* **양팔**이다. dexjoco 는 단일 팔이었다. 두 손이 각자 다른 도구를 든다.
* 액션이 **절대 관절** 26차원(팔 7x2 + 손 6x2)이라 병합은 블록-라스트다.
* 태스크 둘 다 **가동부가 있는 물체를 조작하지 않는다**(쓰레받이·솔·수세미 전부 강체).
  A 가 죽을 것으로 본다.
* **타격이 없다.** 쓸기는 지속적인 밀기다. B 도 거의 죽을 것으로 본다.
* 그릇을 받치는 것은 표면이 아니라 **다른 손**이다. C 의 "supported by the surrounding
  surface" 와 어긋난다.
"""

NGRADE = 5

SIGN = {"A": -1, "B": -1, "C": +1, "D": +1}
WEIGHT = {"A": 2 / 3, "B": 1 / 3, "C": 1 / 3, "D": 2 / 3}
NAME = {"A": "OPERATE_PARTS", "B": "TARGETED_STRIKE",
        "C": "SUPPORTED_PRESS", "D": "CARRY_CLEAR"}

GUIDANCE = (
    "You are judging one moment of a robot's motion with a dexterous hand, to decide "
    "whether the arm could get through the stretch of motion ahead of it faster -- "
    "reaching the same places in fewer and longer steps -- without changing the "
    "outcome.\n\nGoing faster leaves the arm fewer chances to correct itself along the "
    "way. Changing the working configuration of a held object and directing a held tool "
    "onto a localized target depend on the relation between the fingers, object and "
    "target. Maintaining a supported press and carrying an already grasped object "
    "through clear space impose different constraints. Use the visible interaction to "
    "judge these checks; do not infer contact or a secure grasp from commanded motion "
    "alone.\n\nJudge the moment in front of you, not the task as a whole. One task "
    "passes through different interactions from one moment to the next."
)

# **여기만 바꿨다.** 손목 카메라가 없고 두 장 다 장면 뷰다.
VIEW = (
    "You are shown 2 camera views of this one moment, both scene views of the same "
    "workspace from slightly different angles (image 1 is observation.images.ego_left, "
    "image 2 is observation.images.ego_right). There is no wrist camera; neither view "
    "is an eye-in-hand close-up. This robot has two arms, each with a dexterous hand, "
    "and they may be doing different things at the same time -- read both hands."
)

SCALE = (
    "  5 = clearly true of this moment\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
)

_AXES = (
    "A) Are the fingers operating movable parts of the object being held—closing\n"
    "   jaws or folding an articulated part—so that changing the relation between\n"
    "   those parts is the operation, rather than merely holding the object?\n"
    "B) Is a held tool being directed into an impact on a localized target, rather\n"
    "   than making a sustained supported press? Use visible held-tool/target geometry"
    " together with the command motion to judge\n"
    "   preparation for or execution of a targeted strike. Command targets do not"
    " establish\n"
    "   that an impact actually occurred; the presence of a hammer alone is not enough.\n"
    "C) Is a finger or hand pressing a target supported by the surrounding surface,\n"
    "   without lifting it, striking it with a tool, or manipulating movable parts\n"
    "   within a grasp? A surface supports the target; do not assume it is bolted down.\n"
    "D) Is an already grasped object being moved or tilted as a whole through clear\n"
    "   space, without changing the working configuration of its movable parts,\n"
    "   acquiring/releasing the grasp, or entering a visibly tight target region?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "\nAnswer:")
