"""openarm 문항 v2 = v1(=dexjoco v3 그대로)에서 **죽은 둘을 갈아끼우고 누르기 부호를 뒤집었다.**

## v1 이 왜 안 됐나 (analysis/openarm/oa_v1.jsonl, 300청크)

    A OPERATE_PARTS   감점  4등급이상 0.0%   1등급 94%  4등급 0건
    B TARGETED_STRIKE 감점  4등급이상 0.0%   1등급 93%  4등급 0건
    C SUPPORTED_PRESS 가점  11.7%
    D CARRY_CLEAR     가점  56.3%

**A·B 는 진짜 죽었다.** libero 에서 배운 대로 "해당 태스크에만 몰린 낮은 비율" 인지
확인했는데 4등급이 한 건도 없고 1등급이 균일하게 93~94% 다. 이 로봇의 두 태스크에는
해당 사건이 아예 없다 -- 쓰레받이·솔·수세미가 전부 강체라 가동부 조작이 없고, 쓸기는
타격이 아니라 지속적인 밀기다.

그래서 **위험 쪽 가중 1.0 이 통째로 놀고** conf 가 D·C 만으로 정해져 위로 쏠린다:
dustpan 0.667 · scrub 0.690, 국면별 0.52~0.69. 압축 게이트로 못 쓴다.

**그리고 부호가 거꾸로 걸린 자리가 있다.** C(누르기)는 가점인데, 실제로 켜지는 곳이
**세제 펌프를 누르는 순간**이다(ep73 11.7초 C=4 -- 사용자가 짚어준 그 시각이다).
dexjoco 의 "누르기는 관대하다(표면이 흡수한다)" 는 패널에 붙은 버튼 이야기이고,
여기서는 **자유롭게 서 있는 펌프 통**이라 과하게 내려가면 넘어지거나 헤드에서 미끄러진다.

## v2 의 문항

    A TOOL_GRASP    감점  도구를 쥐는 순간 (왼손이 수세미·솔·쓰레받이를 감싼다)
    B PRESS_FREE    감점  **바닥이 잡아 주지 못하는 것을 눌러 내린다** (펌프 통)
    C FINE_TARGET   감점  쥔 것을 좁은 표적으로 몰아넣는다 (큐브를 쓰레받이 턱 너머로)
    D CARRY_CLEAR   가점  **v3 원문 그대로** -- 비교 기준을 하나는 남긴다
    E OPEN_SPACE    가점  아무것도 안 쥐고 빈 공간을 지난다

D 를 글자까지 그대로 둔 것은 v1 과 v2 를 같은 청크에서 비교할 때 **적어도 한 축이
움직이지 않아야** 나머지 변화를 읽을 수 있기 때문이다.

## 예측 (돌리기 전)

* A 는 살아난다. 두 태스크 모두 도구를 쥔다.
* B 는 수세미 태스크에만, 그것도 9~13초 구간에만 켜진다. **전체 비율은 낮게 나온다** --
  그것이 정상이고 죽은 문항이 아니다(libero C 가 2.9% 로 정확히 겨눠져 있던 것과 같다).
* C 는 dustpan 의 후반(턱 넘기기)에 몰린다.
* conf 가 아래로 내려오고 국면 폭이 v1 의 0.52~0.69 보다 넓어진다.

## 가중

**잠정이다.** 실측에 맞춘 적이 없다. 양쪽 합만 1 로 맞췄다.
"""

NGRADE = 5

SIGN = {"A": -1, "B": -1, "C": -1, "D": +1, "E": +1}
WEIGHT = {"A": 0.40, "B": 0.30, "C": 0.30,
          "D": 0.60, "E": 0.40}
NAME = {"A": "TOOL_GRASP", "B": "PRESS_FREE", "C": "FINE_TARGET",
        "D": "CARRY_CLEAR", "E": "OPEN_SPACE"}

GUIDANCE = (
    "You are judging one moment of a two-armed robot's motion, each arm ending in a "
    "dexterous hand, to decide whether it could get through the stretch of motion ahead "
    "faster -- reaching the same places in fewer and longer steps -- without changing "
    "the outcome.\n\nGoing faster leaves the arms fewer chances to correct themselves "
    "along the way. That costs most where a hand has to arrive somewhere exactly or "
    "close on something exactly: taking hold of a tool, pushing a hand-held thing down "
    "onto something that is only standing on the table and can tip or slide, and guiding "
    "what is held into a small target region. It costs least where a hand is crossing "
    "open space, carrying something that is already firmly in the grasp with room all "
    "round it.\n\nThe two hands may be doing different things at the same moment -- one "
    "may hold something steady while the other works. Read both. Use the visible "
    "interaction; do not infer contact or a secure grasp from commanded motion alone.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "different interactions from one moment to the next."
)

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
    "A) Is a hand closing on a tool or object to take hold of it right now, or about\n"
    "   to -- the fingers already around it and only the squeeze left, so that a slip\n"
    "   here loses it -- rather than already holding it securely?\n"
    "B) Is a hand pushing down on something that is only standing on the table and is\n"
    "   not held flat by it -- a bottle, a pump head, an upright container -- so that\n"
    "   going too far tips it over or slides off it, rather than pressing on something\n"
    "   the table holds down?\n"
    "C) Is what a hand holds being guided into or over a small target region -- across\n"
    "   a rim or lip, into a narrow opening, onto a spot only a little larger than the\n"
    "   thing itself -- so that missing that line loses the attempt?\n"
    "D) Is an already grasped object being moved or tilted as a whole through clear\n"
    "   space, without changing the working configuration of its movable parts,\n"
    "   acquiring/releasing the grasp, or entering a visibly tight target region?\n"
    "E) Is a hand moving through open space touching nothing at all -- on its way\n"
    "   somewhere, or drawing back after letting go?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "\nAnswer:")
