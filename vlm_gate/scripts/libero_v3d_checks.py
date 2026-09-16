"""libero 문항 v3d = v3c 를 **robocasa v22 포맷에 맞추고 두 결함을 고친 판.**

## 포맷 (네 벤치마크 공통)

    You are judging ... (가이던스)
    카메라 뷰 N개 설명
    The robot was told: {지시문}
    MEASURED FROM THE PLANNED MOTION over the chunk ahead (...): {계산 사실}
    등급 서술 + 척도 + 문항

v3c 는 가이던스·문항이 파일로 흩어져 있었고 VIEW 는 프로브 스크립트에 하드코딩돼
있었다(그래서 robocasa 에 libero 문장이 통째로 넘어가 있었다). v3d 는 판본 하나가
GUIDANCE·VIEW·SCALE·ASK·SIGN·WEIGHT·NAME 을 다 들고 있다.

## 고친 결함 둘 -- 둘 다 이미 기록에 적혀 있던 것

**1. A 와 C 가 서로를 상쇄한다.** `analysis/libero_retention/V2_DIAGNOSIS.md` 가
"A 와 C 가 같은 국면에서 상쇄한다. 둘 다 해제에서 최고인데 부호가 반대다" 라고
적어 뒀는데 v3c 까지 안 고쳐졌다(`score_v3c.txt`: A 해제 3.56 감점 · C 해제 2.91 가점).
둘은 **같은 축**이다 -- 목적지가 좁으냐 넉넉하냐. 축 하나를 문항 둘로 쪼개 부호를
반대로 주면 서로를 지운다. human_data 에서 그 둘을 하나로 합치니 좁은 컵과 넓은
접시를 **1.19 대 4.28 (4등급이상 0% 대 97%)** 로 완벽히 갈랐다. 여기서도 합친다.

**2. 가이던스가 아무 문항도 안 묻는 성질을 한 문단 통째로 설명한다.** v3c 가이던스
4문단은 "트인 접근 vs 끼어든 접근"인데, 그걸 묻던 v2 의 B 가 v3b 에서 "파지 순간"
으로 재조준되면서(죽은 문항이라 옳은 수정이었다) 문항이 사라졌다. 그런데 그것이
**libero 에서 실측으로 값이 묶인 유일한 위험이다** -- 사다리 최고 계단이 2.4 인데

    grasp_between 1.667 · grasp_confined 1.667 · grasp_stacked 1.819 ·
    grasp_handle 2.00 · push_close 2.034 · grasp_open 2.261      <- 측정됨
    push_object 2.40 · turn_knob 2.40 · place_precise 2.579 ·
    pull_swing 2.859 · place_catcher 8.53 · carry 378.9 ·
    place_open 7.5e9                                              <- 최고 계단 이상 = 미식별

측정된 여섯이 전부 파지·밀기이고, 그 안에서 갈리는 것이 **끼어든 파지(1.667~1.819)
대 트인 파지(2.261)** 다. robocasa 는 그 구분을 B("그 닫히는 자리가 좁은가")로
가지고 있다. libero 에 되살린다.

## v3c -> v3d 대응

    v3c A 작고 고정된 자리(감점) ┐
    v3c C 받아 주는 통(가점)     ┴-> v3d A ROOMY_TARGET (가점 하나로)
    v3c B 파지 순간              -> v3d B GRASP_CLOSING  (글자 그대로)
    (없음)                       -> v3d C THREADED_PICK  (되살림)
    v3c D 손 떼는 순간           -> v3d D HAND_OFF       (관측 상태로 다시 씀)
    v3c E 붙들 필요 없는 일       -> v3d E NO_HOLD        (그대로, 아래 주의)

`D` 를 "손이 이미 벌어졌는가" 로 다시 쓴 이유: human_data 에서 "이미 놓여 있는가"
형태가 **16스텝 창 때문에 내려놓는 중에도 부분 참(2.25)이 되어** 위험 문항을
상쇄했다. 화면에서 바로 보이는 상태(손가락이 벌어졌는가)에 묶으면 창에 안 흔들린다.

## 주의: E 는 거의 죽어 있다

`score_v3c.txt` 에서 E 는 4등급이상 2%, 국면간 표준편차 0.132 로 사실상 상수다.
LIBERO 에 밀기·돌리기 태스크가 적어서일 수 있고(libero_goal 몇 개), 그렇다면
**정상적인 낮은 발동률**이다. 지우지 않고 두되 가중은 실측 적합(R5)이 정한다 --
한 번도 안 켜지는 문항은 적합이 0 근처로 보낸다.

## 가중

**잠정이다.** v3c 의 값은 v2 성질(`CROWDED_PICK`·`WIDE_SURFACE`)이 덮는 태스크 수로
유도된 것인데 그 두 문항이 v3c 에서 교체됐으므로 새 문항에는 근거가 없다. 여기서는
양쪽 합만 1 로 맞춘 잠정값을 둔다. 라벨링 뒤 실측 적합(R5).
"""

NGRADE = 5

SIGN = {"B": -1, "C": -1, "A": +1, "D": +1, "E": +1}

WEIGHT = {"B": 0.60, "C": 0.40,
          "A": 0.50, "D": 0.30, "E": 0.20}

NAME = {"A": "ROOMY_TARGET", "B": "GRASP_CLOSING", "C": "THREADED_PICK",
        "D": "HAND_OFF", "E": "NO_HOLD"}

GUIDANCE = (
    "You are judging one moment of a table-top robot's motion, to decide whether the arm "
    "could get through the stretch of motion ahead of it faster -- reaching the same "
    "places in fewer and longer steps -- without changing the outcome.\n\n"
    "Going faster leaves the arm fewer chances to correct itself along the way. That "
    "costs most when the gripper has to close on one exact thing -- and more still when "
    "the way in is threaded: the object perched on top of another, or wedged in among "
    "things standing close by, so that there is one line in and a longer step swings "
    "wider than that line. It costs least where whatever is being moved only has to land "
    "somewhere that will catch it, and at the moments either side of the work -- crossing "
    "open space, carrying something with room all round it, drawing back after letting "
    "go.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "both kinds from one moment to the next: the same episode that threads a bowl out "
    "from between two dishes then carries it through open air."
)

VIEW = (
    "You are shown 2 camera views of this one moment: a scene view and a wrist "
    "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
    "normally look close in it -- general closeness is normal. Use the wrist view to "
    "judge the close moments: the gripper actually closing on something, and something "
    "being set down into the place it must end up."
)

SCALE = (
    "  5 = clearly true of this moment\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
)

_AXES = (
    "A) Is the place this object has to end up ROOMY FOR IT -- a wide bowl, an open\n"
    "   basket, a broad tray or an open stretch of table with slack all round -- rather\n"
    "   than a plate, a rack, or a spot barely bigger than the thing itself?\n"
    "B) Is the gripper closing on the object right now, or about to -- the fingers\n"
    "   already around it and only the squeeze left, so that a slip here loses it?\n"
    "C) Is that closing spot threaded -- the thing being taken perched on top of another\n"
    "   object, or wedged in among things standing close by -- so that there is only one\n"
    "   line in for the hand?\n"
    "D) Is the hand ALREADY OPEN and coming away, the object left sitting where it was\n"
    "   meant to end up and holding its own weight without the fingers?\n"
    "E) Is this a job that needs no hold kept on anything -- a push, a knob turned, a\n"
    "   drawer pressed shut -- so that once it is sent it carries on without the hand?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")
