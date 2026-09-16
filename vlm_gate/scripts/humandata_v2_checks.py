"""human_data 문항 v2 = v1 에서 **죽은 문항 둘을 빼고 나머지를 이 데이터셋에 맞췄다.**

## 왜 바꾸나 -- 여섯 중 둘이 아무것도 안 하고 있었다

전량 라벨 2,331 청크(`output/humandata_v1/labels.jsonl`)의 등급 분포:

    A  4등급이상  0.5%   1:82% 2:17% 3:1% 4:0% 5:0%   <- 감점 문항이 죽었다
    B  4등급이상 27.0%
    C  4등급이상 36.5%
    D  4등급이상  5.7%
    E  4등급이상  0.0%   1:85% 2:15% 3:0% 4:0% 5:0%   <- 가점 문항이 완전히 죽었다
    F  4등급이상 12.7%

libero 자신의 진단 규칙(`analysis/libero_retention/V2_DIAGNOSIS.md`: "4등급 이상이
0~5% 면 죽은 문항이고 원인은 부호·가중이 아니라 문구가 화면과 안 맞는 것")에 그대로
걸린다. 원인이 분명하다. 이 두 데이터셋의 지시문은

    pnp_task          Pick up the doll / green cube / blue cup and put it into the plate
    long_horizon_task Put the doll and the green cube in the left red cup, and put the
                      mint cube in the right red cup.

* **v1 A** 는 "작고 고정된 자리에 반듯이 **올려놓는가** -- 접시·버너·걸이 -- 받아 주는
  것에 **넣는** 게 아니라" 를 묻는다. 그런데 두 데이터셋 다 container 에 **넣는다.**
  접시도 인형·큐브에 비하면 넓어 받아 주는 쪽이고 빨간 컵은 문자 그대로 넣는 것이다.
  판정기가 "올려놓기가 아니다" 라고 옳게 답했을 뿐이다. 그리고 **진짜 차이(그릇이
  넉넉하냐 좁으냐)는 v1 C 가 이미 묻고 있고 C 는 살아 있다(36.5%).** A 는 C 와 중복이면서
  기하만 틀렸다. 버린다.
* **v1 E** 는 "밀기·노브 돌리기·서랍 닫기" 를 묻는다. 이 데이터셋에 그런 태스크가
  **하나도 없다.** libero 에서도 이미 2% 로 죽어 있었고(libero_goal 몇 개뿐) human_data
  에는 0 이다. 버린다.

**두 문항을 빼면 판별이 좋아진다.** 기존 등급 그대로 가중만 다시 정규화해 계산하면
(라벨을 다시 만들지 않고):

    6문항 현행       tau 0.470 -> long 내려놓기 61% 차단 · 운반 14% · 차 47%
    4문항 (A·E 제거) tau 0.480 -> long 내려놓기 74% 차단 · 운반 15% · 차 59%

    국면별 conf (4문항)
      long  집으러 0.343  파지 0.402  운반 0.525  내려놓으러 0.427  물러남 0.475
      pnp   집으러 0.555  파지 0.571  운반 0.694  내려놓으러 0.678  물러남 0.724

같은 문항으로 **long 의 좁은 빨간 컵 내려놓기는 눌리고(0.427 < 운반 0.525) pnp 의 넓은
접시 내려놓기는 안 눌린다(0.678 ~ 운반 0.694).** 두 데이터셋이 갈리는 것이 이 문항
집합의 검증이고, 죽은 문항 둘이 그 신호를 희석하고 있었다.

## v1 -> v2 글자 대응

    v1 B (파지 순간)      -> v2 A
    v1 C (넉넉한 자리)    -> v2 B
    v1 D (이미 놓였음)    -> v2 C
    v1 F (쥔 채 내려놓기) -> v2 D
    v1 A, v1 E            -> 버림

## 그 밖에 고친 것

* **GUIDANCE 를 이 데이터셋으로 다시 썼다.** v1 은 libero v3c 를 글자까지 그대로 썼고,
  그래서 남의 장면이 들어 있었다 -- "a plate, on a burner, or into a rack"(버너·걸이
  없음), "threads a bowl out from between two dishes"(사발·접시 없음), 그리고 통째로
  "트인 접근 vs 끼어든 접근" 을 설명하는 문단(그걸 묻는 문항이 v3c 에서 이미 교체돼
  없어졌다). 1문단의 `how many of its commanded poses could be dropped` 도 고쳤다 --
  우리는 배속도 버릴 개수도 정하지 않는다.
* **VIEW.** `Use the wrist view only to spot the actual grasp-closure or fine-insertion
  instant` 였다. 삽입 태스크가 없고, `only` 가 내려놓는 순간(D)을 배제한다. 그 순간은
  손목 뷰가 가장 잘 보여 주는 장면이다.
* **D(=v1 F).** `LOWERING what it still holds onto its target` 는 holds onto 가 붙어
  읽혀 구문이 두 갈래였고 `the spot it must end up` 에 on 이 빠져 있었다. robocasa v22
  에서 같은 자리를 고쳤다.
* **부호·가중을 이 모듈에 둔다.** v1 은 `humandata_tau.py` 가 libero 것을 가져와
  런타임에 F 를 끼워 넣었다(`W["F"] = W["B"]` 후 위험쪽 재정규화). 어디에도 적혀 있지
  않아서 `prompts/humandata_v1_sign_weight.txt` 만 없었다 -- 네 벤치마크 중 하나만
  포맷 구성요소가 빠져 있던 자리다.

## 가중

**잠정이다.** 안전쪽 B:C 는 libero 의 8:4 비율을 물려받았고, 위험쪽은 v1 이 F 에
B 와 같은 무게를 주던 것을 이어 균등으로 둔다. 실측에 맞춘 적이 없다.
human_data 에는 배속 사다리가 없으므로 적합 근거는 국면 분리와 τ 뿐이다.
"""

NGRADE = 5

SIGN = {"A": -1, "D": -1, "B": +1, "C": +1}

WEIGHT = {"A": 0.50, "D": 0.50,
          "B": 2 / 3, "C": 1 / 3}

NAME = {"A": "GRASP_CLOSING", "B": "ROOMY_TARGET",
        "C": "ALREADY_RESTING", "D": "STILL_LOWERING"}

GUIDANCE = (
    "You are judging one moment of a table-top robot's motion, to decide whether the arm "
    "could get through the stretch of motion ahead of it faster -- reaching the same "
    "places in fewer and longer steps -- without changing the outcome.\n\n"
    "Going faster leaves the arm fewer chances to correct itself along the way. That "
    "costs most at the two moments where the object can be lost: when the hand is closing "
    "on it, and when it is being brought down into the place it has to end up. Both get "
    "worse the less room there is -- letting a cube down into a cup barely wider than "
    "itself leaves no slack, while letting the same cube down onto an open plate does.\n\n"
    "Most moments are neither. Crossing empty space to reach for something, carrying "
    "something with room all round it, or drawing the hand back after letting go -- a "
    "longer step changes nothing there. That is the common case.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "both kinds from one moment to the next."
)

VIEW = (
    "You are shown 2 camera views of this one moment: a scene view and a wrist "
    "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
    "normally look close in it -- general closeness is normal. Use the wrist view to "
    "judge the close moments: the hand actually closing on the object, and the object "
    "being brought down into the place it must end up."
)

SCALE = (
    "  5 = clearly true of this moment\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
)

_AXES = (
    "A) Is the hand closing on the object right now, or about to -- the fingers\n"
    "   already around it and only the squeeze left, so that a slip here loses it?\n"
    "B) Is the place this object has to end up ROOMY FOR IT -- a broad plate or an\n"
    "   open dish with slack all round -- rather than a cup or a container barely\n"
    "   wider than the thing itself?\n"
    "C) Is the object ALREADY RESTING where it is meant to end up, taking its own\n"
    "   weight, with nothing left but to open the hand and draw back?\n"
    "D) Is the arm STILL LOWERING the object onto its target right now -- the hand\n"
    "   not yet opened, the object not yet taking its own weight, coming down onto\n"
    "   the spot it must end up on?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")
