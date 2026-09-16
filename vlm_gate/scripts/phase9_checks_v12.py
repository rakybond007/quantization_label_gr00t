"""robocasa 문항 v12 후보 = **v10 + 놓기 문단 + F 문구 다듬기.** 아직 라벨을 만들지 않았다.

## v11 에서 배운 것 -- GUIDANCE 의 순위 문장은 일을 하고 있었다

v11 에서 GUIDANCE 의 `matters most / most of all / least` 를 "정답표" 로 보고 걷어냈다.
**틀린 판단이었고 측정으로 확인됐다.** 국면 순서 재현 여유가 0.088 -> 0.016 으로
무너졌다. 등급을 보면 원인이 분명하다:

    C 등급  v10  접근 3.23 · 운반 2.80 · 해제놓음 2.14 · 파지 2.76 · 해제쥔채 2.31  (구분함)
            v11  접근 2.81 · 운반 2.81 · 해제놓음 2.19 · 파지 2.68 · 해제쥔채 2.64  (평평)

`It matters least where the arm only pushes or turns something ... on its hinge, rail
or mount` 이 줄이 C 를 갈라 주고 있었다.

**SYSTEM 이 나빴던 이유는 "판단 방법을 알려줘서" 가 아니었다.** 둘이었다:

1. 답 형식이 모순됐다 -- SYSTEM 은 YES/NO 한 단어를, 문항은 1~5 등급을 요구했다.
2. 매핑이 틀렸다 -- "reaching/carrying = YES(압축 가능)" 인데 폐루프는 반대였다.

robocasa GUIDANCE 는 둘 다 해당 없다. 답 형식을 말하지 않고, 매핑이 실측과 맞는다
(기구가 잡아주는 국면이 contact_knob 3.17 · place_open 3.13 으로 실제로 관대하다).
allex 와 libero GUIDANCE 도 똑같이 무엇이 위험한지 말한다 -- 그게 GUIDANCE 가 하는 일이다.

## v10 에서 바꾼 것 둘

1. **놓기 문단을 더한다.** v10 GUIDANCE 는 닫기(A·B)와 기구가 잡음(C)만 설명하고
   F·D 의 물리가 없었다 -- F 를 넣으면서 GUIDANCE 를 안 건드린 탓이다. **순위 문장은
   그대로 두고** 문단만 보탠다.
2. **F 의 구문 중의성을 없앤다.** `LOWERING what it still holds onto its target` 은
   `holds onto`(붙잡다)가 한 덩어리로 읽혀 갈린다. `SETTING DOWN what it still holds`
   로 끊는다.

A·B·C·D·E 문항, 척도, 부호, 이미지 장수, 조립 순서는 v10 과 같다. 가중은 잠정이다(R5).
"""




NGRADE = 5

# 가중은 실측된 분리 폭에서 읽었다. 두 쪽 합은 v1 처럼 각각 1 이다.
#
# A = 0.6 -- "닫아야 하는가" 는 단계 0(1.032) 과 1~3(평균 0.79) 을 가른다. 폭 0.24.
# B = 0.4 -- "그 자리가 좁은가" 는 단계 1(0.969) 과 2~3(평균 0.75) 을 가른다. 폭 0.22
#            로 A 에 거의 맞먹고, K3 에서는 오히려 더 크다(0.828 대 0.502 = 0.33 對
#            A 의 0.32). v1 에서 B 를 0.333 으로 두고도 아무 일이 없었던 것은 무게가
#            아니라 겨눈 곳이 틀렸기 때문이므로, 다시 겨눈 B 에는 무게를 준다.
#
# C = 0.50 -- 기구가 경로를 정한다는 것은 실측된 안정 쪽 논거이고, 다섯 문항 중 모델이
#            유일하게 확신하는 문항이다(뽑힌 등급 확률 0.773, 엔트로피 0.407,
#            0.5 미만 10%). 가점 쪽 최대.
# D = 0.15 -- "넓은 면에 내려놓는다" 에 가점을 주는데, 그 넓은 면은 조리대이고
#            *ToCounter 가 오히려 약한 쪽이다(0.766). D 가 가장 잘 맞는 태스크들에서
#            증거가 반대를 가리키므로 최소로 둔다. 문구는 그대로 두었다 -- 지우는
#            것보다 무게를 내리는 쪽이 변경이 작다.
# E = 0.35 -- "열린 공간을 건너 옮긴다" 는 PnP 두 부류에 모두 걸려 중립 증거이고,
#            모델도 잘 못 답한다(확률 0.407). 가운데.
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1, "F": -1}

# 문항 이름. **문구·부호·가중은 한 글자도 안 바뀐다** -- 부록 생성기와 진단표가
# 읽을 이름만 붙인 것이고, 라벨에 영향을 주지 않는다(libero_v3c_checks 와 같은 모양).
NAME = {"A": "CLOSE_EXACT", "B": "HEMMED_IN", "C": "FIXTURE_HELD",
        "D": "LETTING_GO", "E": "CARRY_OPEN"}
# F 가중은 **잠정**이다 -- 위험 쪽 셋을 균등으로 두고, 라벨링 뒤에 실측에
# 맞춰 다시 잡는다(R5). 지금 값으로 성능을 판단하면 안 된다.
WEIGHT = {"A": 0.40, "B": 0.27, "F": 0.33,
          "C": 0.50, "D": 0.15, "E": 0.35}

# 기간을 글로 못 박지 않는다. 창은 16스텝인데 robocasa 는 20fps 라 0.80초,
# libero 는 10fps 라 1.60초다. "one second" 라고 쓰면 robocasa 에서 25% 과장이고
# 같은 문구를 libero 에 옮기면 배로 틀린다. allex 도 기간 대신 "the next stretch
# of its motion" 이라고만 한다.
GUIDANCE = (
    "You are judging one moment of a kitchen robot's motion, to decide how much of it "
    "could be thinned out -- how many of its commanded poses could be dropped, letting "
    "the arm travel further between the ones that remain, without changing the outcome.\n\n"
    "What thinning takes away is the arm's chance to correct itself on the way. That "
    "matters most where the gripper has to close on one exact spot, and it matters most "
    "of all when that spot is hemmed in. It matters least where the arm only pushes or "
    "turns something that is already held in place by its own hinge, rail or mount.\n\n"
    "Setting an object down asks for the same care while it is still in the gripper: "
    "it has to come to rest where it is meant to, and a longer step sets it down askew "
    "or knocks it against what is around it. Once it is down and taking its own weight, "
    "opening the fingers and drawing back no longer asks for that.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "both kinds from one moment to the next."
)

# 척도는 **allex(Gemini) 와 같은 값 척도**다. v7 까지는 물리적 근접 사다리였다
# ("4 = 한 센티미터나 한 동작 거리"). 두 가지 때문에 바꿨다.
#
# 1. 근접 사다리는 사건 문항에만 맞고 성질 문항에는 대응되는 상태가 없다. C 는
#    "이 물건이 기구에 붙어 있나" 를 묻는 것이라 "1cm 앞" 이 없다. SYSTEM 을 바꿔도
#    robocasa C 가 94% 그대로였던 것이 그 흔적이다.
# 2. 포맷을 벤치마크마다 다르게 가져갈 이유가 없다. allex 라벨은 값 척도로
#    만들어졌고 다시 만들지 않는다 -- 새로 만드는 쪽이 맞춘다(`prompts/FORMAT.md` §2).
#
# 근접 사다리를 쓴 본래 이유는 값 척도의 2등급이 안 쓰인다는 것이었다(43/9,490 =
# 0.45%). **그건 값 척도 자체가 아니라 옛 문구의 문제였다.** 옛 2등급은 "mostly does
# not hold" 로 1등급과 눈으로 구분되지 않았다. allex 문구("2 = barely true")에서는
# 48,086청크 실측 2등급 사용률이 52.2% 다.
SCALE = (
    "  5 = clearly true of this moment\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
)

_AXES = (
    "A) Is the gripper CLOSING on its target right now, or about to -- the fingers\n"
    "   already around the handle, rim or object and only the squeeze left, so that a\n"
    "   slip here loses it?\n"
    "B) Is that closing spot hemmed in -- reached into a cabinet, a microwave, an oven or\n"
    "   a sink, or lined up with a slot or a handle -- so the gripper has little room\n"
    "   around it as it closes?\n"
    "C) Is the thing being acted on held in place by the furniture or appliance itself\n"
    "   -- a door or drawer on its hinge or rail, a knob, switch, button or lever on its\n"
    "   mount -- so the mechanism, not the arm, decides where it ends up?\n"
    "D) Is the object ALREADY RESTING where it is meant to end up, taking its own\n"
    "   weight, with nothing left but to open the hand and draw back?\n"
    "E) Is the gripper carrying something through open space -- already holding it,\n"
    "   still well away from wherever it is going?\n"
    "F) Is the arm SETTING DOWN what it still holds -- the object in the gripper,\n"
    "   coming down onto the place it must end up, not yet released?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")
