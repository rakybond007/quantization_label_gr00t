"""robocasa 문항 v8 = **v7 문항 그대로, 포맷만 정규화한 판.**

v7 을 그 자리에서 고치면 안 된다 -- 이름이 가리키는 것이 바뀌어 버린다. 실제로 그
사고가 있었다: 09-05 의 204만 청크를 만든 프롬프트가 무엇인지 복구할 수 없고, 레포에
"v7 전문" 으로 올라간 것은 그 라벨보다 10일 뒤에 만든 판이다. 그래서 갈라 둔다.

**문항 글(_AXES)·GUIDANCE 의 내용·부호·가중은 v7 과 같다.** 바뀐 것은 셋뿐이다.

1. **등급 척도를 값 척도로.** v7 은 물리적 근접 사다리였다("4 = 한 센티미터나 한 동작
   거리"). 그것은 사건 문항에만 맞고 성질 문항에는 대응 상태가 없다 -- C 는 "이 물건이
   기구에 붙어 있나" 라서 "1cm 앞" 이 없다. allex(Gemini) 가 값 척도이고 그 라벨은 다시
   만들지 않으므로 새로 만드는 쪽이 맞춘다(`prompts/FORMAT.md` §2).

   근접 사다리를 쓴 본래 이유는 값 척도의 2등급이 안 쓰인다는 것이었다(43/9,490 =
   0.45%). **그건 값 척도 자체가 아니라 옛 문구의 문제였다** -- 옛 2등급은 "mostly does
   not hold" 로 1등급과 눈으로 구분되지 않았다. allex 문구("2 = barely true")에서는
   48,086청크 실측 2등급 사용률이 52.2% 다.

2. **기간 주장을 뺀다.** "one second" -> "one moment". 창은 16스텝인데 robocasa 는
   20fps 라 0.80초다(1.25배 과장). 같은 문구를 libero(10fps, 1.60초)로 옮기면 배로
   틀린다. allex 도 기간 대신 "the next stretch of its motion" 이라고만 한다.

3. **SYSTEM 블록을 안 쓴다.** 이건 이 파일이 아니라 조립부(`vlm_gate.build_messages`
   의 `user_only`)에 있다. v7 때는 이진 게이트용 SYSTEM 이 등급 문항과 같이 나갔고,
   그 마지막 줄이 "Answer with exactly one word: YES or NO." 였다.

## v7 대비 무엇이 달라지나 (측정)

포맷만 바꾼 대조(문항·가중 동일, 568청크)에서 답의 21% 만 다섯 문항 전부 같았고
주로 B 가 바뀌었다(25% 일치). 가중 재적합 뒤 정답 상관은 포맷 차이가 잡음과 구분되지
않는다(쌍대 차이 사다리 +0.01 [−0.29,+0.28] · 천장 +0.20 [−0.26,+0.60]).

**즉 v8 은 성능 때문이 아니라 정합성 때문에 만든 판이다.** SYSTEM 이 등급 문항에 대고
YES/NO 한 단어를 요구하던 것, 기간을 1.25배로 말하던 것을 고친 것이다.

## 주의

이 문항으로 라벨을 만들면 **가중을 그 등급 위에서 다시 잡는다**(R5). v7 라벨에 맞춘
`robocasa_v7r_checks` 의 가중을 그대로 쓰면 안 된다.
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
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1}

# 문항 이름. **문구·부호·가중은 한 글자도 안 바뀐다** -- 부록 생성기와 진단표가
# 읽을 이름만 붙인 것이고, 라벨에 영향을 주지 않는다(libero_v3c_checks 와 같은 모양).
NAME = {"A": "CLOSE_EXACT", "B": "HEMMED_IN", "C": "FIXTURE_HELD",
        "D": "LETTING_GO", "E": "CARRY_OPEN"}
WEIGHT = {"A": 0.60, "B": 0.40,
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
    "A) Does this motion need the gripper to CLOSE on one exact spot -- a handle, a rim,\n"
    "   the body of an object it must pick up -- rather than only pushing or turning\n"
    "   something that stays where it is?\n"
    "B) Is that closing spot hemmed in -- reached into a cabinet, a microwave, an oven or\n"
    "   a sink, or lined up with a slot or a handle -- so the gripper has little room\n"
    "   around it as it closes?\n"
    "C) Is the thing being acted on held in place by the furniture or appliance itself\n"
    "   -- a door or drawer on its hinge or rail, a knob, switch, button or lever on its\n"
    "   mount -- so the mechanism, not the arm, decides where it ends up?\n"
    "D) Is the hand letting go, or about to -- the object already resting where it is\n"
    "   meant to end up, so that what remains is only to open and withdraw?\n"
    "E) Is the gripper carrying something through open space -- already holding it,\n"
    "   still well away from wherever it is going?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")
