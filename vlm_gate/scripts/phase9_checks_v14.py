"""robocasa 문항 v14 후보 = **v10 + 계산 사실에 압축 요구 두 문장.**

문항·GUIDANCE·척도·부호는 v10 과 글자까지 같다. 바뀐 것은 **계산 사실뿐**이고
그것도 이 파일이 아니라 `robocasa_v14_descriptors.py` 에 있다.

## 왜

allex 와 humandata 의 계산 사실에는 압축 요구 두 문장이 있는데 robocasa 에만
없었다. **포맷을 맞춘다는 것은 채워 넣는 구성요소를 같게 한다는 뜻**이고, 이 자리가
비어 있었다. 내용(단위·값·분위)은 벤치마크마다 다른 것이 맞다.

    at 2x this stretch would ask for 1.00 in one step, about average
    at 3x 1.46, larger than most

robocasa 는 델타 EEF 라 병합이 K 개 델타의 **합**이다(절대 타깃인 allex 의
블록-라스트와 다르다). 분위대 경계는 이 데이터셋 자기 분포에서 낸다(창 5,131개).

문서에 이미 적혀 있던 결손이기도 하다 -- "#10 은 2배속 요구가 시연 한계를 넘는데
네 문항 어디에도 그것을 묻는 자리가 없다. 계산 사실이 말해 주는데 문항이 받지
못한다." 문항을 늘리는 대신 사실로 준다.

## 주의

**사실 문장을 늘리면 모든 문항의 답이 바뀐다** -- allex 에서 두 문장을 더했을 때
Gemini 답의 36~51% 가 바뀌었다. 그래서 재야 한다.
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
    "F) Is the arm LOWERING what it still holds onto its target right now -- the\n"
    "   object still in the gripper, coming down onto the spot it must end up, not\n"
    "   yet let go?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")
