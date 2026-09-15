"""allex v4c -- A·B 문구를 짧게, D 제거. 척도는 v4b 의 상태형을 쓴다.

**v4b 가 무엇을 말해 주었나.** 척도를 시간형에서 상태형으로 바꿨더니 C 만
살아났고(4등급+ 0.0% -> 24.2%) A·B·D 는 그대로였다.

    A  1:1  2:12 3:152     3등급 92%  (오히려 더 뭉쳤다)
    B  1:7  2:155 3:3      2등급 94%
    C  1:15 2:55 3:55 4:40 24.2%      <- 살아났다
    D  1:161 2:4           1등급 98%

A·B 는 척도가 아니라 문구 문제다. 둘 다 **3중 연접**이었다 -- "딱딱하고 · 맞미는
힘만이고 · 손가락이 안 감긴", "모양을 못 지키고 · 늘어지고 · 흔들리는". 거기에
"rather than ..." 비교절까지 붙었다. 3중 연접에 "분명히 그렇다" 는 안 나온다.
살아난 C 는 단문이었다.

**연접 자체를 없애면 안 된다.** 쪼개서 A=딱딱한가, B=운반하는가로 만들면

        딱딱함  운반   실측
    Rotate Box    높   낮   0.533 위험
    Bring PolyBag 낮   높   0.767 위험
    Bring Box     높   높   0.900 안전
    Rotate PolyBag 낮  낮   0.867 안전

위험이 **XOR** 이 된다. conf 는 등급의 선형결합이라 XOR 을 못 만든다 -- 쪼개면
Bring Box 가 두 감점을 다 받아 가장 위험해진다. 그래서 연접은 유지하고 **읽히게만**
고친다: 3중 -> 2중, 비교절 제거.

**D 는 남긴다. 죽은 것이 아니라 틀린 답을 하고 있었다.** 계산 사실 기준으로
손에 아무것도 없는 청크가 **75.5%**(2,036/2,696)인데 D 는 그중 91.6% 를
1등급으로 답했다.

    빈손 구간의 D 등급: 1:1864 2:171 3:1
    잡은 구간의 D 등급: 1:639  2:21

서브태스크와 서브태스크 사이의 이동 구간이고 **가장 압축해도 되는 곳**이다.
청크의 3/4 을 차지하므로 빼면 그만큼을 통째로 놓친다.

문구가 원인이다. "ON NOTHING AT ALL ... nothing in them and nothing being
touched" 는 이중 부정에 절대 표현이라 긍정하기 어렵고, `held` 를 계산 사실이
이미 주는데 그것을 되묻고 있었다. 지금 무엇을 하는 중인지 -- **옮겨 가는
중인지** -- 를 묻는다.
"""
from allex_v4_checks import (confidence, expected_grades,  # noqa: F401
                             ratio_for, snap)
from allex_v4_checks import facts_v3 as _facts_v3_twohand


# ---------------------------------------------------------------------------
# 한 팔만 일하는 순간에 양손 개념을 말하지 않는다.
#
# 이 작업은 **양손을 쓰는 것이 박스 조작(돌리기 · 돌리기 좋게 맞추기)뿐**이고
# 나머지(가져오기 · 봉투 뒤집기 · 옆으로 보내기)는 전부 한 팔이다. 청크의
# 44.8% 가 한 팔인데, 기존 사실 문장은 거기서도 "손바닥이 멀리 떨어져 있다",
# "모여든다" 를 말한다 -- 한 팔만 일할 때 두 손목 사이 거리는 의미가 없고,
# 모델이 화면 대신 그것을 읽으면 오답이 된다.
#
# 손가락 관절값으로 한손 파지를 가려 보려 했지만 안 된다. 눈으로 라벨한 16
# 청크에서 최적 문턱의 정확도가 81.2% 이고, 빈손인데 손가락값이 파지 범위에
# 들어오는 경우가 있다(허공에서 미리 손 모양을 잡는 동작). **그래서 쥐었는지는
# 사실로 단정하지 않고 화면에 맡긴다.**
_GAP_CLAUSES = ("the palms are close in to each other",
                "the palms are a middling distance apart",
                "the palms are well apart",
                "and drawing together", "and moving apart",
                "and holding that distance")


def facts_v3(x, task=None):
    t = _facts_v3_twohand(x, task)
    if not x.get("one_handed"):
        return t
    for c in _GAP_CLAUSES:
        t = t.replace("; " + c, "").replace(c + "; ", "")
    # **단정하지 않는다.** 앞 판에서 "so nothing is being held between two palms
    # right now" 를 붙였더니 모델이 그것을 "아무것도 안 들었다" 로 읽어 D(이동
    # 중)가 한손 청크의 98.2% 에서 4등급 이상이 됐고, B(늘어진 짐)는 계속
    # 눌렸다. 한 팔이 일한다는 사실만 말하고, 무엇을 쥐었는지는 화면에 맡긴다.
    return t.replace("only one arm is moving",
                     "only one arm is working and the other is idle, so the "
                     "two-palm measurements do not apply here -- what that one "
                     "hand has, if anything, is for you to see")
from allex_v4b_checks import GUIDANCE, SCALE  # noqa: F401

NGRADE = 5
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1}
# 가중은 라벨 뒤에 다시 잡는다(하네스 R5). 769청크 라벨 위에서 재적합했고,
# 기준은 **conf 가 실측 세 셀의 순서를 재현하는 것**이다:
#   양손 RotateBox 0.533 < 한손 봉투 0.767 < 한손 상자 0.930
# 잠정값(A .60 B .40 C .55 D .45)에서는 봉투가 RotateBox 보다 낮게 나왔다 --
# B 가 과했다. 아래 값에서 순위상관 +1.000 이다.
# **가중과 역치는 gemini-3.8-flash(thinkingLevel low) 라벨에 맞춘 값이다.**
# 판정기가 바뀌면 다시 잡아야 한다(R5) -- 같은 문항에서도 Cosmos 는 A.8/B.2,
# Sonnet 은 A.6/B.4 가 최적이었다.
#
# 눈으로 표시한 정답지 119청크 · 18에피에서 적합했고, 에피 LOO 홀드아웃 AUC 가
# 0.865 로 전체적합 0.813 보다 높다(과적합 아님). C/D 배분은 C 를 0.3 에서
# 1.0 으로 올릴수록 LOO 가 단조 증가하고 0.8 에서 포화한다 -- 구석값이 아니다.
# D 를 완전히 버리지 않고 0.2 를 남긴다.
# **판정기마다 가중이 다르다.** 같은 문항이라도 판정기가 바뀌면 라벨이 바뀌므로
# 가중도 다시 잡는다(R5). 눈으로 표시한 정답지 119청크 · 18에피에서 적합했고
# 괄호 안은 에피 LOO 홀드아웃 AUC 다.
#
#   gemini-3.8-flash(low)  A .70 B .30 C .80 D .20  TAU 0.513   (0.865)
#   cosmos3-nano           A .80 B .20 C .70 D .30  TAU 0.462   (0.754)
#   sonnet-4.5             A .60 B .40 C .10 D .90  TAU --      (0.909)
#
# 라벨러는 `ALLEX_JUDGE` 로 고른다. 안 주면 gemini 값을 쓴다 -- 등급만 맞으면
# conf 는 나중에 다시 계산되므로, 잘못 써도 라벨이 못 쓰게 되지는 않는다.
import os as _os
_W = {
    "gemini": ({"A": 0.70, "B": 0.30, "C": 0.80, "D": 0.20}, 0.513),
    "cosmos": ({"A": 0.80, "B": 0.20, "C": 0.70, "D": 0.30}, 0.462),
    "sonnet": ({"A": 0.60, "B": 0.40, "C": 0.10, "D": 0.90}, 0.500),
}
WEIGHT, TAU = _W.get(_os.environ.get("ALLEX_JUDGE", "gemini"), _W["gemini"])
# 769청크에서 0.539 로 잡았는데 9,179청크 전량 분포에서는 과하게 보수적이었다
# (압축 24.8%, K=2.5 스텝 단축 14.9% 뿐). 0.458 은 압축 67.2% · 스텝 단축
# 40.3% 이고, 769청크 기준으로 안전셀 96.0% 통과 · 위험셀 49.1% 차단이었다.
# **역치는 라벨 분포에서 정한 값이고 폐루프로 검증하지 않았다.** allex 는
# 폐루프가 없으므로, 실제 eval 에서 성공률을 보고 조정해야 한다.
NAME = {"A": "PINCH_RIGID", "B": "POUCH", "C": "FREE_END", "D": "IN_TRANSIT"}
ACTIVE = tuple(sorted(SIGN))

_AXES = (
    "A) Are the two hands SQUEEZING something between them to hold it, and is that\n"
    "   thing HARD -- a box, a carton, something that will not squash?\n"
    "B) Is the item a POUCH OR PACKET -- a soft mailer, a padded envelope, a\n"
    "   plastic parcel -- rather than a rigid carton with square sides?\n"
    "C) Is the thing only being SENT ACROSS -- slid or passed over to the other side,\n"
    "   or shoved on its way -- with no particular spot it has to come to rest on?\n"
    "D) Are the hands ON THEIR WAY -- reaching out towards something, or drawing back\n"
    "   from what they have finished, carrying nothing between them?\n"
)

ASK = (
    "The measurements above are stated as fact -- do not re-estimate or repeat them. "
    "Answer each check from what the cameras show about the MOMENT in front of you, "
    "read together with those measurements.\n"
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one digit "
    "from 1 to 5 per check, rating how far that check describes this moment:\n"
    + SCALE +
    "A grade refers only to the check on that line.\n"
    + _AXES + "Answer:"
)
