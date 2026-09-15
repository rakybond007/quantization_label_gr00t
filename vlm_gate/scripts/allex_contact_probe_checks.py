"""접촉을 VLM 문항만으로 뽑을 수 있는지 재는 독립 문항 모듈.

검증된 v4c(A~D)를 건드리지 않는다. 문항을 덧붙이면 기존 문항의 답이 바뀐다는
음성 결과가 이미 있다 (5번째 TURNING 문항을 넣었을 때 AUC 0.842 -> 0.718).
그래서 접촉만 따로 물어 같은 정답지로 잰다.

대조: _tmp/eye_labels.json 의 -1(불가 = 접근/이송) 37청크 vs 나머지 119.
기준선 두 개 -- 기존 D 문항을 뒤집으면 AUC 0.667, 손 토크 신호는 0.750.

세 표현을 따로 묻는다. 어느 쪽이 사람 라벨과 맞는지 보려는 것이다.
  P) 지금 닿아 있나            (상태)
  Q) 지금 막 잡거나 놓는 중인가   (경계 -- 형님이 물은 contact boundary)
  R) 물체가 저항하고 있나        (힘의 결과)
"""
NGRADE = 5
SIGN = {"P": +1, "Q": +1, "R": +1}
NAME = {"P": "TOUCHING", "Q": "MAKING_OR_BREAKING", "R": "RESISTING"}
WEIGHT = {"P": 0.34, "Q": 0.33, "R": 0.33}
TAU = 0.5

GUIDANCE = (
    "You are shown one moment from a two-armed robot working on a table. "
    "Answer only about what these images show. Do not guess what happens next."
)

ASK = """Answer each with a whole number from 1 to 5.
  1 = definitely no
  2 = probably no
  3 = cannot tell
  4 = probably yes
  5 = definitely yes

P) Is a hand TOUCHING an object right now -- fingers or palm against it, close
   enough that moving the hand would move the object?

Q) Is a hand RIGHT AT THE MOMENT of taking hold or letting go -- fingers closing
   onto something, or opening off something they were holding?

R) Is an object PUSHING BACK on a hand -- being squeezed, lifted, dragged or
   turned, so the hand has to work against it?

Answer with exactly three lines and nothing else, in this form:
P) <number>
Q) <number>
R) <number>"""


def facts(x):
    """접촉 문항은 계산 사실을 받지 않는다.

    이 실험이 답하려는 것은 '장면만으로 접촉을 알 수 있나' 다. 손 토크에서 나온
    수치를 사실로 넣으면 그 신호를 되읽는 것이 되어 비교가 무의미해진다.
    """
    return ""
