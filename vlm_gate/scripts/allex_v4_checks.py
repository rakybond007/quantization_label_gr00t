"""allex v4 문항 -- 단일 배속 게이트용. `phase9_checks_v7` · `libero_v3c_checks` 와 같은 자리.

**앞 판과 무엇이 다른가.** v3 는 서브태스크마다 상한을 두고 그 안에서 K 를 놓는
2단 구조였다. v4 는 배속이 하나다(2.0 또는 2.5). 그러면 문항이 "몇 배속이냐" 를
정할 필요가 없고 **"이 순간이 깨지는 쪽인가" 만** 가르면 된다.

실측이 그 표적을 좁혀 준다(셀마다 30 에피, `analysis/allex_retention/
naive_speedup_eval.json`):

    Pass Box       1.000    Rotate PolyBag 0.867
    Pass PolyBag   0.967    Bring PolyBag  0.767   <- 2.5x 에서 막을 것
    Bring Box      0.900    Rotate Box     0.533   <- 2.0x 부터 막을 것

    2.0x 에서 막을 셀: Rotate Box 하나. 나머지 5셀 평균 유지 0.940
    2.5x 에서 막을 셀: Rotate Box · Bring PolyBag. 나머지 4셀 평균 유지 0.933

**물체 효과가 동작마다 뒤집힌다.** 옮길 때는 봉투가 나쁘고(0.767 < 0.900),
돌릴 때는 상자가 나쁘다(0.533 < 0.867). 그래서 "봉투인가" 를 묻는 문항은 부호가
상황마다 뒤집혀 못 쓴다 -- v3 의 FLIP 이 유지율과 -0.714 로 가장 크게 반대였던
이유가 이것이다.

깨지는 방식이 둘이고 서로 배타적이므로 **감점 문항을 그 둘에 하나씩** 둔다.

    A PINCH_RIGID   맞미는 힘만으로 붙든 딱딱한 것 -- 포즈를 솎으면 두 손의
                    차이가 바뀌고 빠진다. 봉투는 눌리며 물려 있어 버틴다.
                    표적: Rotate Box (0.533) · 가중이 더 크다
    B SWINGING_LOAD 매달려 흔들리거나 모양이 바뀌는 짐 -- 빨리 갈수록 더 흔들린다.
                    상자는 형태가 유지된다. 표적: Bring PolyBag (0.767)

가점은 실제로 안전한 것들에 준다.

    C FREE_END      정확히 놓을 자리가 없다 -- 옆으로 건네거나 밀어 보낸다.
                    표적: Pass (1.000 / 0.967)
    D EMPTY         손에 아무것도 없다.

**Bring Box 에 따로 가점을 주지 않는다.** 봉투를 옮기는 것이 B 에서 이미 감점을
받으므로, 상자를 옮기는 것은 감점이 안 걸리는 것만으로 충분히 갈린다. 앞 판에는
"아직 아래에서 받쳐져 있는가" 문항을 뒀었는데 끌기와 연결한 억지였고 문구도
읽히지 않았다.

v3 진단(38,590행)에서 다섯 중 넷이 4등급 이상 1% 미만이었고 가점 셋 중 둘의
부호가 반대였다. 그래서 문구를 전부 새로 쓴다.

**지시문을 넣는다.** v3 는 뺐다 -- 넣었더니 모델이 화면 대신 지시문을 보고 답해
칸 안에서 상수가 됐고(Rotate Box 77청크가 전부 한 값, 분산 0.112 -> 0.049),
그때 지시문은 **서브태스크 이름이라 칸마다 다른 상수**였기 때문이다.

v4 의 대상(`frontier_demo_cumul/v1_v2_v3_v4`)은 지시문이 **전 데이터셋에 하나**다
("Bring the package over, orient barcode up, then place it on the conveyor.").
구분 정보가 0 이므로 상수화를 일으킬 수 없고, 앞으로 어떤 동작이 오는지의 맥락은
된다. robocasa v7 과 libero v3c 도 에피소드 지시문을 넣는다 -- allex 만 빠져 있었다.

**그리고 이 데이터셋에서는 장면 말고 정보가 없다.** package 에 상자와 봉투가 다
들어가는데 지시문은 둘을 구분하지 않는다. 같은 "orient barcode up" 이 상자면
0.533, 봉투면 0.867 이다(3.3배). 서브태스크 라벨도 없다. A·B 가 물체 이름을 묻지
않고 "무엇이 이것을 붙들고 있나" 를 묻는 이유가 이것이다.

**가중은 잠정값이다(R5).** 라벨링 뒤에 등급으로 다시 잡는다. 각 변의 합은 1 --
conf = (1 + Σw·가점 - Σw·감점)/2, g=(등급-1)/(NGRADE-1) 이 [0,1] 에 들어간다.
"""
# 계산 사실·신뢰도 식·스냅은 v3 와 같은 것을 쓴다. 바뀐 것은 문항과 부호·가중뿐이고,
# 그래야 이 판이 문구 때문에 달라진 것인지 식 때문인지 가릴 수 있다(하네스 R2).
from allex_v3_checks import (confidence, expected_grades, facts_v3,  # noqa: F401
                             ratio_for, snap)

NGRADE = 5

SIGN = {"A": -1, "B": -1, "C": +1, "D": +1}

# 잠정. 감점은 A 가 더 크다 -- 표적인 Rotate Box 가 0.533 으로 가장 깊고,
# Bring PolyBag 은 0.767 이다. 가점은 실측 유지율 순서를 따른다.
WEIGHT = {"A": 0.60, "B": 0.40,
          "C": 0.60, "D": 0.40}

NAME = {"A": "PINCH_RIGID", "B": "SWINGING_LOAD", "C": "FREE_END", "D": "EMPTY"}

# `allex_v3_label.py` 가 읽는 이름. v3 는 문항 키가 이름이라 여기서 켤 것을 골랐는데,
# v4 는 키가 A~D 이므로 전부 켠 것과 같다. 라벨러를 그대로 쓰기 위해 둔다.
ACTIVE = tuple(sorted(SIGN))

GUIDANCE = (
    "You are judging one instant of a two-armed robot with hands, to decide how much "
    "of the next stretch of its motion could be thinned out -- how many of its "
    "commanded poses could be dropped, letting the arms travel further between the "
    "ones that remain, without changing the outcome.\n\n"
    "What thinning takes away is the arms' chance to correct on the way, and there are "
    "two ways that loses the object. One is a hold that exists only as a balance: the "
    "thing is trapped between two hands and nothing is closed around it, so the hold IS "
    "the push of one hand against the other. If that thing is stiff it cannot give, and "
    "a coarser swing changes the balance and drops it. Something soft squashes and "
    "stays pinched.\n\n"
    "The other is a load that does not hold its own shape. A bag or a sagging thing "
    "hangs, swings and shifts while it is carried, and the further the arms travel "
    "between commanded poses the further it swings.\n\n"
    "Most moments are neither. Sliding something across to the other side, pushing "
    "something that is still resting on a surface, or moving with nothing in the hands "
    "-- a longer step changes nothing.\n\n"
    "Judge the moment in front of you, not the name of the job. One segment passes "
    "through several of these from one second to the next."
)

SCALE = (
    "  5 = it is happening right now -- the picture shows the contact, the grip or\n"
    "      the position the check describes\n"
    "  4 = not yet, but the hands are right up against it, one motion away\n"
    "  3 = the hands are heading for it and still some way off\n"
    "  2 = the thing the check is about is there in the picture, but the arms are\n"
    "      busy with something else\n"
    "  1 = there is nothing in this picture the check could be about\n"
)

ASK = (
    "The measurements above are stated as fact -- do not re-estimate or repeat them. "
    "Answer each check from what the cameras show about the MOMENT in front of you, "
    "read together with those measurements.\n"
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one digit "
    "from 1 to 5 per check, rating how far that check describes this moment:\n"
    + SCALE +
    "A grade refers only to the check on that line.\n"
    "A) Is something STIFF being held only by one hand pressing against the other --\n"
    "   nothing closed around it, no fingers wrapped through -- so that the hold is\n"
    "   the balance between the hands and the thing cannot squash to stay caught?\n"
    "B) Is what the arms are moving something that DOES NOT HOLD ITS OWN SHAPE --\n"
    "   hanging, sagging, drooping over the hand, or swinging as the arms travel --\n"
    "   rather than a firm body that arrives the same way it set off?\n"
    "C) Is the thing only being SENT ACROSS -- slid or passed over to the other side,\n"
    "   or shoved on its way -- with no particular spot it has to come to rest on?\n"
    "D) Are the hands ON NOTHING AT ALL -- moving through open space or standing\n"
    "   still, with nothing in them and nothing being touched?\n"
    "Answer:"
)
