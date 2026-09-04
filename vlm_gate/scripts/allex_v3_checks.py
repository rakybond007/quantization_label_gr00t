"""allex checks derived the way robocasa's were: from measured damage.

The v2 ceilings were the annotator's priors, because allex had no eval. It now
has one -- success out of 30 replays per subtask, at 2x and 2.5x:

                    2x        2.5x      2.5 - 2
    Rotate Box      22/30     16/30     -20.0 pt      위험
    Bring PolyBag   28/30     23/30     -16.7 pt      위험
    Bring Box        .        27/30      ~0           안정
    Pass Object      .       ~30/30      ~0           안정
    no contact       -         -         (pinned)     안정

so the pools are Rotate Box + Bring PolyBag against Pass + Bring Box + the
phases where nothing is being touched.

WHAT THIS OVERTURNS. v2 read the bag as the forgiving object: Rotate split
2.0/2.5 with the bag ON TOP, and BRING_SOFT lifted Bring Object from 2.0 to 2.5
whenever check D said "soft plastic bag". The measurement says the opposite --
carrying a bag is the phase that breaks first, 28/30 down to 23/30, while the
box carries fine to 2.5. Every Bring chunk the VLM called a bag was therefore
labelled with a ceiling 0.5 too HIGH, and in the direction that loses episodes.
The bag is forgiving to LAND (it deforms instead of toppling) and unforgiving to
CARRY (it swings and slips), and v2 collapsed those into one number.

WHERE THE NUMBERS COME FROM. There is no eval loop here -- this is a real
robot, and the ratios are the operator's, not something to be re-measured on
demand. The replay counts above are what was observed; 3.0 for passing and for
carrying a box is given as a weak allowance, and Rotate Box is to land between
1.5 and 2.0. The candidate set stops at 3.0: there is no 4x for this robot.

Rotate PolyBag was never replayed, so no check here claims it.
"""
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_common import MERGE_LIMIT_V2  # noqa: E402

# Each check carries the ratio the phase it recognises was measured to survive.
# There is no separate sign: a check that raises the answer and one that lowers
# it differ only in the number, and the answer is their grade-weighted mean.
#
#     lowers    A 2.0  a firm object held only by two palms pressing on it
#               B 2.0  carrying something that hangs and changes shape
#     raises    C 2.5  sliding an object across to the other side
#               D 2.5  touching nothing at all
#               E 2.5  carrying something the hand is closed around
#
# The model is NOT told which way each pushes, for the same reason as robocasa:
# told the direction, it answers toward the ratio it thinks is wanted instead of
# describing what it sees.
# Three checks, not five. The task is small -- one station, three kinds of
# object, one surface -- and the first cut asked five questions of scenes that
# only differ in two or three ways. Four of the five then answered the same
# digit on 99% of chunks.
#
#     lowers   A 2.0  the thing under the hand is a limp poly mailer
#              B 1.5  the object is being turned so a different face comes up
#     raises   C 3.0  the hand is pushing it along the plate, not lifting it
#
# and the base is 2.5, the ratio given for the phases where nothing is being
# handled at all. Nothing answered therefore lands there, which is what it
# should mean: no check recognised the moment.
# Three checks, not five. The task is small -- one station, three kinds of
# object, one surface -- and the first cut asked five questions of scenes that
# differ in two or three ways.
#
# The three name what is being DONE, and between them they partition it: the
# object is being moved, or it is being turned, or neither is happening. A is
# not an action but what the action is being done to, which is the one thing
# here the numbers cannot say.
#
#     lowers   A 2.0  what is handled goes out of shape under the hand
#              B 1.5  the object is turned so a different side faces up
#     raises   C 3.0  the object is moved somewhere, the same side still up
#
# and the base is 2.5, the ratio given for the phases where nothing is being
# handled. Nothing answered lands there, which is what it should mean.
# The checks came out of the pools, not out of these numbers. The measured
# ratios were used the way PROMPT_METHOD says to use them -- as the hint for
# which subtasks are the damaged ones and which are not -- and the ratio below
# is only attached to each phase afterwards, once the phase had earned its
# place by coverage.
#
#     위험 풀   Rotate Box     2x 22/30, 2.5x 16/30
#               Bring PolyBag  2x 28/30, 2.5x 23/30
#     안정 풀   Pass Object    ~30/30 at 2.5x
#               Bring Box      27/30 at 2.5x
#               Rotate PolyBag operator: flipping one fast is fine
#
# Ranking the candidate phases by how many subtasks they cover, minus what they
# wrongly cover on the other side:
#
#     firm thing moved somewhere       +2 -0 = 2   kept, 안정
#     limp thing set down at a place   +1 -0 = 1   kept, 위험
#     firm thing turned in two hands   +1 -0 = 1   kept, 위험
#     limp thing flipped over          +1 -0 = 1   kept, 안정
#     set down at an exact place       +1 -1 = 0   dropped
#     the thing is limp                +1 -1 = 0   dropped
#
# The last one is the finding. NEITHER the object NOR the action separates the
# pools on its own: limp is dangerous to bring and safe to flip, firm is
# dangerous to turn and safe to move. A check that names only one of the two
# covers a damaged subtask and an undamaged one equally and cancels. So every
# phase here is (what, done how) as one thing.
# B(봉투)의 상한은 하나가 아니다. 봉투를 뒤집는 것은 2.5, 옮기는 것은 2.0
# 인데 그 둘은 정지 화면에서 같은 장면이다 -- 다섯 번 물었고 다섯 번 다
# 중간 등급에 붙박였다. 가르는 것은 손바닥 간격이고(AUC 0.827) 그건 이미
# 계산돼 사실로 나간다. 그래서 코드가 고른다: 묻지 않고 준다는 원칙 그대로다.
CEILING = {"A": 1.5, "B": 2.5, "C": 3.0}
BAG_FLIP, BAG_MOVE = 2.5, 2.0
COVER = {"A": 1, "B": 2, "C": 2}      # step 5, recorded; the ratio carries
# 등급 수는 robocasa 의 5 를 따라갈 이유가 없다. 이 데이터에서 정한다.
# 열세 바퀴 내내 모델이 실제로 쓴 등급은 셋이었다 -- 2 와 4 는 거의 안 쓰였고
# 쓰인 판에서도 1·3·5 의 변형이었다. 카메라 두 대가 거의 같은 각도의 넓은
# 화면이라 "한 동작 남았다" 와 "지금 일어난다" 가 눈으로 안 갈린다.
# 3 단으로 내렸다가 되돌린다. 근거로 삼았던 "2 와 4 가 안 쓰였다" 는 그 판들의
# 등급표가 문항마다 따로였고 중간 칸이 "반쯤" 이었기 때문이지 눈금 탓이 아니었다.
# 3 단으로 쓴 판의 2등급은 "곧 일어날 참" 과 "대상은 있는데 팔은 딴 일" 의
# 합집합이라 어느 프레임에서도 방어됐다 -- robocasa 는 그 둘을 4 와 2 로
# 갈라놓았고 2 등급이 8.5% 실제로 쓰였다.
NGRADE = int(os.environ.get("ALLEX_NGRADE", 5))

# There is no 4x here. The candidate ratios for this robot are these five, and
# a label that is not one of them cannot be replayed.
CANDIDATES = (1.0, 1.5, 2.0, 2.5, 3.0)



# Candidates that were dropped, and why -- the ranking is the method's step 3.
#
#   "is it being parked at one exact spot"      covers Bring PolyBag (위험) AND
#       Bring Box (안정): +1 -1 = 0. The place is not what separates them.
#   "is the object firm rather than floppy"     covers Bring Box and Pass (+2)
#       but also Rotate Box (-1), and it is B negated, which counts the same
#       evidence twice. Dropped on both counts.
#   "is the object being turned over"           covers Rotate Box, same as A,
#       but says nothing about WHY turning is fragile. A is the mechanism.
GUIDANCE = (
    "You are judging one instant of a two-armed robot with hands, to decide how far the "
    "next stretch of its motion could be thinned out -- how many of its commanded poses "
    "could be dropped, letting the arms travel further between the ones that remain, "
    "without changing the outcome.\n\n"
    "What decides it is how the object is being held right now. A hand closed around a "
    "box keeps it through a coarse swing. A box held only by two palms pressing inward "
    "does not: the hold IS the difference between the two hands, and thinning the poses "
    "changes that difference. Something that hangs and swings is its own case again -- "
    "there is no rigid body to keep, and the faster the arm moves the further it "
    "swings.\n\n"
    "Judge the moment in front of you, not the name of the job. One segment passes "
    "through several of these from one second to the next."
)

# D is pinned rather than ranked, the way robocasa's E is. Phases where nothing
# is held or touched sit inside every subtask, the damaged ones included, so a
# phase common to all of them can never separate the pools. With no object in
# hand there is no hold to lose, so it takes the safe ceiling by construction.
# t=4 -> t=5, 문항 A 재서술(셋째). A 는 세 판 연속 3등급에 붙박였고, 세 판 다
# 3등급을 "일부만/반쯤" 상태로 썼다 -- 같은 실수를 세 번 했다. 어느 프레임에서도
# 방어되는 등급을 만들면 모델은 거기로 간다.
#
# 그리고 t=4 는 문항끼리 간섭한다는 것을 보여줬다. A 만 고쳤는데 C 가 +2.45 에서
# +0.68 로 깨졌다. 넷을 한 프롬프트에서 같이 답하므로, A 가 봉투를 "통통한"
# 것으로 부르자 모델이 더는 그것을 C 의 "납작한 주름진 것" 으로 보지 않았다.
# **한 문항의 서술은 다른 문항의 답을 바꾼다.** 그래서 R3(한 바퀴에 한 축)은
# 한 문항만 고쳐도 넷을 다 다시 재라는 뜻이기도 하다.
#
# 이번에는 봉투의 모양이 아니라 손 모양으로 간다. 두 봉투 층이 실제로 갈리는
# 자리다: 뒤집는 쪽은 손가락이 봉투를 감아쥐고, 옮기는 쪽은 손가락이 펴진 채
# 옆면에 닿아 있다. D 가 상자에서 하는 구분(한 손이냐 두 손 사이냐)과 같은
# 종류이고, D 는 그것으로 +2.02 를 냈다.
#
# t=3 -> t=4, 문항 A 재서술. 검증 3 에서 A 만 남았다(-0.02). 두 봉투 층의
# 프레임을 실제로 뜯어보니 A 가 이름 붙인 국면 -- 봉투를 손가락으로 집어
# 판에서 들어 올린다 -- 이 여기 없다. 봉투는 들리지 않는다. 판 위에서 밀리고
# 판 위에서 잡힌다. 그러니 A 는 없는 장면을 물었고, 세 등급을 다 쓰면서도
# 어느 층에서나 같은 답이 나왔다.
#
# 두 봉투 층이 실제로 갈리는 자리는 하나뿐이다: **봉투가 손에 눌려
# 우그러졌는가.** 뒤집는 쪽은 손가락이 파고들어 모양이 무너져 있고, 옮기는
# 쪽은 통통한 모양을 그대로 지킨 채 손이 옆면에 붙어 있다. 정지 화면이 담는
# 차이다.
#
# t=2 -> t=3, 등급표 축 하나. 검증 3 에서 A(-0.08)와 D(+0.46)가 떨어졌고
# 원인이 같다: **1등급이 경쟁 배치를 담고 있지 않았다.** A 의 1등급은 "손
# 가까이 주름진 것이 없다" 였는데 turn+bag 장면에는 주름진 것이 손 밑에 있다.
# 그 층에서 1을 고를 수 없으니 3으로 올라오고, 봉투 층 둘이 똑같아진다.
# D 도 같은 이유로 turn+box 에서 켜졌고 그것이 B-D 상관 +0.62 다.
# 이제 각 문항의 1등급이 그 문항이 아닌 쪽의 배치를 이름으로 부른다 --
# A 의 1등급은 C 의 장면이고, D 의 1등급은 B 의 장면이다.
#
# WHAT THE EARLIER DRAFTS GOT WRONG, so the next one does not walk back in.
#
#   ASKED WHAT THE FACTS ALREADY SAY. A draft asked whether the object was
#     pinched between two palms; descriptors() computes that as `held`. 99% of
#     chunks came back at the middle grade. `wrist_rot`, `rot_asym`,
#     `one_handed`, `hand_change` are stated too.
#   ASKED WHAT A STILL CANNOT SHOW. The next draft asked whether the object was
#     "being taken somewhere" and "being turned over". The model sees one frame
#     per camera; motion is not in it. 91% answered the rung that reads "the
#     hands are on it and it has not moved yet", and the turn/move contrast came
#     out at -0.03 and -0.07 -- no separation at all. Every rung below is a
#     configuration that a single frame holds: what is on the plate, what is off
#     it, where the hands are, whether an edge is up.
#   ASKED SOMETHING ALWAYS TRUE. An earlier C asked whether open surface lay
#     ahead. On a sorting plate it always does: 95.4% answered 5.
#   LET THE MIDDLE RUNG SWALLOW THE SAFE CASE. "it mostly keeps its form and
#     only the touched face dents in" describes a cardboard box exactly, so the
#     firm case scored the same as the limp one (2.89 vs 3.00) and the one check
#     that had been working stopped working.
#   USED THE MIDDLE AS A HEDGE. Grades 2 and 4 went unused across 763 chunks.
#     Grade 2 is now the shape that worked in robocasa -- the subject IS on the
#     plate and the hands are on something else.
# 상한도 하한도 문항이 정하지 않는다. 태스크가 준다 -- robocasa 는 naive
# quantization eval 표에서, allex 는 운용자가 알려준 값에서. 문항이 정하는
# 것은 그 사이를 얼마나 쓸 것인가뿐이다.
#
#                    상한   하한
#     Bring Box      3.0    2.0
#     Pass Object    3.0    2.0
#     Bring PolyBag  2.0    1.5     하한이 상한과 같아지면 한 칸 내린다
#     Rotate PolyBag 2.5    2.0
#     Rotate Box     1.5    1.0
#
# 앞 판은 문항이 (행동, 물체) 를 알아내게 만들었고, 그래서 칸을 모르는 상태가
# 출발점이 되어 모든 태스크가 기준값 2.5 를 공유했다. 상한이 2.5 보다 낮은
# 태스크는 증거가 약할 때 상한 위에 남았다(move+bag 위반 100%). 지금은 K 가
# [하한, 상한] 안에서만 움직이므로 넘는 일이 구조적으로 없다.
TASK_RANGE = {
    "move+box": (2.0, 3.0),
    "move+bag": (1.5, 2.0),
    "turn+bag": (2.0, 2.5),
    "turn+box": (1.0, 1.5),
}
DEFAULT_RANGE = (2.0, 2.5)
CANDIDATES = (1.0, 1.5, 2.0, 2.5, 3.0)
# NGRADE 는 위에서 한 번만 정한다. 두 번 정의해 놓았다가 나중 것이 이겨서
# 등급표는 3단인데 프롬프트가 1~5 로 나갔고, 모델이 4·5 를 답했다.

# 감점 / 가점과 그 무게. 무게는 덮는 태스크 수이고 각 변의 합이 1 이 된다.
# E 는 순위에서 빼고 못 박았다 -- robocasa 의 E 와 같은 이유로, 모든 태스크의
# 접근·복귀 구간에 들어 있어 풀을 가를 수 없다. 대신 아무것도 붙들지 않은
# 순간에는 잃을 것이 없다는 것이 확실하므로 가점 쪽의 큰 몫을 고정으로 준다.
# HEFT 는 독립 감점이 아니다. TURN 을 키우는 항이다 -- 독립으로 두면 큰 상자를
# 쥐고 옮길 때 FIRM(가점)과 상쇄되어 Bring Box 가 0점이 된다. 뒤집기는 그 자체로
# 절반의 위험이고, 두 손이 있어야 할 만큼 큰 것을 뒤집는 것이 온전한 위험이다.
# TURN 을 지우면서 HEFT 를 독립 감점으로 올렸다. TURN 은 237청크 중 99.6% 가
# 3등급인 상수였고, risk = TURN x (0.5 + 0.5 x HEFT) 에서 g=0.5 로 굳어 모든
# 청크에 risk 바닥 0.25 를 깔았다. 상한에 못 닿게 만든 것은 아니다 -- snap 이
# 구제한다 -- 하지만 conf 폭을 [0.25, 0.875] 로 잘라 분산을 직접 깎았다.
SIGN = {"TURN": -1, "HEFT": -1, "SHOVE": +1, "FIRM": +1, "FREE": +1}
# 무게는 활성 목록이 정해지면 각 변의 합이 1 이 되게 정규화한다. 하나씩
# 더해 가는 중이라 지금은 문항마다 같은 무게로 두고, 문항이 확정되면 덮는
# 태스크 수로 다시 잡는다.
WEIGHT = {"TURN": 1.0, "HEFT": 1.0, "SHOVE": 1.0, "FIRM": 1.0, "FREE": 1.0}
HEFT_SHARE = 0.5     # 뒤집기 단독이 지는 위험의 몫; 나머지는 HEFT 가 채운다


# Candidates that were dropped, and why -- the ranking is the method's step 3.
#
#   "is it being parked at one exact spot"      covers Bring PolyBag (위험) AND
#       Bring Box (안정): +1 -1 = 0. The place is not what separates them.
#   "is the object firm rather than floppy"     covers Bring Box and Pass (+2)
#       but also Rotate Box (-1), and it is B negated, which counts the same
#       evidence twice. Dropped on both counts.
#   "is the object being turned over"           covers Rotate Box, same as A,
#       but says nothing about WHY turning is fragile. A is the mechanism.
GUIDANCE = (
    "You are judging one instant of a two-armed robot with hands, to decide how far the "
    "next stretch of its motion could be thinned out -- how many of its commanded poses "
    "could be dropped, letting the arms travel further between the ones that remain, "
    "without changing the outcome.\n\n"
    "What decides it is how the object is being held right now. A hand closed around a "
    "box keeps it through a coarse swing. A box held only by two palms pressing inward "
    "does not: the hold IS the difference between the two hands, and thinning the poses "
    "changes that difference. Something that hangs and swings is its own case again -- "
    "there is no rigid body to keep, and the faster the arm moves the further it "
    "swings.\n\n"
    "Judge the moment in front of you, not the name of the job. One segment passes "
    "through several of these from one second to the next."
)

# D is pinned rather than ranked, the way robocasa's E is. Phases where nothing
# is held or touched sit inside every subtask, the damaged ones included, so a
# phase common to all of them can never separate the pools. With no object in
# hand there is no hold to lose, so it takes the safe ceiling by construction.
# t=4 -> t=5, 문항 A 재서술(셋째). A 는 세 판 연속 3등급에 붙박였고, 세 판 다
# 3등급을 "일부만/반쯤" 상태로 썼다 -- 같은 실수를 세 번 했다. 어느 프레임에서도
# 방어되는 등급을 만들면 모델은 거기로 간다.
#
# 그리고 t=4 는 문항끼리 간섭한다는 것을 보여줬다. A 만 고쳤는데 C 가 +2.45 에서
# +0.68 로 깨졌다. 넷을 한 프롬프트에서 같이 답하므로, A 가 봉투를 "통통한"
# 것으로 부르자 모델이 더는 그것을 C 의 "납작한 주름진 것" 으로 보지 않았다.
# **한 문항의 서술은 다른 문항의 답을 바꾼다.** 그래서 R3(한 바퀴에 한 축)은
# 한 문항만 고쳐도 넷을 다 다시 재라는 뜻이기도 하다.
#
# 이번에는 봉투의 모양이 아니라 손 모양으로 간다. 두 봉투 층이 실제로 갈리는
# 자리다: 뒤집는 쪽은 손가락이 봉투를 감아쥐고, 옮기는 쪽은 손가락이 펴진 채
# 옆면에 닿아 있다. D 가 상자에서 하는 구분(한 손이냐 두 손 사이냐)과 같은
# 종류이고, D 는 그것으로 +2.02 를 냈다.
#
# t=3 -> t=4, 문항 A 재서술. 검증 3 에서 A 만 남았다(-0.02). 두 봉투 층의
# 프레임을 실제로 뜯어보니 A 가 이름 붙인 국면 -- 봉투를 손가락으로 집어
# 판에서 들어 올린다 -- 이 여기 없다. 봉투는 들리지 않는다. 판 위에서 밀리고
# 판 위에서 잡힌다. 그러니 A 는 없는 장면을 물었고, 세 등급을 다 쓰면서도
# 어느 층에서나 같은 답이 나왔다.
#
# 두 봉투 층이 실제로 갈리는 자리는 하나뿐이다: **봉투가 손에 눌려
# 우그러졌는가.** 뒤집는 쪽은 손가락이 파고들어 모양이 무너져 있고, 옮기는
# 쪽은 통통한 모양을 그대로 지킨 채 손이 옆면에 붙어 있다. 정지 화면이 담는
# 차이다.
#
# t=2 -> t=3, 등급표 축 하나. 검증 3 에서 A(-0.08)와 D(+0.46)가 떨어졌고
# 원인이 같다: **1등급이 경쟁 배치를 담고 있지 않았다.** A 의 1등급은 "손
# 가까이 주름진 것이 없다" 였는데 turn+bag 장면에는 주름진 것이 손 밑에 있다.
# 그 층에서 1을 고를 수 없으니 3으로 올라오고, 봉투 층 둘이 똑같아진다.
# D 도 같은 이유로 turn+box 에서 켜졌고 그것이 B-D 상관 +0.62 다.
# 이제 각 문항의 1등급이 그 문항이 아닌 쪽의 배치를 이름으로 부른다 --
# A 의 1등급은 C 의 장면이고, D 의 1등급은 B 의 장면이다.
#
# WHAT THE EARLIER DRAFTS GOT WRONG, so the next one does not walk back in.
#
#   ASKED WHAT THE FACTS ALREADY SAY. A draft asked whether the object was
#     pinched between two palms; descriptors() computes that as `held`. 99% of
#     chunks came back at the middle grade. `wrist_rot`, `rot_asym`,
#     `one_handed`, `hand_change` are stated too.
#   ASKED WHAT A STILL CANNOT SHOW. The next draft asked whether the object was
#     "being taken somewhere" and "being turned over". The model sees one frame
#     per camera; motion is not in it. 91% answered the rung that reads "the
#     hands are on it and it has not moved yet", and the turn/move contrast came
#     out at -0.03 and -0.07 -- no separation at all. Every rung below is a
#     configuration that a single frame holds: what is on the plate, what is off
#     it, where the hands are, whether an edge is up.
#   ASKED SOMETHING ALWAYS TRUE. An earlier C asked whether open surface lay
#     ahead. On a sorting plate it always does: 95.4% answered 5.
#   LET THE MIDDLE RUNG SWALLOW THE SAFE CASE. "it mostly keeps its form and
#     only the touched face dents in" describes a cardboard box exactly, so the
#     firm case scored the same as the limp one (2.89 vs 3.00) and the one check
#     that had been working stopped working.
#   USED THE MIDDLE AS A HEDGE. Grades 2 and 4 went unused across 763 chunks.
#     Grade 2 is now the shape that worked in robocasa -- the subject IS on the
#     plate and the hands are on something else.
# 문항은 행동을 이름 붙이고, 정지 화면이 담는 배치는 등급표가 진다. 앞 판은
# 문항 문장까지 배치로 내려서("상자가 두 손 사이에 끼어 있다") 이 작업장
# 밖에서는 말이 안 되는 문장이 됐다. 모델이 스틸을 본다는 제약은 등급표가
# 감당할 짐이지 문항이 감당할 짐이 아니다.
# 행동을 이름 붙이되, 판정 가능하게 하는 배치를 한정어로 단다. robocasa 가
# 그 형식이다 -- "고정된 것을 누르거나 밀거나 돌리는가 -- 버튼, 다이얼, 서랍
# 앞판 -- 아무것도 안 쥔 채로". 행동이 앞에 오고, "고정된" 과 "아무것도 안 쥔
# 채로" 가 한 장에서 판정되게 하며, 예시가 물체를 일반화한다.
#
# 양 끝은 둘 다 재봤고 둘 다 나쁘다. 배치만 쓴 판(t=6)은 "상자가 두 손 사이에
# 끼어 있다" 가 되어 이 작업장 밖에서 말이 안 됐고, 행동만 쓴 판(t=8)은
# "돌리고 있는가" 가 정지 화면에서 답할 수 없어 A 가 +2.01 에서 +0.03 으로,
# C 가 +1.95 에서 +0.45 로 무너졌다. B 만 살아남았는데 "모양이 무너지는 것을
# 다루는가" 는 원래 한 장에서 보이는 성질이라서다.
# 단위 행동이다. 태스크를 가리지 않고 모든 청크에 같은 다섯을 묻는다.
#
# 위험 그룹(Rotate Box, Bring PolyBag)이 공유하는 것은 **손이 물체에 준 상태를
# 계속 유지해야 한다**는 것이다. 상자는 두 손이 서로 미는 힘의 균형으로만
# 붙들려 있고, 봉투는 쥔 자리가 형태를 잃어 쥠 자체가 계속 바뀐다.
#
# 안정 그룹(Pass, Bring Box, Rotate PolyBag)이 공유하는 것은 **접촉이 흐트러져도
# 결과가 남는다**는 것이다. 밀어 보낸 것은 손을 떼도 가고, 감싸 쥔 단단한 것은
# 팔이 흔들려도 손 안에서 안 움직이며, 무른 것을 뒤집는 데는 잃을 파지가 없다.
# robocasa phase9 와 같은 형식이다. 등급표는 하나를 공유하고, 그 하나가
# "이 문항이 이 순간을 얼마나 설명하는가" 의 진행도다.
#
# **기본 가점 문항 하나로 시작해서 하나씩 더한다.** 다섯을 한꺼번에 세워놓고
# 안 되는 것을 고치는 방식으로 열두 바퀴를 돌았고, 그때마다 무엇이 무엇을
# 망쳤는지 알 수 없었다 -- 넷을 한 프롬프트에서 답하므로 하나를 고치면 나머지
# 답도 바뀐다. 빈손으로 지나가는 구간은 잃을 것이 없다는 것이 확실하고
# (robocasa 의 E 와 같은 자리) 모든 태스크에 들어 있으므로, 거기서 시작한다.
#
# POOL 은 후보 전부이고 ACTIVE 가 지금 물을 것이다. ALLEX_CHECKS 로 바꾼다.
# 절차에서 나온 것이다. 각 태스크를 단위 행동으로 쪼개고, 덮는 태스크 수에서
# 반대 풀을 잘못 덮는 수를 뺐다. 위험 풀은 Rotate Box 와 Bring PolyBag.
#
#     빈손으로 지나간다        모든 태스크    풀을 못 가름 -> 못 박음 (가점)
#     두 손이 양쪽에서 함께     Rotate Box    +1 -0 = 1   그러나 held 로 계산됨
#     형태가 안 잡히는 것을 옮김 Bring PolyBag +1 -0 = 1   채택 (감점)
#     옆으로 밀어 보낸다        Pass          +1 -0 = 1   채택 (가점)
#     눌러 넘긴다              Rotate PolyBag +1 -0 = 1  채택 (가점)
#     단단한 것을 쥐고 옮긴다    Bring Box     +1 -0 = 1  채택 (가점)
#     물체를 세워 넘긴다        Rotate 둘     +1 -1 = 0   탈락
#     정확한 자리에 놓는다      Bring 둘      +1 -1 = 0   탈락
#
# 개수가 절차에서 나온다 -- 감점 1 + 가점 3 + 못 박음 1. robocasa 의 2+3 을
# 따라갈 이유가 없다. Rotate Box 의 위험은 문항이 아니라 계산 사실(`held`,
# `wrist_rot`)과 상한 1.5 가 담는다.
# 확정된 다섯. 후보 여덟에서 겹치는 셋을 뺐다.
#   FLOP  "봉투 살을 움켜쥐고 넘긴다" 는 TURN 의 부분집합이라 같은 사건을
#         반대 부호로 두 번 센다. 폴리백 뒤집기는 TURN 걸림 + HEFT 안 걸림
#         으로 이미 구별된다.
#   LOOSE "형태를 못 잡는가" 는 FIRM 의 부정이라 한 축을 두 번 세는 것이다.
#         Bring PolyBag 의 위험은 상한 2.0 이 이미 담는다.
#   OPEN  이 판은 대체로 비어 있어 "빈 판으로 가는가" 는 항상 참이 될 자리다
#         (robocasa 의 C 가 95% 5등급을 받은 그 자리). SHOVE 와도 겹친다.
POOL = {
 "TURN":  "Is the robot working the thing round so a DIFFERENT SIDE OF IT comes up --\n"
          "   flipping it, tipping it, rolling it over?",
 "HEFT":  "Is the thing being handled TOO MUCH FOR ONE HAND -- big, heavy or stiff\n"
          "   enough that both hands have to be on it to manage it?",
 "SHOVE": "Is the robot SENDING the thing across the surface -- shoving or sliding it\n"
          "   away -- rather than lifting it?",
 # 뒷절 "gripped so that it does not shift about in the hand" 을 뺐다. 그 절이
 # 물체 문항을 파지 문항으로 바꿔놓았고, 가장 단단해 보이는 파지는 두 손 클램프
 # -- 즉 위험 태스크다. 그래서 FIRM 이 move+box 1.44 / turn+box ~2.5 로 갈려
 # 부호가 -0.64 로 뒤집혔다. 물체 강성을 읽는다면 두 상자 층이 같아야 한다.
 # 게다가 파지 정도는 이미 "256px 두 장에서 안 보인다" 로 폐기된 축이고, 그것이
 # 뒷절로 몰래 들어와 있었다. 역대 최고 점수(+2.87)를 낸 형태로 되돌린다.
 "FIRM":  "Is the thing under the hands KEEPING ITS SHAPE -- straight edges and flat\n"
          "   faces, no dent, fold or sag where they press on it -- rather than giving\n"
          "   way wherever it is touched?",
 "FREE":  "Are the hands EMPTY and moving through clear space -- nothing held, nothing\n"
          "   being touched?",
}
ACTIVE = tuple(os.environ.get("ALLEX_CHECKS", "FREE").split(","))
_CHECKS = tuple((q, POOL[q]) for q in ACTIVE)

# 문항마다 다른 눈금이 아니라 하나를 공유한다. 이 눈금이 재는 것은 그 문항이
# 이 순간을 얼마나 설명하느냐이지 그 상태의 강도가 아니다.
_LADDER3 = (
    "3 = it is happening right now -- the picture shows what the check describes",
    "2 = it is about to happen, or the thing it is about is there and the arms are\n"
    "    on something else",
    "1 = the check does not describe this moment",
)
_LADDER5 = (
    "5 = it is happening right now -- the picture shows what the check describes",
    "4 = not yet, but the hands are right up against it, one motion away",
    "3 = the hands are heading for it and still some way off",
    "2 = the thing the check is about is there, but the arms are busy with\n"
    "    something else",
    "1 = there is nothing in this picture the check could be about",
)
_LADDER = _LADDER3 if NGRADE == 3 else _LADDER5

# 물어보는 순서대로 A) B) C) ... 로 다시 붙인다. 모델은 자리로 답한다.
LETTERS = tuple(chr(ord("A") + i) for i in range(len(_CHECKS)))
_AXES = "".join(f"{L}) {t}\n" for L, (_, t) in zip(LETTERS, _CHECKS))

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat them. "
       "Answer each check from what the cameras show about the MOMENT in front of you, "
       "read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else -- one "
       f"digit from 1 to {NGRADE} per check, rating how far that check describes this moment:\n"
       + "\n".join("  " + l for l in _LADDER)
       + "\nA grade refers only to the check on that line.\n" + _AXES + "Answer:")


# 임계값은 층별 중앙값에서 잡았다. 이 값들이 두 봉투 층을 가르는 자리다:
# gap_mean 은 뒤집기 0.41~0.44 / 옮기기 0.51 (AUC 0.827), arm_speed 는
# 뒤집기 0.045 / 옮기기 0.027 (0.716), hand_change 는 0.022 / 0.013 (0.638).
GAP_NEAR, GAP_FAR = 0.45, 0.52
SPEED_SLOW, SPEED_FAST = 0.020, 0.040
FING_STILL, FING_WORK = 0.010, 0.020
ROT_LITTLE, ROT_LOT = 15.0, 25.0


def facts_v3(x):
    """계산값을 사실 문장으로. 임계 판정까지 끝내고 결론만 준다.

    robocasa 의 facts_text 와 같은 형식이다. 앞 판은 날숫자를 그대로 넘겼고
    ("the palms are 0.44 m apart"), 그러면 해석이 모델 몫이 된다. gap_mean 이
    봉투를 옮기는 구간과 뒤집는 구간을 AUC 0.827 로 가르는데도 모델이 그것을
    쓰지 않은 이유가 이것이다 -- 가르는 것은 숫자가 아니라 그 숫자에 대한
    판정이고, 판정은 코드가 해야 한다.

    여섯 줄이 조건과 무관하게 항상 나온다. 조건부로 줄이 붙었다 빠졌다 하면
    프롬프트 길이가 프레임마다 달라지고, 배치에서 패딩이 필요해진다.

    사람이 붙인 서브태스크 이름은 들어가지 않는다. 주석은 "Bring Object" 라고만
    하고 어느 물체인지는 안 쓰는데 배속이 정확히 거기서 갈린다.
    """
    p = []
    p.append("only one arm is moving" if x.get("one_handed") else "both arms are moving")
    g = x["gap_mean"]
    p.append("the palms are close in to each other" if g < GAP_NEAR else
             "the palms are a middling distance apart" if g < GAP_FAR else
             "the palms are well apart")
    p.append("and drawing together" if x["closing"] else
             "and moving apart" if x["opening"] else "and holding that distance")
    v = x["arm_speed"]
    p.append("the arms are barely moving" if v < SPEED_SLOW else
             "the arms are moving at a normal pace" if v < SPEED_FAST else
             "the arms are moving fast")
    h = x["hand_change"]
    p.append("the fingers are still" if h < FING_STILL else
             "the fingers are shifting a little" if h < FING_WORK else
             "the fingers are working")
    r = x["wrist_rot"]
    p.append("the wrists barely turn" if r < ROT_LITTLE else
             "the wrists turn a fair amount" if r < ROT_LOT else
             "the wrists turn a great deal")
    return ("MEASURED FROM THE PLANNED MOTION over the next ~1 second (these are computed "
            "facts, not estimates): " + "; ".join(p) + ". Skipping to every 2nd target "
            f"would demand {x['merge_demand_k2']:.3f} rad in one step, every 3rd "
            f"{x['merge_demand_k3']:.3f} rad (this robot never exceeded {MERGE_LIMIT_V2} "
            "rad in the demonstrations).")


def confidence(picks):
    """다섯 답에서 나오는 확신. 0 이면 하한, 1 이면 상한.

    HEFT 는 독립 항이 아니라 TURN 의 배수다. 독립으로 두면 큰 상자를 쥐고
    옮길 때 HEFT(감점)와 FIRM(가점)이 같이 걸려 0 이 되고, 안정 풀 태스크가
    점수를 못 받는다. 뒤집기는 그 자체로 절반의 위험이고, 두 손이 있어야 할
    만큼 큰 것을 뒤집는 것이 온전한 위험이다.

    가점 쪽은 걸린 것들의 가중평균이다. 합으로 하면 한 문항만 걸렸을 때 그
    문항의 무게가 그대로 천장이 되어, 명백히 안전한 순간조차 상한에 닿지
    못한다.
    """
    g = {q: (float(p) - 1.0) / (NGRADE - 1) for q, p in zip(ACTIVE, picks)
         if p is not None}
    g = {q: w for q, w in g.items() if w > 0}
    def side(sign):
        w = {q: v for q, v in g.items() if SIGN.get(q) == sign}
        if not w:
            return 0.0
        return sum(WEIGHT[q] * v for q, v in w.items()) / sum(WEIGHT[q] for q in w)

    if "TURN" in ACTIVE:
        # 짝으로 쓸 때만. HEFT 가 TURN 을 키운다.
        risk = g.get("TURN", 0.0) * (HEFT_SHARE + (1 - HEFT_SHARE) * g.get("HEFT", 0.0))
    else:
        risk = side(-1)
    safe = side(+1)
    return float(min(1.0, max(0.0, (1.0 + safe - risk) / 2.0)))


def ratio_for(picks, cell=None):
    """태스크가 준 [하한, 상한] 안에서, 확신이 정한 자리.

    상한은 그 구간을 통째로 그 배속에 돌려 성공한 값이고, 성공률은 처음부터
    끝까지 압축한 결과라 그 안의 어느 단위 행동이 위험했는지는 말해주지
    않는다. 그래서 상한은 태스크가 주고, 그 안에서 얼마나 쓸지는 단위 행동을
    묻는 다섯 문항이 정한다.
    """
    lo, hi = TASK_RANGE.get(cell, DEFAULT_RANGE)
    return float(lo + confidence(picks) * (hi - lo))


def snap(k, candidates=CANDIDATES):
    """Nearest ratio this robot can actually be replayed at.

    A tie goes DOWN. 1.75 sits exactly between two candidates and either would
    round it; the lower one compresses less, and being one step under a phase's
    real tolerance costs nothing while being one step over loses the episode.
    """
    return min(candidates, key=lambda c: (abs(c - k), c))
