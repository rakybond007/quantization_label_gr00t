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
CEILING = {"A": 2.0, "B": 1.5, "C": 2.5, "D": 3.0}
COVER = {"A": 1, "B": 1, "C": 1, "D": 2}      # step 5, recorded; the ratio carries
NGRADE = 5

# There is no 4x here. The candidate ratios for this robot are these five, and
# a label that is not one of them cannot be replayed.
CANDIDATES = (1.0, 1.5, 2.0, 2.5, 3.0)

BASE = float(os.environ.get("ALLEX_BASE_K", 2.5))

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
_CHECKS = (
 ("A", "Is something WRINKLED AND LIMP -- a mailer, a sack -- PICKED UP by the fingers,\n"
       "   lifted off the plate rather than left lying on it?",
  ("fingers have it gathered up and it sags below them, clear of the plate",
   "fingers have a fold of it and that part is up off the plate",
   "fingers are pinching an edge of it, all of it still down",
   "something wrinkled is on the plate and the hands are on something else",
   "it lies flat on the plate with a hand resting on top of it")),
 ("B", "Is a SQUARE-EDGED BOX caught BETWEEN TWO HANDS, one on each of two opposite faces?",
  ("it is up on an edge between the two hands, a face that was down now showing",
   "it is between the two hands with one side lifted clear of the plate",
   "a hand is flat on each of two opposite faces, the box still down",
   "such a box is on the plate and only one hand is at it",
   "nothing is caught between two hands")),
 ("C", "Is a FLAT WRINKLED SHEET on the plate being worked by a hand ON TOP OF IT, with the\n"
       "   sheet still lying there?",
  ("it is folding over under the hand and its underside is coming into view",
   "an edge of it is up under the hand, the rest flat on the plate",
   "the hand is down on it and it lies flat",
   "such a sheet is on the plate and the hands are on something else",
   "there is no flat wrinkled thing under a hand")),
 ("D", "Is a SQUARE-EDGED BOX held by ONE HAND ONLY, with nothing on its far side?",
  ("one hand is on it and the far side of it is open, no second hand there",
   "one hand is on it and the other hand is away doing something else",
   "one hand is near it, not yet touching, and no second hand is coming",
   "such a box is on the plate and the hands are on something else",
   "it is caught between two hands, one on each of two opposite faces")),
)

_AXES = "".join(
    f"{q}) {text}\n" + "".join(f"   {5-k} = {a}\n" for k, a in enumerate(anchors))
    for q, text, anchors in _CHECKS)

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat them. "
       "Answer each check from what the cameras show about the MOMENT in front of you, "
       "read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else -- one "
       "digit per check, using that check's own grades:\n\n" + _AXES + "Answer:")


def facts_v3(x):
    """The measured facts, WITHOUT the human annotation.

    v2 opened with 'The human annotation for this segment is "Bring Object"'.
    That line cannot carry the ceiling here even if we wanted it to: the
    annotation says Object, never which object, and the ratio for Bring splits
    3.0 / 2.0 on exactly that. Which one it is has to be seen, which is check A.

    With the pair read off the checks, the annotation has no work left to do,
    so it is not sent -- a label the model can lean on is a label it will lean
    on, and Rotate would pull every mailer toward the box's ratio.
    """
    hands = "one arm is moving" if x.get("one_handed") else "both arms are moving"
    stop = ("the motion is running down toward a stop" if x.get("slowing")
            else "the motion is not slowing down")
    gap = ("closing" if x["closing"] else "separating" if x["opening"] else "steady")
    return (f"MEASURED (computed, not estimates): {hands}; the palms are "
            f"{x['gap_mean']:.2f} m apart and {gap}; the arms move "
            f"{x['arm_speed']:.3f} rad per step and {stop}; the grasp centre "
            f"travels {x['translation']*100:.0f} cm; the wrists turn "
            f"{x['wrist_rot']:.0f} deg. Skipping to every 2nd target would demand "
            f"{x['merge_demand_k2']:.3f} rad in one step, every 3rd "
            f"{x['merge_demand_k3']:.3f} rad (this robot never exceeded "
            f"{MERGE_LIMIT_V2} rad in the demonstrations).")


def ceiling_from_checks(picks, levels=CEILING):
    """The ratio this moment tolerates, in units of K.

    Each check names one phase and carries the ratio that phase was measured
    at, so the grade-weighted mean of the ratios is already in units of K. The
    phases do not overlap -- a limp thing lifted clear of the plate is not a
    limp thing lying on it, a box between two hands is not a box standing alone
    with one hand on it -- so a mean is the right way to blend the moments that
    are two things at once, and no clamping is needed.

    An earlier design split this into "what the object is" and "what is being
    done" and could not put them back together: a mailer being carried and a
    mailer being flipped got the same object answer, and whatever ratio that
    answer carried was wrong for one of them. The phase is the unit.

    Nothing answered means no phase was recognised, which takes the base -- 2.5,
    the ratio for a station where nothing is being handled.
    """
    g = {q: (float(p) - 1.0) / 4.0 for q, p in zip("ABCD", picks) if p is not None}
    g = {q: w for q, w in g.items() if w > 0}
    if not g:
        return BASE
    aim = sum(w * levels[q] for q, w in g.items()) / sum(g.values())
    return float(BASE + max(g.values()) * (aim - BASE))


def snap(k, candidates=CANDIDATES):
    """Nearest ratio this robot can actually be replayed at.

    A tie goes DOWN. 1.75 sits exactly between two candidates and either would
    round it; the lower one compresses less, and being one step under a phase's
    real tolerance costs nothing while being one step over loses the episode.
    """
    return min(candidates, key=lambda c: (abs(c - k), c))
