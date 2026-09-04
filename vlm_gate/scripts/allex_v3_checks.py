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
# The object does NOT push one way. A mailer is worse to CARRY than a box --
# it swings and slips, 28/30 down to 23/30 -- and better to TURN than a box --
# there is no two-hand hold to lose, so flipping it fast is fine. Firmness
# reverses sign depending on what is being done to the thing.
#
# A ratio hung on each check cannot say that. Whatever number "it goes out of
# shape" carries, it lands on carrying and turning alike, and every earlier
# draft that tried it either sent flipped mailers down to the box's 1.5 or
# lifted carried mailers up to the box's 3.0. So the ratio is read off the
# PAIR: what is being done, and what it is being done to.
#
#                     firm        goes out of shape
#     taken somewhere  3.0              2.0
#     turned over      1.5              2.5
#     neither                2.5
#
# Three of the four are replayed (27/30, 23/30, 16/30 at the ratios above and
# 22/30 for the box at 2x); turning a mailer is the operator's, not replayed.
K_TABLE = {("move", "firm"): 3.0, ("move", "soft"): 2.0,
           ("turn", "firm"): 1.5, ("turn", "soft"): 2.5}
NGRADE = 5

# There is no 4x here. The candidate ratios for this robot are these five, and a
# label that is not one of them cannot be replayed.
CANDIDATES = (1.0, 1.5, 2.0, 2.5, 3.0)

# What "3.0 as a weak allowance" means arithmetically. A plain weighted mean
# gives a check its full ceiling however faintly it fired: check C answered 3
# and nothing else answered lands on 3.0 exactly as C answered 5 would, because
# normalising divides the weight straight back out. So the mean sets the
# DIRECTION and the strength of the evidence sets how far along it we go, from a
# base of 2.0 -- the ratio this robot is simply run at. Faint evidence stays near
# 2.0 in either direction; only an unambiguous scene reaches 3.0 or 1.5.
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
# WHAT THE FIRST DRAFT GOT WRONG, so the next one does not walk back into it.
#
#   ASKED WHAT THE FACTS ALREADY SAY. The old A asked whether the object was
#     pinched between two palms. descriptors() computes exactly that -- `held`
#     is gap < 0.42 m with both arms moving -- and states it. Asked something it
#     had already been told, the model answered the middle grade on 99% of
#     chunks. `wrist_rot`, `rot_asym`, `one_handed`, `hand_change` are stated
#     too; nothing here may ask for them again.
#   ASKED SOMETHING ALWAYS TRUE. The old C asked whether there was open surface
#     ahead with no slot at the far end. On a sorting plate there always is:
#     95.4% answered 5.
#   DESCRIBED A SCENE INSTEAD OF NAMING A BEHAVIOUR. Its replacement was worse
#     in a quieter way -- "bearing on it from one side, with the object's weight
#     still on the plate" pins down where the hand is, which face it touches and
#     what the object rests on. robocasa's checks name an action and then give
#     instances of it ("pressing, pushing or turning something FIXED IN PLACE --
#     a button, a dial, a drawer front"), and that is what carries: the action
#     is the question, the instances only show what it looks like here.
#   NAMED SCENES THAT DO NOT OCCUR HERE. The old B topped out at "it hangs from
#     the hand and its lower half droops". Nothing hangs at this station: things
#     lie on the plate and are pressed and turned there. The model could not
#     reach grade 5 and stopped at 3, so the one check that worked was still
#     using half its range.
#   USED THE MIDDLE AS A HEDGE. "closing onto it", "half wrapped" are
#     defensible in any frame. Grades 2 and 4 were never used once across 763
#     chunks, so the ladders were really three-valued and mostly the middle one.
#
# Hence: grade 2 on every ladder is now the shape that worked in robocasa --
# the subject IS in the picture and the hands are on something else -- which is
# a thing the eye can check, unlike "somewhat".
_CHECKS = (
 ("A", "Does what is being handled GO OUT OF SHAPE under the hand -- a bag, a sack, cloth,\n"
       "   anything that creases or sags -- rather than keeping its form?",
  ("the hand marks it as it works, and the shape stays changed",
   "it gives under the hand but springs back to roughly its form",
   "it mostly keeps its form and only the touched face dents in",
   "such a thing is in the picture and the hands are on something else",
   "what the hands have keeps its shape and its edges")),
 ("B", "Is the object being TURNED so that a different side faces up -- flipped, tipped,\n"
       "   rolled -- rather than left the way it lies?",
  ("it is up off its resting side and a new one is coming to the top",
   "it has begun to tip and one side is lifting clear",
   "the hands are set to turn it and it has not gone over yet",
   "something that would be turned is there and the hands are elsewhere",
   "the same side stays up throughout")),
 ("C", "Is the object being TAKEN SOMEWHERE ELSE -- carried, slid, pushed across -- with the\n"
       "   same side still facing up?",
  ("it is on its way, travelling with the hands",
   "the hands have it and it has just begun to go",
   "the hands are on it and it has not moved yet",
   "something to be moved is there and the hands are elsewhere",
   "nothing is going anywhere")),
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


def ceiling_from_checks(picks, table=K_TABLE):
    """The ratio this moment tolerates, in units of K.

    Two axes, read off three checks. B and C say what is being done -- turned,
    or taken somewhere -- and A says what it is being done to. The pair picks a
    cell of K_TABLE; the grades interpolate between cells rather than snapping
    to one, because a moment is rarely purely one thing.

    The base is what a moment nothing recognises is worth: 2.5, the ratio for a
    station where nothing is being handled. Evidence moves K away from it only
    as far as the strongest action check is sure, which is what "3.0 as a weak
    allowance" means -- a faint C does not buy the full 3.0.

    Nothing answered stays at the base. A did not fire either, so we do not
    know what the object is; there is no cell to read.
    """
    g = {q: (float(p) - 1.0) / 4.0 for q, p in zip("ABC", picks) if p is not None}
    soft = g.get("A", 0.0)                       # 0 = firm, 1 = goes out of shape
    turn, move = g.get("B", 0.0), g.get("C", 0.0)
    act = turn + move
    if act <= 0:
        return BASE
    k_turn = (1 - soft) * table[("turn", "firm")] + soft * table[("turn", "soft")]
    k_move = (1 - soft) * table[("move", "firm")] + soft * table[("move", "soft")]
    aim = (turn * k_turn + move * k_move) / act
    return float(BASE + max(turn, move) * (aim - BASE))


def snap(k, candidates=CANDIDATES):
    """Nearest ratio this robot can actually be replayed at.

    A tie goes DOWN. 1.75 sits exactly between two candidates and either would
    round it; the lower one compresses less, and being one step under a phase's
    real tolerance costs nothing while being one step over loses the episode.
    """
    return min(candidates, key=lambda c: (abs(c - k), c))
