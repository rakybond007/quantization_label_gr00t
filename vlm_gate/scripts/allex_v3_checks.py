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
CEILING = {"A": 2.0, "B": 1.5, "C": 3.0}
COVER = {"A": 1, "B": 1, "C": 2}
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


def ceiling_from_checks(picks, levels=CEILING):
    """The ratio this moment tolerates, in units of K.

    A plain weighted mean cannot express a ceiling. A mailer being pushed along
    the plate answers both A (it is limp) and C (it is being pushed); averaging
    2.0 with 3.0 puts the mailer ABOVE the 2.0 it was measured at, because C's
    ceiling was read off a box. A ceiling is not an average of opinions -- the
    most restrictive evidence has the last word.

    So the two sides are not symmetric:

      the permissive check (C) sets how far ABOVE the base ratio this moment
      could go, and only as far as it is sure. Faint evidence stays near the
      base -- that is "3.0 as a weak allowance".

      the restrictive checks (A, B) then clamp what is left, least restrictive
      first, each in proportion to how sure it is. Nothing they do can be undone
      by a permissive check that fired earlier.

    With nothing answered at all, no check recognised the moment. That is the
    base -- 2.5, the ratio for a station where nothing is being handled.
    """
    g = {q: (float(p) - 1.0) / 4.0 for q, p in zip("ABC", picks) if p is not None}
    up = {q: w for q, w in g.items() if q == "C" and w > 0}
    if up:
        aim = sum(w * levels[q] for q, w in up.items()) / sum(up.values())
        k = BASE + max(up.values()) * (aim - BASE)
    else:
        k = BASE
    for q in sorted(("A", "B"), key=lambda q: -levels[q]):   # least restrictive first
        k -= g.get(q, 0.0) * max(0.0, k - levels[q])
    return float(k)


def snap(k, candidates=CANDIDATES):
    """Nearest ratio this robot can actually be replayed at.

    A tie goes DOWN. 1.75 sits exactly between two candidates and either would
    round it; the lower one compresses less, and being one step under a phase's
    real tolerance costs nothing while being one step over loses the episode.
    """
    return min(candidates, key=lambda c: (abs(c - k), c))
