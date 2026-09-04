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

WHAT IS STILL MISSING. The grid is only {2, 2.5}, so every ceiling below is one
of two values and the spread of this whole file is 0.5. Three points would fix
it, and until they exist the ratios here should be read as an ordering:
  - Pass and Bring Box at 3x and 4x -- their true ceiling is above the grid,
    and v2's prior for Pass was 3.0 with nothing to back it.
  - Rotate Box at 1.5x -- it is already at 22/30 at 2x, and without the
    uncompressed rate we cannot say 2.0 is safe rather than merely the least
    bad measured point.
  - Rotate PolyBag at anything. It was never replayed, so it has no ceiling
    here and no check claims it.
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
CEILING = {"A": 2.0, "B": 2.0, "C": 2.5, "D": 2.5, "E": 2.5}
COVER = {"A": 1, "B": 1, "C": 1, "D": None, "E": 1}   # D is pinned, not ranked
NGRADE = 5

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
_CHECKS = (
 ("A", "Is a firm object held ONLY between two palms pressing inward on it, so that the\n"
       "   hold is the difference between the two hands?",
  ("both hands are pressed on it and it is turning between them now",
   "both hands are pressed on it and have not started to turn it",
   "both hands are on it but are still closing onto it",
   "both hands are approaching it from either side, not yet touching",
   "the object is not held between two palms")),
 ("B", "Is the thing being carried something that HANGS AND CHANGES SHAPE -- a plastic bag,\n"
       "   a sack, cloth -- rather than holding its own form?",
  ("it is held up and visibly sagging and swinging",
   "it is held up and hanging, not swinging at the moment",
   "it is limp but still resting on a surface, being taken up",
   "it is soft-looking but the hand is not on it yet",
   "what is being handled keeps its own shape")),
 ("C", "Is the robot sliding or passing an object ACROSS to the other side, where it would\n"
       "   still be fine a few centimetres off?",
  ("the object is travelling sideways across the workspace now",
   "the hand has it and the sideways push is beginning",
   "the hand is on it, the direction it will go is sideways",
   "the object sits where such a push would start, hand not on it",
   "nothing is being sent across")),
 ("D", "Are BOTH hands clear of every object -- nothing held, nothing touched?",
  ("both hands are in open space, well away from anything",
   "both hands are empty and moving away from what was just handled",
   "both hands are empty and moving toward something, still short of it",
   "one hand is empty, the other is on or near an object",
   "an object is held or touched")),
 ("E", "Is the object CLOSED INSIDE a hand -- fingers wrapped around it -- so that shaking\n"
       "   the arm would not move it within the grasp?",
  ("the fingers are wrapped around it and it is being carried",
   "the fingers are wrapped around it and it is not moving yet",
   "the fingers are closing around it",
   "the hand is open at it, about to close",
   "nothing is enclosed in a hand")),
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
    """Grade-weighted mean of the ceilings the answered checks carry.

    Returns a ratio in units of K, not a score. A moment that is two things at
    once lands between their ceilings, which is why the checks are graded rather
    than picked.

    Nothing answered means no check recognised the moment. That is the hands-
    empty case, so it takes D's ceiling rather than a middling default: there is
    no hold to lose in it either.
    """
    g = {q: (float(p) - 1.0) / 4.0 for q, p in zip("ABCDE", picks) if p is not None}
    tot = sum(g.values())
    if tot <= 0:
        return float(os.environ.get("ALLEX_IDLE_CEILING", levels["D"]))
    return float(sum(v * levels[q] for q, v in g.items()) / tot)
