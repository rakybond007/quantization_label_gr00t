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
CEILING = {"A": 1.5, "B": 2.0, "C": 3.0, "D": 2.5, "E": 3.0}
COVER = {"A": 1, "B": 1, "C": 1, "D": None, "E": 1}   # D is pinned, not ranked
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
BASE = float(os.environ.get("ALLEX_BASE_K", 2.0))

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
# The first draft of these ladders broke four of the method's rules, and each
# break is a specific way the label goes wrong -- they are written down so the
# next draft does not walk back into them.
#
#   GRADED THE PROGRESS, NOT THE THING ASKED. A asked whether the hold is two
#     palms, then graded 5 vs 4 on whether the box had started TURNING. The
#     ladder measured a different quantity from the question. Now every level is
#     one quantity: how much of the object's weight the press is carrying.
#   PUT MOTION IN THE LADDER. "swinging", "being carried", "the push is
#     beginning" are all read off the action, and the facts already state arm
#     speed, travel and whether the motion is running down. Asking for them
#     again gets a worse answer to a question already settled in numbers.
#   LET A LEVEL FIRE WITH THE HAND OFF THE OBJECT. "it is soft-looking but the
#     hand is not on it yet", "the object sits where such a push would start" --
#     a bag lying in the background then dragged a firm-box chunk's ratio down,
#     and open counter beside a careful placement dragged one up. Every level
#     above 1 now requires the hand to be on the thing.
#   ASKED FOR THE CONCLUSION. C said "where it would still be fine a few
#     centimetres off". That IS the ceiling; asking the model for it invites the
#     answer it thinks is wanted. C now asks what is at the far end -- open
#     surface, or a slot to fit into -- which is visible.
_CHECKS = (
 ("A", "Is the object's weight resting on TWO PALMS PRESSED INWARD on it, rather than on\n"
       "   fingers or on a surface?",
  ("it is off the surface, held between the two palms and nothing else",
   "the palms have it and one edge is just lifting off the surface",
   "the palms are on either side and closing onto it, it still rests on the surface",
   "one palm is flat on it, the other is on the far side and not yet flat",
   "nothing is pressed between two palms")),
 ("B", "Does the thing IN THE HAND give way under its own weight -- sagging, folding, its\n"
       "   grasped spot pinched in?",
  ("it hangs from the hand and its lower half droops well below the grip",
   "it hangs from the hand and its outline bends away from straight",
   "the hand has closed on it and the grasped spot is visibly dented in",
   "the hand is on it and its surface gives where the fingers press",
   "what the hand has keeps its own outline")),
 ("C", "Is the object being taken ACROSS OPEN SURFACE, with no slot, shelf or container\n"
       "   waiting at the far end?",
  ("it is out over open surface with clear space all the way to the far side",
   "it has left its starting place and the space ahead of it is open",
   "it is held and the body has turned to face the open side",
   "it is held and open surface lies to one side of it",
   "there is no open run for it, or nothing is held")),
 ("D", "Are BOTH hands empty -- holding nothing and touching nothing?",
  ("both are empty and stand in open space with nothing within reach",
   "both are empty and the nearest object is behind them",
   "both are empty and an object is near but no hand is on it",
   "both are empty and one rests on a surface",
   "a hand holds or touches something")),
 ("E", "How far do the FINGERS WRAP AROUND the object -- is it closed inside the hand, or\n"
       "   lying on it?",
  ("the fingers meet around it and it is enclosed in the hand",
   "the fingers wrap most of the way round and part of it stands out",
   "the fingers are round one side of it and the palm carries the rest",
   "it lies on the palm and the fingers only lean on it",
   "nothing is inside a hand")),
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

    A plain weighted mean cannot express a ceiling. A bag closed inside a hand
    answers both B (it gives way) and E (the fingers are round it); averaging
    2.0 with 3.0 gave 2.33 and the bag came out ABOVE the 2.0 it was measured
    at, because E's ceiling was read off a box. A ceiling is not an average of
    opinions -- the most restrictive evidence has the last word.

    So the two sides are not symmetric:

      the permissive checks (C, D, E) set how far ABOVE the base ratio this
      moment could go, and only as far as the strongest of them is sure. Faint
      evidence stays near the base -- that is "3.0 as a weak allowance".

      the restrictive checks (A, B) then clamp what is left, least restrictive
      first, each in proportion to how sure it is. Nothing they do can be undone
      by a permissive check that fired earlier.

    With nothing answered at all, no check recognised the moment, so it takes
    the base ratio rather than any check's -- D is what says the hands are
    empty, and if D did not fire we do not get to assume it.
    """
    g = {q: (float(p) - 1.0) / 4.0 for q, p in zip("ABCDE", picks) if p is not None}
    up = {q: w for q, w in g.items() if q in ("C", "D", "E") and w > 0}
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
