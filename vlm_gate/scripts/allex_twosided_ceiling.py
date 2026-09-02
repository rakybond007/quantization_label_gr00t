"""Stage 2 as five one-thing questions, weighted by the ceilings you set by hand.

The replay recordings carry no subtask labels -- tasks.jsonl is a single
{"task": ""} -- so the shipped stage 2, which looks a ceiling up by subtask name
and then clamps it with four checks, has nothing to look up. The first
replacement asked one four-way question whose levels were arriving / careful
placing / soft object / travelling. Those are four different things, not four
strengths of one thing, so the "grades" were categories wearing an ordinal
costume: over 287 chunks levels 1 and 2 were never chosen once, on a recording
that plainly contains boxes being gripped and turned.

Here the four subtasks and their hand-set ceilings supply the evidence instead,
the way measured K2 damage does for robocasa. A low ceiling is a judgement that
the phase is dangerous to compress; a high one that it is safe. So each question
names ONE phase, and its weight is how far that phase's ceiling sits from the
scale's end:

    Bring Object    2.0   risk       weight (3 - 2.0) = 1.0
    Rotate Box      2.0   risk       weight (3 - 2.0) = 1.0
    Rotate PolyBag  2.5   stable     weight (2.5 - 1) = 1.5
    Pass Object     3.0   stable     weight (3.0 - 1) = 2.0

Every number here is yours. A moment that is none of the four -- arms lowered or
between parcels -- answers 1 across the board and takes 3.0, since there is no
pose, grip or placement to preserve; that is the only value not read off the
spec, and it is the top of the scale rather than a middling default.

The ceiling each question carries IS its weight, and the answer is the grade-
weighted mean of them. Taking the mean over the evidence actually present, rather
than a stability-minus-risk difference over the full weight sum, is what lets a
single confident answer reach its own subtask's number: "this is Pass Object and
nothing else" returns 3.0, not 2.36. The difference form pinned a lone answer at
its share of the total weight, so the safest phases never got near the top of the
scale you wrote.

B names a FIRM box on purpose. Written object-agnostically as "turning an object
they hold" it fired on bags being flipped as well, and correlated 0.58 with C --
but the spec separates Rotate Box (2.0) from Rotate PolyBag (2.5) on the object
alone, so the object is the whole distinction and the check has to carry it.

A points at the measured slowing. Asked as "putting it down where it has to line
up" it sat at grade 3 on 89% of chunks, hedging, while the computed `slowing`
flag was true on 7% -- the model was not reading the one fact that settles it.

Sign is not lost by averaging: the low-ceiling phases pull the mean down and the
high-ceiling ones pull it up, which is the same subtract-and-add structure with
its scale fixed to the spec.
"""
import os

# One check per subtask, carrying the ceiling you set for it. The fifth check,
# "are the hands empty", was dropped: it measures the same thing as the
# instruction to answer 1 everywhere when nothing is being handled, and the
# duplication showed as a -0.94 correlation against A, which it is the mirror
# of -- empty hands cannot also be placing something.
CEILING = {"A": 2.0, "B": 2.0, "C": 2.5, "D": 3.0}
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1}
_w = {q: (3.0 - c) if SIGN[q] < 0 else (c - 1.0) for q, c in CEILING.items()}
_r = sum(v for q, v in _w.items() if SIGN[q] < 0)
_s = sum(v for q, v in _w.items() if SIGN[q] > 0)
WEIGHT = {q: v / (_r if SIGN[q] < 0 else _s) for q, v in _w.items()}

GUIDANCE = (
    "You are judging one instant of a two-armed robot working a parcel line, to decide "
    "how far the next second of its motion could be thinned out -- how many of its "
    "commanded poses could be dropped, letting the arms travel further between the ones "
    "that remain, without changing what happens.\n\n"
    "Thinning is harmless while a hand is going somewhere and the exact path does not "
    "matter, or while what is handled is soft and has no exact pose to hold. It stops "
    "being harmless when something firm has to be set down in a particular place, or "
    "turned to a particular angle between two palms.\n\n"
    "You are shown ONE instant from two cameras, not a stretch of time. Judge what the "
    "picture shows."
)

# The ladder is anchored to what the HANDS are doing, not to what is in view.
# The first version graded by the presence of the thing a check is about, which
# works where targets come and go -- robocasa's buttons and drawers are off
# screen most of the time -- but not on a parcel line, where boxes are in every
# frame. Measured over 115 chunks that pinned A at grade 3 on 92% of them and
# never let B past 2: the model could not answer 1 ("nothing here this is
# about") because a box was always there, so it never answered 5 either.
# Level 3 must not be a hedge. Anchoring it to "on the way to doing it, not
# arrived" made it true of a robot that is always moving toward something, and
# the model took it every time rather than commit to 4 or 5: over 115 chunks A
# sat at 3 on all of them and no check ever reached 5 except C. Here 3 is a real
# situation -- two phases overlapping -- and the "about to / just finished"
# reading, which is the genuinely marginal one, sits at 2 where it belongs.
# Each check gets its own ladder. One shared wording had to describe four
# different situations at once, so its middle rungs fitted some and not others:
# "the two are going on at once" is a real state for carrying-versus-placing and
# nonsense for is-this-bag-soft. Written per check, every rung names something
# that can be seen.
#
# Nothing here points at the measurements. Stage 2 judges the scene and stage 1
# judges the motion; a check that vision cannot settle should come back 3, and
# the action side is where that moment gets caught. Telling this half to read
# the slowing flag would collapse the split.
_CHECKS = (
 ("A", "Are the hands bringing an object to REST -- lowering it onto the place it is\n"
       "   meant for and letting go there?",
  ("the object is down on its place and the grippers are opening off it",
   "the object is just above where it goes and still coming down",
   "an object is held near its place -- setting it down and passing over it look\n"
   "      the same from here",
   "it has just been let go and the hands are lifting away",
   "nothing is being set down")),
 ("B", "Are the hands turning a FIRM box or carton -- one that holds its shape --\n"
       "   rolling or swinging it to a different facing?",
  ("a firm box is held and its facing is changing",
   "a firm box is held and the wrists have begun to twist",
   "a firm box is held, and turning it and carrying it look the same from here",
   "a firm box has just been turned and is being released or re-gripped",
   "no firm box is being turned -- nothing held, or what is held is soft")),
 ("C", "Is what the hands are handling a SOFT bag or sack -- something that sags and\n"
       "   changes shape, with no fixed pose of its own?",
  ("a soft bag is in the grippers and visibly sagging",
   "a soft bag is in the grippers, its shape only partly in view",
   "something is held whose stiffness cannot be told from here",
   "a soft bag has just been released, or the hands are closing on one",
   "what is held is firm, or nothing is held")),
 ("D", "Are the hands moving an object ACROSS -- sending it on its way to the far\n"
       "   side, where it only has to end up over there?",
  ("an object is held and travelling across, well clear of where it started",
   "an object is held and the move across has begun",
   "an object is held in transit, and crossing over and placing look the same\n"
   "      from here",
   "the move across has just finished, or is about to start",
   "nothing is being carried across")),
)

_AXES = "".join(
    f"{q}) {text}\n" + "".join(f"   {5-k} = {a}\n" for k, a in enumerate(anchors))
    for q, text, anchors in _CHECKS)

ASK = ("Answer only from what the cameras show about the MOMENT in front of you. The "
       "measurements above are context; the checks below are about the scene, and a "
       "check the pictures cannot settle should get its middle grade rather than a "
       "guess.\n"
       "Many moments are none of these -- the arms are out of frame or lowered, or "
       "between one parcel and the next with nothing in hand. Answer 1 to every check "
       "when that is what you see; that is the correct answer there.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else -- one "
       "digit per check, using that check's own grades:\n\n" + _AXES + "Answer:")


def ceiling_from_twosided(picks):
    """Grade-weighted ceiling, on the same 1..3 scale the spec is written in.

    `picks` is five digits in A..E order, any of which may be None when the model
    did not answer that slot; a missing slot is dropped from its side rather than
    filled in, so a partial answer narrows the evidence instead of inventing it.
    """
    g = {q: (float(p) - 1.0) / 4.0 for q, p in zip("ABCDE", picks) if p is not None}
    tot = sum(g.values())
    if tot <= 0:
        # None of the phases are happening -- the arms are out of frame, lowered,
        # or moving between parcels. There is no pose, grip or placement to
        # preserve, so this is the freest kind of moment, not an unknown one:
        # it takes the top of the scale rather than a mid default. Stage 1 still
        # decides how much of that ceiling the chunk actually gets.
        return float(os.environ.get("ALLEX_IDLE_CEILING", 3.0))
    k = sum(v * CEILING[q] for q, v in g.items()) / tot
    return float(min(3.0, max(1.0, k)))
