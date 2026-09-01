"""Stage 2 as one graded question: the model reads the scene and names the ceiling.

The shipped stage 2 asks four yes/no checks and derives a ceiling from a
per-subtask prior looked up by label. That cannot run on a recording with no
subtask labels — the replay set has none, its task field is the empty string —
and the lookup is partly redundant anyway: measured over v1, the checks already
identify the situation from the scene alone on two of the four subtasks,
Rotate Box at 99.8% and Rotate PolyBag at 100%.

Where it fails is Pass Object, at 3.2%: the "parking at one exact spot" check
answers 0.535 on it against the "moving sideways" check's 0.330, so a transport
reads as a placement. Both are a hand carrying something, and the thing that
separates them — how forgiving the destination is — is often not yet on screen.

Asking four yes/no checks and reverse-engineering a ceiling from them turns that
into a hard error. Asking for the ceiling directly, over graded tokens read from
the logits, turns it into a probability spread between two levels instead: the
confusion survives as uncertainty rather than as a wrong answer.

The levels are anchored to situations rather than named as numbers, and the
ceiling each implies is stated. Level 2 covers both precise placement and a
two-handed rotation, which is this run's change — those situations were pinned
at 1.0 before and are allowed 2x here.
"""
import os

# level -> the ceiling it asserts. Ordinal single digits, because the judge reads
# one token at the answer slot and "2.5" is two.
LEVELS = (1.0, 2.0, 2.5, 3.0)

STAGE2_GRADED_GUIDANCE = (
    "You are deciding how far the next second of this robot's motion can be thinned out — "
    "how many of its commanded poses could be dropped, letting the arm travel further between "
    "the ones that remain, without changing what happens.\n\n"
    "Skipping is harmless while the hand is going somewhere and the exact path there does not "
    "matter. It stops being harmless when the hand has to arrive somewhere precisely, when the "
    "shape of the grip is doing the work, or when what is held can shift while it moves.\n\n"
    "Judge the moment in front of you, not the task as a whole. The same handling passes "
    "through several of these situations from one second to the next."
)

STAGE2_GRADED_ASK = (
    "The measurements above already state how the arm and the hands move; do not repeat them. "
    "Answer only from what the cameras show about the MOMENT in front of you.\n"
    "Choose the ONE line below that best matches this moment. Answer on its own line as "
    "\"A) 3\"— the digit of that line and nothing else.\n"
    "1 = the hand is arriving at its target right now: closing on it, setting it down, or "
    "coming to a stop where the exact final position is what matters. Nothing here can be "
    "skipped.\n"
    "2 = the hand is placing something with care, or turning a firm object held between both "
    "palms, so the pose has to be tracked but a small overshoot is recoverable.\n"
    "3 = the thing being handled is soft and can shift or flop — a bag or sack — so the motion "
    "must stay smooth but its path is not exact.\n"
    "4 = the hand is simply travelling: carrying something across open space, reaching toward "
    "an object it has not touched yet, or withdrawing afterwards. Ending a few centimetres off "
    "would change nothing.\n"
    "Answer:"
)


def ceiling_from_graded(dist, levels=LEVELS):
    """Expected ceiling under the model's distribution over the levels.

    The expectation is used rather than the argmax so that a model torn between
    "carrying" and "placing" lands between their ceilings instead of committing
    to one of them. That is the whole reason for grading this question: the
    Pass/Bring confusion is real and should show up as a value in between.
    """
    d = list(dist)[:len(levels)]
    tot = sum(d)
    if tot <= 0:
        return float(os.environ.get("ALLEX_DEFAULT_CEILING", 2.0))
    return float(sum(p * k for p, k in zip(d, levels)) / tot)


def argmax_ceiling(dist, levels=LEVELS):
    """The level the model actually picked — kept for diagnosis, not for the ratio."""
    d = list(dist)[:len(levels)]
    return float(levels[max(range(len(d)), key=lambda i: d[i])])
