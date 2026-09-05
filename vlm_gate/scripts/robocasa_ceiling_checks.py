"""robocasa checks that carry a ceiling, so the answer names a ratio.

phase9 asks whether a moment is risky and turns the answers into an ordering;
a threshold then picks K. That cannot say "this chunk takes 4x" -- it only says
which chunks are safer than which. These checks each carry the ratio their phase
was measured to tolerate, and the graded answers average those ratios, so what
comes out is in units of K.

The ceilings are measured, not assumed. Success at K=1,2,3,4 (naive, 24 tasks x
50 episodes) gives each task the largest K it survives within 6 points of its
uncompressed rate:

    4x  CloseDrawer CoffeePressButton TurnOffMicrowave TurnOffStove TurnOnStove
    3x  CloseSingleDoor PnPCounterToMicrowave TurnOnMicrowave TurnSinkSpout
    2x  PnPCounterToCab PnPCounterToStove PnPSinkToCounter
    1x  the other twelve

and each check's ceiling is the mean over the tasks it covers.

varK is deliberately absent from this. Its bound IS the controller clip, which
this project releases rather than respects, and its own numbers show the cap
never binds: varK3 and varK4 land at 231 and 238 steps with 0.626 and 0.627
success, so raising kmax changed nothing and the "4x" result is not a 4x result.
"""
import os

# phase -> the ratio it tolerated, averaged over the tasks the phase covers.
#
# There is no separate sign here, and that is the point: a check that raises the
# ratio and one that lowers it differ only in the number they carry. The mean of
# the five is 2.33, so
#
#     raises   E 4.00  travelling
#              A 3.67  pressing or turning something fixed, holding nothing
#     middling C 1.80  putting a held object INTO an enclosure
#     lowers   D 1.20  taking one OUT, or setting it on an open surface
#              B 1.00  gripping a handle and dragging it open
#
# and because the answer is a weighted MEAN, a low-ceiling check firing alongside
# a high one drags the result down -- which is what a penalty is. phase9 said the
# same thing as a weight on a sign; this says it as a value in units of K, which
# is why no rank normalisation or tau ladder is needed to get a ratio out.
#
# The model is NOT told which way each check pushes. It is asked what the scene
# shows and the arithmetic is ours; telling it the direction invites it to answer
# toward the ratio it thinks is right instead of describing the moment.
CEILING = {"A": 3.67, "B": 1.00, "C": 1.80, "D": 1.20, "E": 4.00}
COVER = {"A": 6, "B": 5, "C": 5, "D": 5, "E": None}   # E rests on physics, not coverage
NGRADE = 5

GUIDANCE = (
    "You are judging one instant of a kitchen robot, to decide how far the next second "
    "of its motion could be thinned out -- how many of its commanded poses could be "
    "dropped, letting the arm travel further between the ones that remain, without "
    "changing the outcome.\n\n"
    "What decides it is what the hand is doing to the world right now. Pressing a fixed "
    "button tolerates a coarse approach; holding a handle and dragging it does not, "
    "because losing the grip halfway loses the task.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "several of these from one second to the next."
)

# E is fixed as safe rather than derived. Every task contains transport, the
# damaged ones included, so a phase present in all of them can never separate the
# pools -- and treating it as a candidate would let the damaged tasks argue that
# transport is dangerous, which is exactly backwards. It is excluded from the
# ranking and pinned instead: with nothing being placed and nothing being held
# against a mechanism, there is no pose to lose.
_CHECKS = (
 ("A", "Is the hand pressing, pushing or turning something FIXED IN PLACE -- a button, a\n"
       "   dial, a drawer front -- while holding nothing?",
  ("the hand is on it and pushing or turning it now",
   "the hand is touching it, about to push or turn",
   "the hand is reaching for it and still short of it",
   "such a control is in the picture but the hand is elsewhere",
   "there is nothing fixed being worked here")),
 ("B", "Is the hand GRIPPING a handle or lever and dragging it -- pulling a drawer or door\n"
       "   open, swinging a faucet -- so that letting go part way loses it?",
  ("the handle is gripped and moving under the pull",
   "the handle is gripped and the pull is starting",
   "the hand is closing on the handle",
   "a handle is in view but nothing is gripping it",
   "no handle or lever is being worked")),
 ("C", "Is a held object being put INTO something -- a cabinet, a microwave, a sink, under\n"
       "   a dispenser -- where walls or a frame close in around where it has to go?",
  ("the object is down inside and being released",
   "the object is at the opening and going in",
   "the object is being carried toward the opening",
   "such an opening is in view but nothing is going into it",
   "nothing is being put into anything")),
 ("D", "Is a held object being taken OUT of an enclosure, or set down on an open surface --\n"
       "   lifted clear of a cabinet or pan, or placed on a counter or plate?",
  ("the object is being set down or lifted clear right now",
   "the object is at the lip, a moment from clearing or landing",
   "the object is being carried toward that spot",
   "such a spot is in view but nothing is being moved to it",
   "nothing is being taken out or set down")),
 ("E", "Are the hands simply travelling -- crossing space toward something not yet touched,\n"
       "   carrying something with its destination still well away, or withdrawing?",
  ("nothing is near: the hand is in open space, mid-reach or mid-carry",
   "the hand is closing the last stretch toward something",
   "the hand is about to make contact or arrive",
   "contact has just been made or just released",
   "the hand is at its object or its destination right now")),
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

    Returns a ratio, not a score: a chunk the model reads as purely transport
    comes back at E's ceiling, one it reads as dragging a handle at B's. A
    moment that is several things at once lands between them, which is the
    reason for grading rather than picking one.

    Nothing answered means no check applies. That is the empty-handed, nothing-
    nearby case, so it takes the transport ceiling rather than a middling
    default -- there is no pose to lose in it either.
    """
    g = {q: (float(p) - 1.0) / 4.0 for q, p in zip("ABCDE", picks) if p is not None}
    tot = sum(g.values())
    if tot <= 0:
        return float(os.environ.get("ROBOCASA_IDLE_CEILING", levels["E"]))
    return float(sum(v * levels[q] for q, v in g.items()) / tot)
