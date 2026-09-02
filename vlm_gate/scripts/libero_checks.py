"""LIBERO's checks: one baseline question, three drawn from measured damage.

The specialised three come from the pools the same way robocasa's did -- the 40
tasks split by K4 damage, candidate phases read off each pool's instructions,
ranked by how many of that pool's tasks they cover. The questions themselves
share nothing with robocasa's: what compression breaks here is not what it
breaks there, and a check written for one benchmark carries the wrong sign in
the other. LIBERO's baskets are enclosed and safe; robocasa's cabinets are
enclosed and dangerous.

Question D is different in kind and is marked so. It is not drawn from damage,
because it CANNOT be: the pools are compared across tasks, and every one of the
40 tasks contains a transport phase, so transport never separates them and never
reaches the coverage ranking. It shows up in robocasa only by an accident of
that benchmark's task mix -- ten of its twenty-four tasks are long carries
across a kitchen, so there transport did separate whole tasks. The phase is real
either way: with nothing held near its destination there is no pose to lose, and
thinning the commands changes nothing. Leaving it out would mean no check ever
says a moment is safe to open up.
"""

# K4 damage (K4-boundary minus uncompressed, 50 episodes each) is the evidence.
# Placing onto a plate breaks; dropping into a basket does not:
#   plate    -0.65 -0.23 -0.27 -0.53 -0.19 -0.76 -0.08 -0.27 -0.57 -0.35
#   basket   -0.04 -0.37 -0.18 -0.06 -0.06 -0.04 -0.14 -0.08 -0.20 -0.04
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1}
COVER = {"A": 6, "B": 3, "C": 14}          # tasks each specialised check covers
_r = sum(COVER[q] for q in COVER if SIGN[q] < 0)
_s = sum(COVER[q] for q in COVER if SIGN[q] > 0)
WEIGHT = {q: COVER[q] / (_r if SIGN[q] < 0 else _s) for q in COVER}
WEIGHT["D"] = None                          # decided separately; see module docstring
NGRADE = 5

GUIDANCE = (
    "You are judging one instant of a table-top robot arm, to decide how far the next "
    "second of its motion could be thinned out -- how many of its commanded poses could "
    "be dropped, letting the arm travel further between the ones that remain, without "
    "changing what happens.\n\n"
    "Thinning moves the hand off its path by a little. Whether that matters depends on "
    "how much room the moment leaves: an object dropped into a basket lands fine "
    "anywhere inside it, an object set on a plate has to end up on the plate.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "several of these from one second to the next."
)

_CHECKS = (
 ("A", "Is the hand heading for a target no bigger than the thing it is holding -- the\n"
       "   face of a plate, the inside of a bowl, a switch, a slot -- where two or three\n"
       "   centimetres off misses it?",
  ("the object is over that target and coming down onto it",
   "the object is above the target, a moment from being released",
   "the hand is carrying toward it and about half way there",
   "the target is in the picture but the hand is doing something else",
   "there is no such target in this picture")),
 ("B", "Does this object have ONE assigned place among several that look alike -- the LEFT\n"
       "   plate rather than the right, the spot beside another object -- so that putting\n"
       "   it in the wrong one of them fails?",
  ("the object is being set into the one place it belongs, with lookalikes beside it",
   "the object is over that place and about to go down",
   "the object is in transit and more than one candidate place is in view",
   "such places are in the picture but nothing is being placed",
   "nothing here has an assigned place")),
 ("C", "Is the object going into something that catches it -- a basket, an open drawer, a\n"
       "   wide flat top -- where it only has to land inside and not at any spot?",
  ("the object is over the opening and being let go into it",
   "the object is above it and coming down",
   "the object is being carried toward it and is part way there",
   "such a container is in view but nothing is going into it",
   "there is nothing here to drop into")),
 ("D", "Are the hands between things -- reaching toward an object they have not touched\n"
       "   yet, or carrying one with its destination still well away?",
  ("nothing is close: the hand is in open space, mid-reach or mid-carry",
   "the hand is closing the last stretch toward something",
   "the hand is about to make contact or arrive",
   "contact has just been made or just released",
   "the hand is at its object or its destination right now")),
)

_AXES = "".join(
    f"{q}) {text}\n" + "".join(f"   {5-k} = {a}\n" for k, a in enumerate(anchors))
    for q, text, anchors in _CHECKS)

ASK = ("Answer only from what the cameras show about the MOMENT in front of you.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else -- one "
       "digit per check, using that check's own grades:\n\n" + _AXES + "Answer:")
