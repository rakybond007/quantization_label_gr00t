"""phase9's five checks: the questions, their signs and their weights.

Kept apart from the runners so a labeller can import them without the module
executing an argv-driven script on the way in.
"""
NGRADE = 5

# Risk questions subtract from the confidence, stability questions add to it, and
# each side's weights sum to 1 so the two sides balance despite being 2 and 3.
#
# Weights are the COUNT of pool tasks a question covers, minus the count it
# wrongly covers in the opposite pool. Counting tasks rather than summing their
# damage is deliberate: the risk pool holds six tasks, so each damage figure
# carries a lot of sampling error, and weighting by damage let two borderline
# tasks (+0.040, +0.020) push B down to 0.120 -- removing B entirely then moved
# the final ranking by 0.019 Spearman, i.e. a question that was not doing
# anything. A count asks how common the phase is, which is the steadier quantity.
#
# The endpoint-tolerance question that covers the whole risk pool was tried here
# and rejected: it answers at 0.78 correlation with A, so the two are one
# question wearing two labels, and it also collided with D's "a few centimetres
# off would change nothing" -- D's separation halved and level 2 usage fell from
# 8.5% to 4.1%. Task overlap is fine; answer overlap is not.
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1}
WEIGHT = {"A": 4 / 6, "B": 2 / 6,                    # risk pool: 6 tasks
          "C": 4 / 15, "D": 5 / 15, "E": 6 / 15}     # stable pool: E is 8 - 2 over

GUIDANCE = (
    "You are judging one second of a kitchen robot's motion, to decide how much of it "
    "could be thinned out -- how many of its commanded poses could be dropped, letting "
    "the arm travel further between the ones that remain, without changing the outcome.\n\n"
    "Two things decide that. Some moments hinge on one exact contact or one exact "
    "placement, and thinning them destroys the thing that mattered. Other moments have "
    "their path held by a mechanism, or a destination so forgiving that arriving a few "
    "centimetres off changes nothing.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "both kinds from one second to the next."
)

# The input is ONE instant -- three cameras at the same frame index, side by side
# (gen_robocasa_tiles_shard.py concatenates v[fi] across readers). So the scale
# cannot talk about how much OF THE MOMENT is the condition; there is no span to
# take a fraction of. It is a ladder of physical proximity instead, which is what
# a single still actually shows: contact, almost-contact, approaching, present but
# unengaged, absent.
#
# That also gives level 2 a visible meaning. The previous scale's 2 was "mostly
# does not hold", indistinguishable by eye from 1, and it took 43 of 9,490 slots.
SCALE = (
    "  5 = it is happening right now -- the picture shows the contact, the grip or\n"
    "      the position the check describes\n"
    "  4 = not yet, but the hand is right up against it, a centimetre or a single\n"
    "      motion away\n"
    "  3 = the hand is heading for it and still some way off\n"
    "  2 = the thing the check is about is there in the picture, but the arm is busy\n"
    "      with something else\n"
    "  1 = there is nothing in this picture the check could be about\n"
)

_AXES = (
    "A) Does what this motion achieves come down to ONE small contact -- pressing a\n"
    "   button, flicking a switch, turning a dial to a set position -- so that the\n"
    "   contact either lands correctly or the task fails?\n"
    "B) Is the place the held object is going INTO closed in on more than one side by\n"
    "   walls, a door frame or a shelf, leaving little room around it?\n"
    "C) Is the thing being moved carried on a hinge or a rail -- a door, a drawer --\n"
    "   so the mechanism, not the arm, decides the path it takes?\n"
    "D) Is an object being set down on a wide open surface, where landing a few\n"
    "   centimetres off would change nothing?\n"
    "E) Is the gripper carrying something across open space, with nothing near enough\n"
    "   to be struck?\n"
)

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")

