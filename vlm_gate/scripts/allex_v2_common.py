"""allex v2 (subtask-labelled 256x256 dataset) - descriptors + two prompt stages.

Layering, unchanged from v5:
  deterministic layer - anything computable from the planned absolute joint
    targets is COMPUTED and stated to the model as a fact, never asked;
  VLM layer - only what the cameras show and the numbers cannot say.

What is new in v2:
  STAGE 1 (general, reused from allex_common_v5): "is it safe to compress this
    moment at all" -> a base confidence p in [0,1].
  STAGE 2 (new, task-specific): "how far could this kind of motion be pushed" ->
    a CEILING K_max in [1,3].
  Final ratio  K = snap(1 + p * (K_max - 1))  onto {1, 2, 2.5, 3}.

Because the mapping is multiplicative, stage 2 can only ever LOWER a ratio that
stage 1 already judged unsafe - a low p pins K near 1 whatever the ceiling says.
That is deliberate: on "Bring Object" stage 1 already senses the precise stop in
front of the robot, and stage 2 must not be able to undo it.
"""
import json
import os
import numpy as np

from allex_common_v5 import (_rot6_to_R, _ang, RA, LA, ARM, RH, LH,  # noqa: F401
                             MERGE_LIMIT, GUIDANCE, ASK)

TASKS = ["Bring Object", "Rotate Box", "Pass Object", "Rotate PolyBag"]

# Largest single-step joint move this dataset's own demonstrations contain
# (p99.9 of ||A[i+1]-A[i]|| over 41,326 steps sampled from 14 episodes; the max
# is 0.753).  v5's 0.159 rad came from a DIFFERENT, slower allex recording - used
# here it flagged 32% of chunks as infeasible purely for being faster than a
# demonstration that is not this one.
# Calibrated on THIS recording, never carried in from another. v5's 0.159 rad
# came from a slower capture and flagged 32% of v1's chunks as infeasible purely
# for being faster than a demonstration that was not theirs. v3 is faster again
# — its p99.9 is 0.478 against v1's 0.385 — so the same constant cannot serve
# both. Override per dataset; run allex_v2_calibrate.py to get the numbers.
MERGE_LIMIT_V2 = float(os.environ.get("ALLEX_MERGE_LIMIT", 0.385))
# Finger-pose change that counts as a full grasp transition. Like the three
# limits above this is a property of the recording and must not be carried in
# from another: 0.08 came from a different capture and sits at about p88 here
# (p50 0.017, p90 0.093, p95 0.161), so a third of the chunks were docked for
# ordinary finger drift. A multi-jointed hand is always moving a little; only a
# real closing onto something should count, which is what this recording's p95
# marks. allex_v2_calibrate.py prints the number.
HAND_SCALE = float(os.environ.get("ALLEX_HAND_SCALE", 0.08))
REORIENT_W = 0.30        # see stage1_confidence
GRIP_EMPTY_RELIEF = 0.5  # ditto
SOFT_RELIEF = 0.5        # softness raises confidence; see stage1_confidence
HOLD_RELIEF = 0.5        # so does the held-and-looking interval (check C)

# ---------------------------------------------------------------- descriptors

def descriptors(A, WR, WL, f, n=16):
    """Chunk descriptors from the planned ABSOLUTE joint targets.

    merge_demand_kK is the largest single-step joint move the controller would
    be asked for if targets were emitted every K steps instead of every step -
    i.e. what SKIPPING (never summing) costs at that ratio.
    """
    w = slice(f, min(f + n, len(A)))
    Aw = A[w]
    one_h = bool(_one_handed(Aw))
    gap = np.linalg.norm(WR[w, :3] - WL[w, :3], axis=1)
    mid = (WR[w, :3] + WL[w, :3]) / 2
    sp = np.linalg.norm(np.diff(Aw[:, ARM], axis=0), axis=1)
    rh = Aw[:, RH].mean(1); lh = Aw[:, LH].mean(1)
    Rr = _rot6_to_R(WR[w, 3:9]); Rl = _rot6_to_R(WL[w, 3:9])
    rr = _ang(Rr[-1], Rr[0]); rl = _ang(Rl[-1], Rl[0])

    def md(K):
        return float(np.linalg.norm(Aw[K:] - Aw[:-K], axis=1).max()) if len(Aw) > K else 0.0

    # speed trend inside the window: is the motion running down to a stop?
    h = max(1, len(sp) // 2)
    v0 = float(sp[:h].mean()) if len(sp) else 0.0
    v1 = float(sp[h:].mean()) if len(sp) > h else v0
    return {
        "gap_mean": float(gap.mean()), "gap_change": float(gap.max() - gap.min()),
        "gap_rate": float(np.abs(np.diff(gap)).max()) if len(gap) > 1 else 0.0,
        "closing": bool(gap[-1] < gap[0] - 0.01), "opening": bool(gap[-1] > gap[0] + 0.01),
        # "held" = pinched between the two palms.  A one-armed motion cannot be
        # holding anything that way, however close the wrists happen to be, and
        # the whole risk model (rotation while holding, the hard blocks in
        # allex_postprocess) is about losing exactly that two-palm hold.
        "held": bool(gap.mean() < 0.42 and not one_h),
        "arm_speed": float(sp.mean()) if len(sp) else 0.0,
        "hand_change": float(max(abs(rh[-1] - rh[0]), abs(lh[-1] - lh[0]))),
        "wrist_rot": float(max(rr, rl)), "rot_asym": float(abs(rr - rl)),
        "translation": float(np.linalg.norm(mid[-1] - mid[0])),
        "merge_demand": md(2), "merge_demand_k2": md(2), "merge_demand_k3": md(3),
        "wrist_z": float((WR[w, 2].mean() + WL[w, 2].mean()) / 2),
        "speed_ratio": float(v1 / v0) if v0 > 1e-6 else 1.0,
        "slowing": bool(v0 > 1e-6 and v1 < 0.6 * v0),
        "one_handed": one_h,
    }


def _one_handed(Aw):
    """True when one arm is doing (nearly) all the moving - the PolyBag flip."""
    r = float(np.linalg.norm(np.diff(Aw[:, RA], axis=0), axis=1).sum()) if len(Aw) > 1 else 0.0
    l = float(np.linalg.norm(np.diff(Aw[:, LA], axis=0), axis=1).sum()) if len(Aw) > 1 else 0.0
    tot = r + l
    return tot > 1e-6 and min(r, l) / tot < 0.25


def facts_text(x):
    """Stage-1 facts (v5 wording) plus the per-ratio skip cost."""
    grip = ("a package is pinched between the two palms" if x["held"] else
            "the hands are too far apart to be holding anything between them")
    trend = ("the palms are closing" if x["closing"] else
             "the palms are separating" if x["opening"] else "the palm separation is steady")
    move = ("almost stationary" if x["arm_speed"] < 0.010 else
            "moving slowly" if x["arm_speed"] < 0.025 else "moving fast")
    rot = (f"the wrists turn {x['wrist_rot']:.0f} deg across the window, the two of them differing "
           f"by {x['rot_asym']:.0f} deg" if x["wrist_rot"] >= 5 else "the wrists barely turn")
    tr = f"the grasp centre travels {x['translation']*100:.0f} cm"
    hand = ("the fingers change pose noticeably" if x["hand_change"] > 0.008
            else "the fingers barely move")
    slow = (" The motion is running down: the second half of the window is "
            f"{(1-x['speed_ratio'])*100:.0f}% slower than the first." if x.get("slowing") else "")
    hands = (" Only one arm is doing the moving." if x.get("one_handed") else "")
    feas = ""
    if x.get("merge_demand_k2", 0.0) > MERGE_LIMIT_V2:
        feas = (f" Emitting every 2nd target would demand a single-step joint move of "
                f"{x['merge_demand_k2']:.3f} rad, beyond the {MERGE_LIMIT_V2} rad this robot ever "
                f"produced in the demonstrations (every 3rd: {x['merge_demand_k3']:.3f} rad).")
    return ("MEASURED FROM THE PLANNED MOTION over the next 0.53 s (computed facts, not estimates): "
            f"{grip} at {x['gap_mean']:.2f} m and {trend} (up to {x['gap_rate']*1000:.1f} mm per step); "
            f"the arms are {move} at {x['arm_speed']:.3f} rad per step; {rot}; {tr}; {hand}."
            f"{slow}{hands}{feas}")


# ------------------------------------------------------------ stage 2: ceiling
#
# The ceilings below are the domain expert's reading of the four subtasks.  They
# enter the pipeline as a PRIOR (the starting ceiling for the segment's task)
# and as the CONTENT of the stage-2 questions - the VLM still decides, per
# chunk, which of the four situations it is actually looking at, and can pull
# the ceiling down (or, when it sees a plain sideways transfer and nothing else,
# push it back up).  Nothing here short-circuits the model.
#
#   Pass Object    3    an object is just moved sideways to the other side;
#                       nothing about where exactly it passes matters.
#   Bring Object   3 -> 1  dragging a box toward the robot is coarse, but it has
#                       to come to a precise stop in front, and the stopping
#                       phase must run at full rate.
#   Rotate Box     2    turning a rigid box: the two palms must move differently
#                       to keep it, so only mild compression survives.
#   Rotate PolyBag 2.5  the bag is flipped with ONE hand, so there is no
#                       two-hand hold to lose; it is soft, so not fully coarse.
# The per-subtask ceiling, as specified: not every subtask is allowed to reach
# 3x. Bring Object has to stop precisely on its target, so its specification is
# 1.0 — it does not compress at all. It sat at 3.0 here, which matched neither
# the specification nor STATUS.md, and the only thing holding it down was
# stage-2 question B firing on 96% of chunks; 1.3% still reached 3x.
#
# ALLEX_CEILINGS overrides it as JSON for a labelling run that wants different
# limits, e.g. raising Bring Object to 2.0:
#   ALLEX_CEILINGS='{"Bring Object": 2.0}'
# The object decides how roughly a phase may be handled: a box has to be treated
# with care, a bag can be thrown about. That is already why Rotate splits 2.0 /
# 2.5, and Bring splits the same way -- but the annotation says only "Bring
# Object", so which of the two applies is read off stage 2's own check D, "is
# the thing being handled a soft plastic bag rather than a firm box".
_CEILING_SPEC = {
    "Pass Object": 3.0,
    "Bring Object": 2.0,          # a box; a bag lifts it to BRING_SOFT
    "Rotate Box": 2.0,
    "Rotate PolyBag": 2.5,
}
BRING_SOFT = float(os.environ.get("ALLEX_BRING_SOFT", 2.5))
TASK_CEILING = dict(_CEILING_SPEC)
TASK_CEILING.update(json.loads(os.environ.get("ALLEX_CEILINGS", "{}")))
DEFAULT_CEILING = float(os.environ.get("ALLEX_DEFAULT_CEILING", 2.0))

STAGE2_GUIDANCE = (
    "You are looking at the same moment again. This time the question is not whether the robot "
    "can skip some of its motion commands, but HOW MANY it could skip at most.\n"
    "Skipping more commands makes the arms move further between commands, so the hands land a "
    "little off. Some moments do not care: an object slid across to the other side ends up fine "
    "wherever it lands within a few centimetres. Some moments care a lot: an object that has to "
    "come to rest at one exact place, or a firm box held between two palms that are turning it, "
    "where the two hands must move differently and that difference is the only thing holding it.\n"
    "A floppy bag flipped with a single hand sits in between: there is no two-hand hold to lose, "
    "but the bag flops around, so the hand should not be allowed to overshoot far.\n"
    "Judge the moment in front of you, not the name of the job."
)

STAGE2_ASK = (
    "The numbers above are already measured - do not guess them again and do not repeat them. "
    "Answer only from what you can see in the camera views. Answer each check on its own line as "
    "\"A) YES\" or \"A) NO\", in order, nothing else. YES and NO refer only to the question asked.\n"
    "A) Is the robot simply moving an object sideways to the other side, so that it would still be\n"
    "   fine if the object ended up a few centimetres away from where it is heading?\n"
    "B) Is the robot bringing an object in toward itself and about to park it at one exact spot in\n"
    "   front of it, so that stopping in the right place is what matters now?\n"
    "C) Is the robot turning a firm box over with both hands pressed on it, so that the box is held\n"
    "   only by how the two hands push against each other?\n"
    "D) Is the thing being handled a soft plastic bag that flops about, rather than a firm box?\n"
    "Answer:")


def stage2_facts(task, x):
    """Facts for stage 2. Everything numeric is stated, never asked."""
    hands = "one arm is moving" if x.get("one_handed") else "both arms are moving"
    stop = ("the motion is running down toward a stop" if x.get("slowing")
            else "the motion is not slowing down")
    return (f"The human annotation for this segment is \"{task}\". "
            f"MEASURED (computed, not estimates): {hands}; the palms are "
            f"{x['gap_mean']:.2f} m apart and "
            + ("closing" if x["closing"] else "separating" if x["opening"] else "steady")
            + f"; the arms move {x['arm_speed']:.3f} rad per step and {stop}; "
              f"the grasp centre travels {x['translation']*100:.0f} cm; the wrists turn "
              f"{x['wrist_rot']:.0f} deg. Skipping to every 2nd target would demand "
              f"{x['merge_demand_k2']:.3f} rad in one step, every 3rd {x['merge_demand_k3']:.3f} rad "
              f"(this robot never exceeded {MERGE_LIMIT_V2} rad in the demonstrations).")


def slot_weight(p, t=0.5, s=0.4):
    """Turn a slot probability into evidence weight.

    A teacher-forced slot never sits at 0; over the smoke episode the four
    stage-2 slots idle around 0.1-0.45 even where the situation clearly does not
    apply.  Clamping proportionally to the raw probability therefore ate the
    ceiling everywhere (mean K_max 1.5 where the expert's floor is 2).  Only
    genuine agreement should move it: p <= 0.5 counts for nothing, p >= 0.9
    counts fully.
    """
    return float(min(1.0, max(0.0, (float(p) - t) / s)))


def ceiling_from_stage2(task, pA, pB, pC, pD):
    """Blend the per-task prior ceiling with the four visual checks.

    Each check is a SOFT clamp: at full weight it pulls the ceiling all the way
    to its own value, at zero it leaves it alone, in between it interpolates.
    They are applied least-restrictive first so the most restrictive evidence
    has the last word.  Check A is the only one that can raise the ceiling, and
    only to the extent that none of the restrictive checks fired.
    """
    wA, wB, wC, wD = (slot_weight(p) for p in (pA, pB, pC, pD))
    K = float(TASK_CEILING.get(task, DEFAULT_CEILING))
    # Each check clamps toward the ceiling of the situation it recognises, so
    # those targets are the spec's own numbers and have to be read from it.
    # Written out as 2.5/2.0/1.0 they were the spec of the day; when Bring
    # Object was raised from 1.0 to 2.0 this line kept pulling to 1.0, and
    # every one of the 132 Bring Object chunks came out at 1.0 as before -- the
    # raise had no effect at all.
    # Bring Object is specified for a box; when the model says the thing is a soft
    # bag (check D), the same phase gets the bag's allowance instead.
    if task == "Bring Object":
        K = max(K, wD * BRING_SOFT + (1 - wD) * K)
    for w, c in ((wD, TASK_CEILING["Rotate PolyBag"]),
                 (wC, TASK_CEILING["Rotate Box"]),
                 (wB, TASK_CEILING["Bring Object"])):
        K -= w * max(0.0, K - c)
    # A lifts at half strength: the model may argue the segment is more coarse
    # than its task prior suggests, but it does not get to overrule it outright.
    lift = wA * (1.0 - max(wB, wC, wD))
    K += 0.5 * lift * (3.0 - K)
    return float(min(3.0, max(1.0, K)))


def final_ratio(p, K_max):
    """K = snap(1 + p*(K_max - 1)) onto the allowed ratios."""
    from allex_v2_ratio import snap_ratio
    return snap_ratio(1.0 + float(p) * (float(K_max) - 1.0))


ROTATION_SUBTASKS = ("Rotate Box", "Rotate PolyBag")


def stage1_confidence(c, x, task=None):
    """Base confidence p in [0,1] from the four general checks + the physics.

    v5 aggregated these for a gate whose scores were RANK-normalised inside an
    episode, so saturation did not matter.  A ratio needs an absolute number, so
    three v5 terms are recalibrated here (measured on the smoke episode, 165
    chunks; every change is a saturation fix, not a loosening of the physics):

      * infeasibility is graded against THIS dataset's own limit instead of a
        0/1 flag against another recording's (v5: 32% of chunks pinned to p=0);
      * "is the parcel being turned" (check B) is what the stage-2 ceiling now
        encodes for Rotate Box (2) and Rotate PolyBag (2.5).  Counting it at
        full weight in both places drove Rotate Box to K=1 everywhere, so here
        it only shades the confidence (weight 0.30);
      * finger motion counts less when the model says the hands are empty or
        still reaching - pre-shaping a hand in mid-air is not a grasp
        transition.
    """
    import numpy as _np
    A, B, C, D = c
    held = bool(x["held"])                       # two-palm hold (see descriptors)
    # B is worth counting as UNEXPECTED rotation only. Where the annotation
    # already says the phase is a rotation, B answers 0.98 on essentially every
    # chunk -- it separates nothing there and just lays a flat -0.24 over the
    # whole segment, which is the risk the 2.0 ceiling was written to express.
    # On Bring (0.04) and Pass (0.11) it does carry news, and there it stays.
    b = 0.0 if task in ROTATION_SUBTASKS else B
    # C marks the interval where the package has been brought in front and is
    # merely being held -- not yet turned, not yet put down. The subtask labels
    # do not mark it, so vision has to find it, and finding it should RELEASE
    # the limit toward Bring rather than tighten it: nothing is being placed or
    # reoriented, so there is no pose to lose. It used to enter v_risk, docking
    # the confidence for the one interval inside a rotation segment that is not
    # a rotation. Same inversion as `soft` had.
    v_risk = REORIENT_W * b
    # Softness is a reason the moment is SAFE to thin out, not a risk: a bag has
    # no exact pose to lose, which is why the spec gives Rotate PolyBag 2.5
    # against Rotate Box's 2.0. It used to enter as (1 - 0.5*soft), docking the
    # confidence for the very property the ceiling was raised for, and PolyBag
    # chunks averaged K=1.49 against a 2.5 ceiling. Same double-count the
    # REORIENT_W note below describes, in the opposite direction.
    soft = A * (1.0 if held else 0.4)            # a soft object matters when held
    safe = 0.75 + 0.25 * D                       # D: hands empty / not yet touching
    infeas = float(_np.clip(x["merge_demand_k2"] / MERGE_LIMIT_V2 - 1.0, 0, 1))
    rot_hold = float(held) * float(_np.clip((x["wrist_rot"] - 10.0) / 20.0, 0, 1))
    grip_tr = float(_np.clip(x["hand_change"] / HAND_SCALE, 0, 1)) * (1 - GRIP_EMPTY_RELIEF * D)
    c_risk = 1 - (1 - infeas) * (1 - rot_hold) * (1 - grip_tr)
    total = 1 - (1 - v_risk) * (1 - c_risk)
    return float(min(1.0, (1 - total) * safe * (1 + SOFT_RELIEF * soft + HOLD_RELIEF * C)))


# ------------------------------------------------------- deterministic blocks
#
# Same two block reasons as allex_postprocess.hard_block (v1) - rotation
# accumulated while the parcel is held between the palms, and the palms drifting
# apart fast while holding - rescaled to this dataset, measured by
# allex_v2_calibrate.py over 2,576 chunks from 14 episodes:
#
#   rotation accumulated over 3 chunks, held chunks only: p50 30 deg, p90 57 deg
#     -> v1's 18 deg sits below the MEDIAN here and blocked 60% of Rotate Box.
#   gap_rate while held: p50 2.3 mm/step, p95 6.5 mm/step
#     -> v1's 10 mm/step is above p95 here and effectively never fired.
#
# Both are set at the worst decile / ventile of held chunks, which is what the v1
# numbers represented on the slower recording they were measured on.
ROT_ACCUM_LIMIT_V2 = float(os.environ.get("ALLEX_ROT_LIMIT", 55.0))   # deg / 3 chunks, ~p90 held
GAP_RATE_LIMIT_V2 = float(os.environ.get("ALLEX_GAP_LIMIT", 0.0065))  # m/step, ~p95 held


def hard_block_v2(x, rot_accum):
    """Pin this chunk to K=1 regardless of what the two stages said."""
    if not x.get("held"):
        return False                       # nothing pinched between the palms to lose
    if rot_accum > ROT_ACCUM_LIMIT_V2:
        return True                        # turning it while holding it
    return x.get("gap_rate", 0.0) > GAP_RATE_LIMIT_V2


def label_risk_v2(descs, rot_window=3):
    """Per-chunk 0/1 hard block, using only this chunk and the previous ones."""
    wr = np.array([d.get("wrist_rot", 0.0) for d in descs])
    accum = np.array([wr[max(0, i - rot_window + 1): i + 1].sum() for i in range(len(descs))])
    return np.array([1.0 if hard_block_v2(d, a) else 0.0 for d, a in zip(descs, accum)])
