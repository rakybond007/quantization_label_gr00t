"""D1 v2 annotation policy. Scores are preferences, never success probabilities."""
import math
import re

from fractional_blocks import block_sizes

CATEGORIES = ["1", "2", "3", "4", "5", "U"]
GRID = [1., 1.5, 2., 2.5]
VERSION = "robocasa_d1_v2_proposal_1"


def validate_result(result):
    if result.get("error"):
        raise ValueError(str(result["error"]))
    text = result.get("text", "")
    pattern = r"\s*A\)\s*([12345U])\s*B\)\s*([12345U])\s*C\)\s*([12345U])\s*D\)\s*([12345U])\s*"
    match = re.fullmatch(pattern, text)
    if match is None:
        raise ValueError(f"not four exact category slots: {text!r}")
    picks = list(match.groups())
    if result.get("picks") != picks or result.get("grade_token_picks") != picks:
        raise ValueError("parsed and generated category token slots disagree")
    return picks, validate_probabilities(result.get("grade_probs"))


def validate_probabilities(probs):
    if not isinstance(probs, (list, tuple)) or len(probs) != 4:
        raise ValueError("expected four probability vectors")
    out = []
    for row in probs:
        if len(row) != 6:
            raise ValueError("expected six categories")
        row = [float(x) for x in row]
        if any(not math.isfinite(x) or x < 0 for x in row):
            raise ValueError("invalid probability")
        total = sum(row)
        if not math.isclose(total, 1., abs_tol=1e-4):
            raise ValueError(f"category mass is not one: {total}")
        out.append([x/total for x in row])
    return out


def nearest(score, cap):
    if cap not in GRID or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid score/cap")
    target = 1 + 1.5*score
    candidates = [k for k in GRID if k <= cap]
    distance = min(abs(k-target) for k in candidates)
    return min(k for k in candidates if abs(k-target) <= distance + 1e-12)


def select_ratio(probs, picks, cap, fixed=False):
    probs = validate_probabilities(probs)
    if len(picks) != 4 or any(x not in CATEGORIES for x in picks):
        raise ValueError("invalid picks")
    expected, values, unknown, entropy = [], [], [], []
    for p, pick in zip(probs, picks):
        mass = sum(p[:5])
        u = pick == "U" or p[5] >= max(p[:5]) or mass <= 1e-12
        e = sum((i+1)*p[i] for i in range(5))/mass if mass > 1e-12 else None
        expected.append(e)
        values.append((e-1)/4 if not u else None)
        unknown.append(u)
        entropy.append(-sum(x*math.log(x) for x in p if x > 0))
    low = [0. if x is None else max(0., min(1., x)) for x in values]
    high = [1. if x is None else max(0., min(1., x)) for x in values]
    clo = max(low[2:])*(1-max(high[:2]))
    chi = max(high[2:])*(1-max(low[:2]))
    kl, kh = nearest(clo, cap), nearest(chi, cap)
    valid = kl == kh
    before = kl if valid else 1.
    ratio = 1. if fixed else before
    score = clo if not any(unknown) else None
    sizes = block_sizes(16, ratio)
    return dict(ratio_rule=VERSION, question_activation=values,
                expected_grade_conditional_known=expected, unknown=unknown,
                unknown_prob=[p[5] for p in probs], grade_entropy=entropy,
                score_low=clo, score_high=chi, compression_score=score,
                risk_score=max(low[:2]) if not any(unknown[:2]) else None,
                support_score=max(low[2:]) if not any(unknown[2:]) else None,
                ratio_target=1+1.5*score if score is not None else None,
                score_boundary_distance=min(abs(score-b) for b in (1/6, .5, 5/6)) if score is not None else None,
                task_cap=float(cap), ratio_before_contact=before, ratio=ratio,
                ratio_class=GRID.index(ratio), ratio_valid=bool(valid or fixed),
                ratio_reason="legacy_contact" if fixed else "semantic_unknown" if not valid else "score_nearest",
                legacy_fixed=bool(fixed), block_sizes=sizes, action_count=len(sizes),
                effective_chunk_ratio=16/len(sizes))
