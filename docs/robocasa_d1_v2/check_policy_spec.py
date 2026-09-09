"""CPU specification checks only. Reads no labels and writes no dataset."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

CFG = json.loads(Path(__file__).with_name("policy_v1.json").read_text())
GRID = tuple(F(str(x)) for x in CFG["ratio_grid"])
REF = F(str(CFG["reference_ratio"]))


def select(score, cap=GRID[-1]):
    target = 1 + (REF - 1) * score
    return min((k for k in GRID if k <= cap), key=lambda k: (abs(k-target), k))


def bounds(values):
    lo = [F(0) if x is None else x for x in values]
    hi = [F(1) if x is None else x for x in values]
    return max(lo[2:])*(1-max(hi[:2])), max(hi[2:])*(1-max(lo[:2]))


def main():
    z = tuple(F(i, 4) for i in range(5))
    outcomes = set()
    n = 0
    for values in product(z, repeat=4):
        low, high = bounds(values)
        assert low == high
        outcomes.add(select(low))
        for axis in range(4):
            if values[axis] == 1:
                continue
            raised = list(values)
            raised[axis] += F(1, 4)
            newer = select(bounds(raised)[0])
            assert (newer <= select(low)) if axis < 2 else (newer >= select(low))
        for cap in GRID:
            assert select(low, cap) <= cap
            assert select(low, cap) == min(select(low), cap)
        n += 1
    assert outcomes == set(GRID)
    eps = F(1, 10**9)
    for boundary, lower, upper in zip((F(1, 6), F(1, 2), F(5, 6)), GRID, GRID[1:]):
        assert select(boundary-eps) == lower
        assert select(boundary) == lower
        assert select(boundary+eps) == upper
    # Confident 3-grade answers cannot reach 2x under this 2.5 reference mapping.
    three = {select(bounds(v)[0]) for v in product((F(0), F(1, 2), F(1)), repeat=4)}
    assert F(2) not in three
    # Expected grades follow the user's weighted-grade convention, not argmax.
    p = (F(3, 10), F(0), F(7, 10), F(0), F(0))
    expected = sum((i+1)*v for i, v in enumerate(p))
    assert expected == F(12, 5)  # 2.4
    assert (expected-1)/4 == F(7, 20)  # 0.35 on this five-grade coordinate
    # Highest speed remains reachable without any probability being exactly one.
    low_axis = sum(F(i, 4)*(F(9, 10) if i == 0 else F(1, 40)) for i in range(5))
    high_axis = sum(F(i, 4)*(F(9, 10) if i == 4 else F(1, 40)) for i in range(5))
    assert select(high_axis*(1-low_axis)) == F(5, 2)
    # Unknown redundant support must not veto an already established alternative.
    assert bounds([F(0), F(0), None, F(1)]) == (F(1), F(1))
    # Unknown risk genuinely spans different choices.
    low, high = bounds([None, F(0), F(0), F(1)])
    assert select(low) != select(high)
    # Complete monotone bounds for all numeric/unknown patterns.
    patterns = 0
    for values in product((*z, None), repeat=4):
        low, high = bounds(values)
        assert 0 <= low <= high <= 1
        for cap in GRID:
            assert select(low, cap) <= select(high, cap)
        patterns += 1
    print(json.dumps({"status": "PASS", "numeric_combinations": n,
                      "numeric_or_unknown_patterns": patterns,
                      "reachable_ratios": sorted(float(k) for k in outcomes),
                      "three_grade_reachable_ratios": sorted(float(k) for k in three),
                      "weighted_grade_example": float(expected),
                      "nondegenerate_probability_top_speed_reachable": True,
                      "scope": "exact rational arithmetic specification; no GPU, data join or VLM validation"}, indent=2))


if __name__ == "__main__":
    main()
