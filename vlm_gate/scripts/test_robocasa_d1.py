"""Meaningful policy/token/reader and F-level inference parity tests; CPU only."""
from itertools import product
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from robocasa_d1_policy import select_ratio, validate_result, validate_probabilities
from label_robocasa_d1 import complete_keys


def main():
    workspace = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(workspace/"F_level_hj/papers/reproducing/FLARE"))
    from flare.ratio_d1_inference import select_level
    seen = set()
    n = 0
    for grades in product(range(6), repeat=4):
        p = np.eye(6)[list(grades)].tolist()
        picks = [str(x+1) if x < 5 else "U" for x in grades]
        for cap in (1., 1.5, 2., 2.5):
            result = select_ratio(p, picks, cap)
            inferred, fallback = select_level(np.eye(4)[result["ratio_class"]], p, cap)
            assert inferred == result["ratio_class"]+1
            assert fallback == (not result["ratio_valid"])
            contact = select_ratio(p, picks, cap, fixed=True)
            assert contact["ratio"] == 1 and contact["ratio_valid"]
            if max(grades) < 5 and cap == 2.5:
                seen.add(result["ratio"])
            n += 1
    assert seen == {1., 1.5, 2., 2.5}
    for bad in ([[0]*6]*4, [[float('nan')]*6]*4, [[-1,2,0,0,0,0]]*4):
        try: validate_probabilities(bad)
        except ValueError: pass
        else: raise AssertionError("invalid probabilities accepted")
    p = np.eye(6)[[0,0,0,4]].tolist()
    good = dict(text="A) 1\nB) 1\nC) 1\nD) 5", picks=["1","1","1","5"],
                grade_token_picks=["1","1","1","5"], grade_probs=p)
    validate_result(good)
    for bad in ({**good,"text":good['text']+'\nExplanation 3'}, {**good,"grade_token_picks":["5"]*4}):
        try: validate_result(bad)
        except ValueError: pass
        else: raise AssertionError("misaligned generated tokens accepted")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/"raw.jsonl"
        path.write_text(json.dumps(dict(episode_index=0,frame_index=0))+'\n{"partial":')
        assert complete_keys(path) == {(0,0)}
        assert path.read_text().endswith('\n')
        assert path.with_suffix('.recovery.jsonl').exists()
    print(f"PASS: {n} policy/inference cases, contact, malformed probabilities/tokens, transactional tail recovery")


if __name__ == "__main__": main()
