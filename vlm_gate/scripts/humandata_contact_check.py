"""human_data conf 를 **손 검증 접촉 정답** 위에서 검증한다.

`pnp_task/contact_label_v2.json` 은 101 에피 전부가 사람 손을 거친 판이다
(68 `user_verified_v1_fused` = 사용자가 직접 확인 · 33
`tactile_gripper_proposal_frame_review` = 다시 계산 뒤 두 카메라 프레임으로 확인,
`CONTACT_LABEL_HANDOFF.md`). 에피마다 `onset`(첫 접촉) `offset`(놓는 순간)이 있다.

그리퍼 전이로 국면을 유도하던 것과 맞춰 보면 파지가 onset 과 +0.02 로 맞고 순서도
단조인데, **"물러남" 이 접촉구간의 0.88 지점 -- 아직 안 놓은 상태다.** 그리퍼 명령
열림 전이가 검증된 접촉 종료보다 이르다. 그래서 pnp 는 유도를 버리고 접촉 정답을
직접 쓴다.

`long_horizon_task/contact_label.json` 은 51 에피 전부 `gripper_position > 0.02`
규칙이라 **사람 확인이 없다** -- 내 유도와 같은 종류이므로 독립 정답이 아니다.
long 은 따로 눈으로 확인한다.

  CHECKS=humandata_v2_checks python humandata_contact_check.py <labels.jsonl>
"""
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CK = importlib.import_module(os.environ.get("CHECKS", "humandata_v2_checks"))
ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data"
Q = tuple(sorted(CK.SIGN))


def conf(r):
    g = CK.NGRADE - 1
    return 0.5 * (1 + sum(CK.SIGN[q] * CK.WEIGHT[q] * ((r[q] - 1) / g) for q in Q))


# 접촉 정답에서 바로 나오는 다섯 구간. 경계는 접촉 길이의 비율로 준다.
BINS = [("접촉 전(집으러)", lambda u: u < -0.05),
        ("첫 접촉(파지)", lambda u: -0.05 <= u <= 0.10),
        ("접촉 중반(운반)", lambda u: 0.10 < u <= 0.65),
        ("접촉 후반(내려놓는 중)", lambda u: 0.65 < u <= 1.0),
        ("놓은 뒤(물러남)", lambda u: u > 1.0)]


def main():
    R = [json.loads(l) for l in open(sys.argv[1])]
    R = [r for r in R if all(q in r and r[q] for q in Q)]
    CT = json.load(open(f"{ROOT}/pnp_task/contact_label_v2.json"))["episodes"]
    rows = []
    for r in R:
        if r["ds"] != "pnp_task":
            continue
        e = CT.get(f"{r['ep']:04d}")
        if not e:
            continue
        on, off = e["onset"], e["offset"]
        rows.append(((r["f"] - on) / max(off - on, 1), conf(r), r))
    print(f"pnp 청크 {len(rows)} · 문항 {''.join(Q)} · 판본 "
          f"{os.environ.get('CHECKS', 'humandata_v2_checks')}")
    print(f"\n{'손검증 접촉 구간':<24}{'n':>5}{'conf':>8}   " + "".join(f"{q:>6}" for q in Q))
    for name, f in BINS:
        v = [(c, r) for u, c, r in rows if f(u)]
        if not v:
            continue
        print(f"  {name:<22}{len(v):>5}{np.mean([c for c, _ in v]):>8.3f}   "
              + "".join(f"{np.mean([r[q] for _, r in v]):>6.1f}" for q in Q))
    print("\n기대: 첫 접촉과 접촉 후반(내려놓는 중)이 낮고, 접촉 전·중반·놓은 뒤가 높다.")


if __name__ == "__main__":
    main()
