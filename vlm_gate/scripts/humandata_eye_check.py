"""human_data 판을 **채점 도구 두 개**로 판정한다. 전량 라벨링 전에 돌리는 것이다.

정답 둘:

1. `pnp_task` -- 데이터셋이 주는 **손 검증 접촉 라벨**
   (`contact_label_v2.json`: 101에피 중 68 `user_verified_v1_fused` = 사용자가 직접
   확인, 33 `tactile_gripper_proposal_frame_review` = 다시 계산 뒤 두 카메라 프레임
   으로 확인). 에피마다 `onset`(첫 접촉) `offset`(놓는 순간)이 있다.

2. `long_horizon_task` -- **내가 눈으로 본 기록**
   (`analysis/human_data/EYE_LABELS_long.jsonl`, 54장). 손 검증본이 없어서
   (51에피 전부 `gripper_position > 0.02` 규칙이고 tactile 센서가 아예 없다)
   프레임을 뽑아 대조표로 직접 보고 적었다.

**무엇이 참이어야 하나.** 두 데이터셋이 목적지 폭에서 갈린다 -- pnp 는 넓고 평평한
접시, long 은 물체보다 간신히 넓은 빨간 컵이다. 그래서

    long   내려놓는 중(쥔 채 컵 위) <  운반          이어야 하고
    pnp    내려놓는 중(접시 위)     >= 운반 이어도 된다 (접시는 넉넉하다)

이 대조가 문항 집합의 검증이다. 한쪽만 보면 "놓기는 늘 위험" 이라는 상수를 재게 된다.

  CHECKS=humandata_v2_checks python humandata_eye_check.py <labels.jsonl>
"""
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = importlib.import_module(os.environ.get("CHECKS", "humandata_v2_checks"))
ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data"
EYE = f"{BASE}/analysis/human_data/EYE_LABELS_long.jsonl"
Q = tuple(sorted(CK.SIGN))


def conf(r):
    g = CK.NGRADE - 1
    return 0.5 * (1 + sum(CK.SIGN[q] * CK.WEIGHT[q] * ((r[q] - 1) / g) for q in Q))


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    return float(((a[:, None] < b[None, :]).sum()
                  + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def main():
    R = [json.loads(l) for l in open(sys.argv[1])]
    R = [r for r in R if all(q in r and r[q] for q in Q)]
    idx = {(r["ds"], r["ep"], r["f"]): r for r in R}
    print(f"라벨 {len(R):,} · 문항 {''.join(Q)} · 판본 "
          f"{os.environ.get('CHECKS', 'humandata_v2_checks')}")

    # --- 1. pnp: 손 검증 접촉 구간 ---
    CT = json.load(open(f"{ROOT}/pnp_task/contact_label_v2.json"))["episodes"]
    BINS = [("접촉 전(집으러)", lambda u: u < -0.05),
            ("첫 접촉(파지)", lambda u: -0.05 <= u <= 0.10),
            ("접촉 중반(운반)", lambda u: 0.10 < u <= 0.65),
            ("접촉 후반(내려놓는 중)", lambda u: 0.65 < u <= 1.0),
            ("놓은 뒤(물러남)", lambda u: u > 1.0)]
    rows = []
    for r in R:
        if r["ds"] != "pnp_task":
            continue
        e = CT.get(f"{r['ep']:04d}")
        if e:
            rows.append(((r["f"] - e["onset"]) / max(e["offset"] - e["onset"], 1), conf(r)))
    print(f"\n[pnp · 손 검증 접촉 정답]  {len(rows)} 청크")
    got = {}
    for name, f in BINS:
        v = [c for u, c in rows if f(u)]
        if v:
            got[name] = v
            print(f"  {name:<22}{len(v):>5}  conf {np.mean(v):.3f}")
    if "접촉 후반(내려놓는 중)" in got and "접촉 중반(운반)" in got:
        print(f"  내려놓는 중 vs 운반: {np.mean(got['접촉 후반(내려놓는 중)']):.3f} "
              f"vs {np.mean(got['접촉 중반(운반)']):.3f}  "
              f"-- 접시는 넉넉하므로 눌리지 않아도 된다")

    # --- 2. long: 눈 기록(앵커로 보증된 구간) ---
    E = [json.loads(l) for l in open(EYE)]
    hit, hit_v = {}, {}
    for e in E:
        for ph, (a, b) in e["spans"].items():
            for f in range(a, b):
                r = idx.get((e["ds"], e["ep"], f))
                if r is None:
                    continue
                c = conf(r)
                hit.setdefault(ph, []).append(c)
                if e["verified"]:
                    hit_v.setdefault(ph, []).append(c)
    nv = sum(1 for e in E if e["verified"])
    print(f"\n[long · 눈 기록] 주기 {len(E)} 중 앵커를 직접 본 주기 {nv} ({nv*3}장). "
          f"구간은 그 앵커가 보증한다.")
    print(f"  {'국면':<12}{'전체 n':>7}{'conf':>8}    {'확인주기 n':>9}{'conf':>8}")
    for ph in ("1집으러", "2파지", "3운반", "4내려놓는중", "5놓은뒤"):
        v, w = hit.get(ph, []), hit_v.get(ph, [])
        if not v:
            continue
        print(f"  {ph:<12}{len(v):>7}{np.mean(v):>8.3f}    {len(w):>9}"
              + (f"{np.mean(w):>8.3f}" if w else f"{'--':>8}"))
    for tag, H in (("전체", hit), ("눈으로 확인한 주기만", hit_v)):
        if H.get("4내려놓는중") and H.get("3운반"):
            a = auc(H["4내려놓는중"], H["3운반"])
            print(f"  [{tag}] AUC(내려놓는 중 < 운반) = {a:.3f}   "
                  f"{np.mean(H['4내려놓는중']):.3f} vs {np.mean(H['3운반']):.3f}")
    print("  **통과 조건** -- 좁은 컵에 쥔 채 내려놓는 순간이 운반보다 낮아야 한다.")
    print("  pnp 는 접시가 넉넉하므로 같은 자리가 눌리지 않아야 대조가 성립한다.")


if __name__ == "__main__":
    main()
