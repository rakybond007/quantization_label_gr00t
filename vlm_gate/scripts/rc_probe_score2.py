"""robocasa 문항 판을 **정답이 있는 두 가지로만** 채점한다.

태스크 내 AUC 는 폐기했다. 그 지표의 정답표(`robocasa_limits.json`)에서 국면 10개 중
7개가 데이터로 식별되지 않기 때문이다(`analysis/subaction/robocasa_limit_bands.json`):

    place_precise   1.25 ~ 1.25    식별
    place_open      1.25 ~ 1.667   식별
    pull_swing      1.5  ~ 1.83    식별
    나머지 7개      하한만, 위쪽 안 묶임

병목 모형에서 천장은 가장 빡빡한 국면 하나가 설명하면 끝이라, 그 하나가 될 필요가
없는 국면은 하한 이상 어떤 값이든 적합이 같다. `grasp_open` 4.00 은 측정값이 아니라
탐색 상단이다(2.00 과 RMSE 가 동일).

**그래서 실측이 뒷받침하는 주장은 두 개뿐이고, 그 두 개만 검사한다.**

  검사1 (F): 놓기가 빡빡하다. "쥔 채 내려놓는 중"(= place_precise 1.25 /
            place_open 1.25~1.667)이 **접근(>=3.17)·운반(>=1.83)·놓고 물러난 뒤**
            보다 낮아야 한다. 이 세 짝만 채점한다 -- **파지는 뺀다.** grasp_* 는
            하한뿐이라 파지가 어디 있어야 하는지 데이터가 말하지 않고, 넣으면
            (v10/v18 처럼) 파지가 묶음 평균을 끌어내려 비교가 오염된다.

  검사2 (C): 문·서랍을 쥐고 당겨 여는 것(pull_swing 1.5~1.83)은 빡빡하고, 밀어
            닫기·버튼·꼭지(contact_coarse/knob >= 3.17)는 관대하다. C 는 뒤쪽에만
            높아야 한다. v10 의 C 는 둘을 한 문항에 묶어 Open* 에도 최대 가점을 줬다.
            **분리 AUC 만 보면 안 된다** -- KNOB/PUSH 의 절대값을 같이 깎아도 AUC 는
            오른다. 부류별 평균을 같이 찍는 이유다.

  python rc_probe_score2.py analysis/ratio_prompt/rc_v18.jsonl [비교판.jsonl ...]
"""
import importlib
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")

# 태스크를 실측 분해의 국면 부류로 가른다(robocasa_subactions.json 과 같은 갈래)
PULL = ("OpenDrawer", "OpenSingleDoor", "OpenDoubleDoor")          # pull_swing 1.5~1.83
# KNOB 과 PUSH 를 나눠 둔다. 둘 다 contact_* >= 3.17(가장 관대)이지만 KNOB 은 어휘
# 정의가 "손잡이·꼭지를 **잡고** 돌림" 이라, C 가 "쥔다" 를 벌점으로 쓰면 관대한
# 쪽까지 같이 깎이는지 여기서만 보인다(v19 에서 0.960 -> 0.700 으로 실제로 그랬다).
KNOB = ("TurnOnStove", "TurnOffStove", "TurnOnSinkFaucet", "TurnOffSinkFaucet")
PUSH = ("CloseDrawer", "CloseSingleDoor", "CloseDoubleDoor",
        "CoffeePressButton", "TurnOffMicrowave", "TurnOnMicrowave", "TurnSinkSpout")
_RNG = np.random.default_rng(0)


def ci(a, b, n=2000):
    """짝비교 AUC 의 95% 부트스트랩 구간. n 이 36~126 이라 구간을 안 보면
    노이즈를 개선으로 읽는다 -- v18/v19 판정에서 실제로 그랬다."""
    v = [auc(_RNG.choice(a, len(a)), _RNG.choice(b, len(b))) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def conf(rec, CK, Q):
    g = {q: (rec[q] - 1) / (CK.NGRADE - 1) for q in Q}
    s = sum(CK.WEIGHT[q] * g[q] for q in Q if CK.SIGN[q] > 0)
    r = sum(CK.WEIGHT[q] * g[q] for q in Q if CK.SIGN[q] < 0)
    return (1 + s - r) / 2


def auc(a, b):
    """P(a 무작위 하나 < b 무작위 하나). 0.5 = 못 가름."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    return float(((a[:, None] < b[None, :]).sum()
                  + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def report(path):
    ver = os.path.basename(path).split("_")[-1].split(".")[0]      # rc_v18.jsonl -> v18
    CK = importlib.import_module(f"phase9_checks_{ver}")
    Q = sorted(CK.SIGN)
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if all(q in r for q in Q)]
    for r in rows:
        r["conf"] = conf(r, CK, Q)
    print(f"\n{'='*66}\n{ver}  ({len(rows)} 타일 · 문항 {''.join(Q)})\n{'='*66}")

    # --- 검사1: 데이터가 순서를 말해 주는 짝만 본다 ---
    # 식별된 국면만 쓴다. 파지는 grasp_* 하한뿐(>=1.667/1.83)이라 어느 쪽이 맞는지
    # 데이터가 말하지 않으므로 **채점에서 뺀다** -- 넣으면 비교가 오염된다.
    P = {}
    for ph, hv, tag in (("접근", False, "접근"), ("파지", True, "파지(쥔채)"),
                        ("파지", False, "파지(놓은뒤)"), ("운반", True, "운반"),
                        ("해제", True, "해제(쥔채)"), ("해제", False, "해제(놓은뒤)")):
        P[tag] = [r["conf"] for r in rows if r["phase"] == ph and r["held"] == hv]
    print("\n[검사1] 실측이 순서를 말해 주는 짝비교  (해제쥔채 = place_* 1.25~1.67)")
    print("  국면별 conf 평균")
    for tag, v in P.items():
        note = "  <- 데이터 없음(하한뿐), 채점 제외" if tag.startswith("파지") else ""
        if v:
            print(f"    {tag:<12} {len(v):3d}개  {np.mean(v):.3f}{note}")
    PAIRS = [("해제(쥔채)", "접근", "place_* 1.25~1.67  <  approach >=3.17"),
             ("해제(쥔채)", "운반", "place_* 1.25~1.67  <  carry >=1.83"),
             ("해제(쥔채)", "해제(놓은뒤)", "내려놓는 중  <  놓고 물러난 뒤(제약 없음)")]
    aucs = []
    print("  짝비교 AUC (낮아야 하는 쪽이 실제로 낮은 비율)")
    for lo, hi, why in PAIRS:
        a = auc(P[lo], P[hi])
        c = ci(P[lo], P[hi])
        aucs.append(a)
        print(f"    {a:.3f} [{c[0]:.2f},{c[1]:.2f}]   {why}")
    print(f"  평균 {np.mean(aucs):.3f}   0.5 = 못 가름 · 구간이 겹치면 판 구분 불가")
    # --- 검사2: C 가 당겨 열기와 밀기/돌리기를 가르나 ---
    cg = lambda ts: [(r["C"] - 1) / (CK.NGRADE - 1) for r in rows if r.get("task") in ts]
    pull, knob, pushonly = cg(PULL), cg(KNOB), cg(PUSH)
    push = knob + pushonly
    print(f"\n[검사2] C 가 '쥐고 당겨 열기'(빡빡 1.5~1.83) 를 "
          f"'밀기·돌리기'(관대 >=3.17) 와 가르나")
    print(f"  Open* 쥐고 당겨 열기 {len(pull):3d}개  C 평균 {np.mean(pull):.3f}"
          if pull else "  Open* 타일 없음")
    print(f"  Close*/Turn*/Press   {len(push):3d}개  C 평균 {np.mean(push):.3f}"
          if push else "  Close*/Turn* 타일 없음")
    if pull and push:
        c = ci(pull, push)
        print(f"  AUC(당겨열기 < 밀기·돌리기) = {auc(pull, push):.3f} "
              f"[{c[0]:.2f},{c[1]:.2f}]   0.5 = C 가 둘을 구분 못 함")
    print("  부류별 C 평균 -- 분리만 보지 말고 관대한 쪽 절대값이 유지되는지 볼 것")
    for tag, v, why in (("PULL 쥐고 당겨 열기", pull, "1.5~1.83 빡빡 · 낮아야"),
                        ("KNOB 잡고 돌림    ", knob, ">=3.17 최관대 · 높아야"),
                        ("PUSH 밀기·버튼    ", pushonly, ">=3.17 최관대 · 높아야")):
        if v:
            print(f"    {tag} {len(v):3d}개  {np.mean(v):.3f}   {why}")
    return rows


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p if p.startswith("/") else f"{BASE}/{p}")
