"""libero 문항을 측정된 손상에서 끌어낸다. `PROMPT_METHOD.md` 절차 그대로.

    1. 전체 태스크를 두 풀로 가른다 (측정된 손상)
    2. 각 풀의 지시문에서 국면 후보를 뽑는다
    3. 덮는 태스크 수로 줄을 세운다 -- 반대 풀을 잘못 덮는 만큼 뺀다
    3-1. 쌍도 본다 (혼자서는 0 이지만 둘이 함께면 가르는 것)
    4. 부호는 어느 풀에서 나왔는지가 정한다
    5. 가중치는 덮는 태스크 수, 각 변의 합이 1

손상은 **2.5배**에서 잰다. 1.667·2.0 은 안정 쪽이 천장에 붙어 순서가 안 나오고,
2.5배가 0.02 부터 1.00 까지 벌어져 가장 잘 가른다.

사다리 정본은 `analysis/eval_results/libero_ladder_noclip.md` 이고 원본은
taekwan 의 `eqs_libero_naive_*` 다. 서버에서 돌리면 원본을 다시 읽고, 맥에서
돌리면 아래 박아 둔 표를 쓴다(둘은 같은 값이다).

    python vlm_gate/scripts/derive_libero_questions.py
"""
import json
import os
from pathlib import Path

# --- 사다리 --------------------------------------------------------------
# 태스크당 50에피. 1.0 = taekwan libero_all_Fine, 2.5 = eqs_libero_naive_k4_noclip.
S1 = {
    "spatial/0": 1.00, "spatial/1": 0.98, "spatial/2": 1.00, "spatial/3": 0.98,
    "spatial/4": 0.94, "spatial/5": 0.98, "spatial/6": 1.00, "spatial/7": 0.98,
    "spatial/8": 1.00, "spatial/9": 0.98,
    "object/0": 1.00, "object/1": 0.98, "object/2": 1.00, "object/3": 0.94,
    "object/4": 1.00, "object/5": 0.96, "object/6": 1.00, "object/7": 1.00,
    "object/8": 1.00, "object/9": 0.98,
    "goal/0": 0.98, "goal/1": 0.96, "goal/2": 0.98, "goal/3": 0.88,
    "goal/4": 0.98, "goal/5": 1.00, "goal/6": 1.00, "goal/7": 1.00,
    "goal/8": 1.00, "goal/9": 0.86,
    "long/0": 0.84, "long/1": 1.00, "long/2": 0.92, "long/3": 0.92,
    "long/4": 0.86, "long/5": 0.94, "long/6": 0.86, "long/7": 0.96,
    "long/8": 0.80, "long/9": 0.82,
}
S25 = {
    "spatial/0": 0.20, "spatial/1": 0.56, "spatial/2": 0.68, "spatial/3": 0.10,
    "spatial/4": 0.60, "spatial/5": 0.22, "spatial/6": 0.50, "spatial/7": 0.54,
    "spatial/8": 0.28, "spatial/9": 0.40,
    "object/0": 1.00, "object/1": 0.58, "object/2": 0.84, "object/3": 0.90,
    "object/4": 0.84, "object/5": 0.92, "object/6": 0.62, "object/7": 1.00,
    "object/8": 0.60, "object/9": 0.94,
    "goal/0": 0.60, "goal/1": 0.18, "goal/2": 0.90, "goal/3": 0.16,
    "goal/4": 0.92, "goal/5": 1.00, "goal/6": 0.46, "goal/7": 0.88,
    "goal/8": 0.84, "goal/9": 0.22,
    "long/0": 0.48, "long/1": 0.02, "long/2": 0.54, "long/3": 0.92,
    "long/4": 0.76, "long/5": 0.86, "long/6": 0.40, "long/7": 0.86,
    "long/8": 0.16, "long/9": 0.40,
}
INSTR = {
    "spatial/0": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "spatial/1": "pick up the black bowl next to the ramekin and place it on the plate",
    "spatial/2": "pick up the black bowl from table center and place it on the plate",
    "spatial/3": "pick up the black bowl on the cookie box and place it on the plate",
    "spatial/4": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "spatial/5": "pick up the black bowl on the ramekin and place it on the plate",
    "spatial/6": "pick up the black bowl next to the cookie box and place it on the plate",
    "spatial/7": "pick up the black bowl on the stove and place it on the plate",
    "spatial/8": "pick up the black bowl next to the plate and place it on the plate",
    "spatial/9": "pick up the black bowl on the wooden cabinet and place it on the plate",
    "object/0": "pick up the alphabet soup and place it in the basket",
    "object/1": "pick up the cream cheese and place it in the basket",
    "object/2": "pick up the salad dressing and place it in the basket",
    "object/3": "pick up the bbq sauce and place it in the basket",
    "object/4": "pick up the ketchup and place it in the basket",
    "object/5": "pick up the tomato sauce and place it in the basket",
    "object/6": "pick up the butter and place it in the basket",
    "object/7": "pick up the milk and place it in the basket",
    "object/8": "pick up the chocolate pudding and place it in the basket",
    "object/9": "pick up the orange juice and place it in the basket",
    "goal/0": "open the middle drawer of the cabinet",
    "goal/1": "put the bowl on the stove",
    "goal/2": "put the wine bottle on top of the cabinet",
    "goal/3": "open the top drawer and put the bowl inside",
    "goal/4": "put the bowl on top of the cabinet",
    "goal/5": "push the plate to the front of the stove",
    "goal/6": "put the cream cheese in the bowl",
    "goal/7": "turn on the stove",
    "goal/8": "put the bowl on the plate",
    "goal/9": "put the wine bottle on the rack",
    "long/0": "put both the alphabet soup and the tomato sauce in the basket",
    "long/1": "put both the cream cheese box and the butter in the basket",
    "long/2": "turn on the stove and put the moka pot on it",
    "long/3": "put the black bowl in the bottom drawer of the cabinet and close it",
    "long/4": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "long/5": "pick up the book and place it in the back compartment of the caddy",
    "long/6": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "long/7": "put both the alphabet soup and the cream cheese box in the basket",
    "long/8": "put both moka pots on the stove",
    "long/9": "put the yellow and white mug in the microwave and close it",
}

# --- 후보 국면 -----------------------------------------------------------
# 각 풀의 지시문을 읽고 뽑은 것. 어느 태스크가 걸리는지는 지시문과 장면에서
# 사람이 정한다 -- 여기 적힌 목록이 그 판정이고, 문항 문구는 이것을 부른다.
CAND = {
    # ---- 위험 풀에서
    "SMALL_TARGET": dict(
        pool="risk",
        gloss="놓을 자리가 좁고 정해진 한 점이다 -- 접시, 버너, 걸이. "
              "받아 주는 통도 아니고 넓은 면도 아니다",
        tasks=["spatial/0", "spatial/1", "spatial/2", "spatial/3", "spatial/4",
               "spatial/5", "spatial/6", "spatial/7", "spatial/8", "spatial/9",
               "goal/1", "goal/8", "goal/9", "long/2", "long/4", "long/6",
               "long/8"]),
    "CROWDED_PICK": dict(
        pool="risk",
        gloss="집을 것이 다른 물건 위에 얹혀 있거나 사이에 끼어 있어, 손이 들어갈 "
              "길이 한 줄뿐이다. 같은 '그릇을 접시에' 인데 goal/8(자유롭게 놓인 "
              "그릇)은 안정이고 spatial 의 넷은 위험인 것이 이 갈래다",
        tasks=["spatial/0", "spatial/1", "spatial/3", "spatial/5", "spatial/6",
               "spatial/7", "spatial/8", "spatial/9"]),
    "THIN_GRIP": dict(
        pool="risk",
        gloss="납작한 판때기라 손가락이 얇은 면에 내려앉아야 한다",
        tasks=["object/1", "object/6", "object/8", "long/1", "long/5", "long/6"]),
    "TIGHT_MOUTH": dict(
        pool="risk",
        gloss="좁은 입구를 지나 안쪽 자리에 넣는다 -- 열어 놓은 서랍 안, 전자레인지 안",
        tasks=["goal/3", "long/9", "spatial/4"]),
    "TWO_IN_ROW": dict(
        pool="risk",
        gloss="같은 자리에 연달아 둘을 놓아, 앞의 것이 뒤의 것의 자리를 좁힌다",
        tasks=["long/0", "long/1", "long/4", "long/6", "long/7", "long/8"]),
    # ---- 안정 풀에서
    "CATCHER": dict(
        pool="stable",
        gloss="목적지가 받아 준다 -- 통, 바구니, 그릇. 몇 센티 빗나가도 그대로다",
        tasks=["object/0", "object/1", "object/2", "object/3", "object/4",
               "object/5", "object/6", "object/7", "object/8", "object/9",
               "goal/6", "long/0", "long/1", "long/7"]),
    "WIDE_SURFACE": dict(
        pool="stable",
        gloss="넓은 면 위에 올린다 -- 어디에 내려놓아도 된다",
        tasks=["goal/2", "goal/4", "long/3", "long/5"]),
    "NO_HOLD": dict(
        pool="stable",
        gloss="붙들고 있을 필요가 없는 일이다 -- 한 번 밀거나 돌리면 끝난다",
        tasks=["goal/0", "goal/5", "goal/7", "long/2", "long/3", "long/9"]),
    "FREE_TRANSIT": dict(
        pool="stable",
        gloss="빈 공간을 지나는 중이다 -- 닿을 것이 가까이 없다",
        tasks=[]),   # 지시문으로는 못 세고 장면에서만 참이다. 아래 주석 참조.
}

RISK_CUT, STABLE_CUT = -0.60, -0.16


def main():
    dmg = {k: round(S25[k] - S1[k], 3) for k in S1}
    order = sorted(dmg, key=lambda k: dmg[k])
    risk = [k for k in order if dmg[k] <= RISK_CUT]
    stable = [k for k in order if dmg[k] >= STABLE_CUT]
    neutral = [k for k in order if k not in risk and k not in stable]

    print(f"=== 1단계 두 풀 (2.5배 손상 기준, 위험 ≤{RISK_CUT} / 안정 ≥{STABLE_CUT}) ===")
    print(f"위험 {len(risk)} · 중립 {len(neutral)} · 안정 {len(stable)}\n")
    for nm, pool in (("위험", risk), ("안정", stable)):
        print(f"-- {nm}")
        for k in pool:
            print(f"   {dmg[k]:+.2f}  {k:<12} {INSTR[k][:66]}")
        print()

    print("=== 3단계 후보 줄 세우기 ===")
    print(f"{'후보':<14} {'출신':<7} {'자기풀':>6} {'반대풀':>6} {'점수':>6}  덮는 태스크")
    scored = []
    for key, c in CAND.items():
        own = risk if c["pool"] == "risk" else stable
        opp = stable if c["pool"] == "risk" else risk
        a = [t for t in c["tasks"] if t in own]
        b = [t for t in c["tasks"] if t in opp]
        scored.append((key, c, len(a), len(b), len(a) - len(b)))
    for key, c, a, b, sc in sorted(scored, key=lambda r: -r[4]):
        print(f"{key:<14} {c['pool']:<7} {a:>6} {b:>6} {sc:>6}  {len(c['tasks'])}개")

    print("\n=== 3-1단계 쌍 ===")
    keys = list(CAND)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            if CAND[ki]["pool"] != CAND[kj]["pool"]:
                continue
            both = set(CAND[ki]["tasks"]) & set(CAND[kj]["tasks"])
            own = risk if CAND[ki]["pool"] == "risk" else stable
            opp = stable if CAND[ki]["pool"] == "risk" else risk
            a, b = len(both & set(own)), len(both & set(opp))
            if both and a - b >= 2:
                print(f"{ki} + {kj}: 둘 다 걸리는 것 {sorted(both)} "
                      f"-> 자기풀 {a} / 반대풀 {b}")

    keep = [(k, c, a) for k, c, a, b, sc in scored if sc >= 2]
    print("\n=== 4·5단계 부호와 가중치 (점수 ≥2 만 남김) ===")
    for pool, sign in (("risk", "감점"), ("stable", "가점")):
        sel = [(k, c, a) for k, c, a in keep if c["pool"] == pool]
        tot = sum(a for _, _, a in sel)
        for k, c, a in sel:
            print(f"  {k:<14} {sign}  {a}/{tot} = {a / tot:.3f}")
    out = {"damage_at_2p5": dmg, "risk": risk, "stable": stable,
           "neutral": neutral,
           "selected": {k: {"pool": c["pool"], "covers": a} for k, c, a in keep}}
    p = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "analysis" \
        / "libero_question_derivation.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\n-> {os.path.normpath(p)}")


if __name__ == "__main__":
    main()
