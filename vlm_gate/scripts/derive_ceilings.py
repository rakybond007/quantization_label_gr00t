"""태스크별 배속 상한표. 벤치마크를 인자로 받는다.

    python vlm_gate/scripts/derive_ceilings.py <bench>
    -> vlm_gate/analysis/<bench>_task_ceilings.json

상한은 두 조건이 다 성립하는 가장 높은 배속이다(`derive_task_ceilings.py` 와
같은 규칙).

    1. 성공률이 허용 폭(TOL) 안에서 유지된다
    2. 스텝이 줄어든다 -- **짝지은 에피소드로만 본다**

둘째를 짝짓지 않으면 안 된다. `steps_on_success` 는 성공한 에피소드만의 평균이라
조건마다 부분집합이 다르고, 배속을 올려 성공률이 떨어지면 살아남은 쪽이 더 쉬운
집합이라 배속 효과가 아니라 집합이 바뀐 효과가 나온다.

## 잰 배속과 라벨 눈금은 다르다 -- 가까운 칸으로 옮긴다

라벨 눈금은 `1.0 · 1.5 · 2.0 · 2.5` 인데 실제로 잰 배속은 거기 딱 맞지 않는다.

    robocasa  1.000  1.500  2.000  2.500        맞는다
    libero    1.000  1.667  2.000  2.500        1.667 이 1.5 자리다

**정확한 1.5 를 고집할 이유가 없다.** 실행 규칙 자체가 1.5 를 정확히 못 낸다 --
재예측 평균을 5 에 맞추는 장부를 쓰면 K=1.5 의 실제 배속이 1.429 다. 눈금은
디코더가 가진 칸의 이름이지 물리량이 아니다.

**가까운 칸으로 옮기는 것은 안전한 쪽으로 틀린다.** 1.667 을 견딘 태스크는 1.5 를
당연히 견딘다. 반대로 1.667 에서 깨지면 1.5 를 모르는 채 한 칸 내리는데, 그것도
보수적이다. 추이를 보는 데 문제 없다.

상한표는 **명령 클리핑을 푼** 실행으로 만든다. 구동부 한계(dyn)까지 3배로 연
실행은 쓰지 않는다 -- libero 에서 1.429배가 이미 0.616 으로 무너져 압축이 아니라
제어기 손상을 재고 있었다(2026-09-08, 각 2000에피).
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

GRID = (1.0, 1.5, 2.0, 2.5)
TOL = 0.10          # 절대 허용폭. 50에피 표준오차 0.07 의 1.4배
REL_TOL = 0.25      # 상대 허용폭. 기준선 대비 이만큼 넘게 잃으면 깨진 것으로 센다
FLOOR = 0.35        # 기준선이 이보다 낮으면 상한을 논하지 않는다
MIN_PAIRS = 10

_T = Path("/sjw_alinlab2/home/taekwan/Data")
_R = Path("/rlwrld-unified-checkpoints/taekwan")
_O = Path(os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output"))

# 벤치마크 -> [(잰 배속, 결과 디렉터리 또는 eval_results.json 의 실행 이름)].
# 첫 줄이 기준선이어야 한다.
#
# 다른 클러스터(ALIN)에서 돈 실행은 이 서버에 원자료가 없고 `eval_results.json`
# 에만 있다. 그런 칸은 실행 **이름**을 적어 두면 거기서 읽는다. 대신 에피소드
# 단위 자료가 없으므로 **짝지은 스텝 검정을 못 하고 성공률 규칙만 쓴다** --
# 그 칸에는 "(짝검정 없음)" 이 붙는다.
#
# **robocasa 사다리는 조건이 섞여 있다.**
#
#     1.0  baseline_full_v2        클립 켬   <- 기준선
#     1.5  K1p5_clip3              클립 풂
#     2.0  baseline_compress_K2    클립 켬
#     2.5  K2p5_clip3              클립 풂
#
# 상한 1.5 를 받은 태스크는 "1.5(클립 풂) 통과, 2.0(클립 켬) 실패" 이므로, 조건이
# 바뀌는 자리에서 깨진 것을 배속 탓으로 읽을 위험이 있다.
#
# **그래도 그대로 쓴다. robocasa 에서 클립을 풀어도 성공률이 안 움직이는 것이
# 측정돼 있다** (24태스크, 압축을 건 채로):
#
#     K=3  클립1 0.497 -> 클립3 0.500   +0.002  t +0.16
#     K=4  클립1 0.404 -> 클립4 0.382   -0.022  t -1.52
#
# 배경도 맞는다 -- 병합 명령이 컨트롤러 범위를 벗어나는 비율이 robocasa K2 는
# 7.4% 인데 libero K2 는 40.7% 다. libero 는 클립이 실제로 무는 벤치이고 robocasa
# 는 아니다. **libero 사다리는 그래서 클립 조건을 섞지 않았다.**
LADDERS = {
    "robocasa": [
        (1.000, _O / "robocasa/baseline_full_v2_with_action_steps"),
        (1.500, "baseline_compress_K1p5_clip3"),      # ALIN, json 에서 읽는다
        (2.000, _O / "robocasa/baseline_compress_K2"),
        (2.500, "baseline_compress_K2p5_clip3"),      # ALIN, json 에서 읽는다
    ],
    "libero": [
        (1.000, _T / "libero_all_Fine"),
        (1.667, _R / "eqs_libero_naive_k2_noclip_repeat"),   # 1.5 자리
        (2.000, _R / "eqs_libero_naive_k2_replan6"),
        (2.500, _R / "eqs_libero_naive_k4_noclip"),
    ],
}


def snap(r):
    """잰 배속을 라벨 눈금의 가장 가까운 칸으로."""
    g = np.asarray(GRID, dtype=float)
    return float(g[int(np.argmin(np.abs(g - float(r))))])


def load(root):
    """태스크 -> [(성공, 스텝), ...]  에피소드 순서 그대로."""
    o = {}
    for pat in ("*/*_results.txt", "*/*/*_results.txt", "*/prediction.txt"):
        for f in Path(root).glob(pat):
            k = f.parent.name.replace("vid_", "")
            if f.name.endswith("_results.txt"):
                k += "/" + f.stem.split("_")[0]
            if k in o:
                continue
            d = []
            for line in open(f, errors="replace"):
                q = line.strip().split("\t")
                ok = st = None
                if len(q) >= 2 and q[1].strip() in ("True", "False"):
                    ok = q[1].strip() == "True"
                    if len(q) >= 3:
                        try:
                            st = float(q[2])
                        except ValueError:
                            pass
                elif "is_success:" in line:
                    ok = "True" in line.split("is_success:")[1][:12]
                    if "action_steps:" in line:
                        try:
                            st = float(line.split("action_steps:")[1].split()[0])
                        except (ValueError, IndexError):
                            pass
                if ok is not None:
                    d.append((ok, st))
            if d:
                o[k] = d
    return o


JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "analysis", "eval_results", "eval_results.json")


def load_json(bench, name):
    """`eval_results.json` 에서 태스크별 성공률만. 에피소드 단위 자료는 없다."""
    d = json.load(open(JSON))["runs"].get(bench, {}).get(name)
    if not d:
        return None
    # 에피소드 목록을 흉내 낸다 -- 성공 수만큼 True, 나머지 False. 성공률은
    # 정확하고 스텝은 None 이라 짝검정이 자동으로 건너뛰어진다.
    return {t: [(True, None)] * v["n_success"]
               + [(False, None)] * (v["n"] - v["n_success"])
            for t, v in d["tasks"].items()}


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "libero"
    if bench not in LADDERS:
        raise SystemExit(f"모르는 벤치마크 {bench}. 아는 것: {sorted(LADDERS)}")

    D, have, nopair = {}, [], set()
    for r, p in LADDERS[bench]:
        if isinstance(p, str):
            d = load_json(bench, p)
            if d is None:
                print(f"[ ] {r:5.3f} json 에 {p} 없음")
                continue
            D[r] = d
            nopair.add(r)
            have.append(r)
            print(f"[o] {r:5.3f} -> 눈금 {snap(r):.1f}  태스크 {len(d)}  "
                  f"{p}  (json · 짝검정 없음)")
            continue
        if not Path(p).exists():
            print(f"[ ] {r:5.3f} 없음 -- {p}")
            continue
        D[r] = load(p)
        have.append(r)
        print(f"[o] {r:5.3f} -> 눈금 {snap(r):.1f}  태스크 {len(D[r])}  {Path(p).name}")
    if not have or have[0] != LADDERS[bench][0][0]:
        raise SystemExit("기준선이 없다")
    base = D[have[0]]

    print("\n%-28s %s | %6s  %s" % (
        "task", " ".join(f"{r:>6.3f}" for r in have), "상한", "왜"))
    out = {}
    for tk in sorted(base):
        sr = {r: (np.mean([v[0] for v in D[r][tk]]) if tk in D[r] else None)
              for r in have}
        if sr[have[0]] is None:
            continue
        b0 = sr[have[0]]
        # 기준선이 바닥이면 상한을 논할 자리가 아니다. TurnOffStove 는 압축 없이도
        # 0.20 이라 어느 배속에서든 차이가 잡음이다.
        if b0 < FLOOR:
            out[tk] = 1.0
            print("%-28s %s | %6.2f  %s" % (
                tk[:28], " ".join(f"{sr[r]:6.2f}" if sr[r] is not None else "     -"
                                  for r in have), 1.0,
                f"기준선 {b0:.2f} 이 바닥({FLOOR}) 미만 -- 판정 안 함"))
            continue
        ceil, why = 1.0, "-"
        for r in have[1:]:
            if sr[r] is None:
                why = f"{r:.3f} 결과 없음"
                break
            # 절대 폭만 보면 기준선이 낮은 태스크에서 너무 관대해진다 --
            # CoffeeSetupMug 은 0.25 -> 0.16 으로 36% 를 잃는데 절대차가 -0.09 라
            # 통과한다. 상대 폭을 같이 본다.
            if sr[r] < b0 - TOL:
                why = f"{r:.3f} 에서 {sr[r] - b0:+.2f}"
                break
            if b0 > 0 and (b0 - sr[r]) / b0 > REL_TOL:
                why = f"{r:.3f} 에서 {(sr[r] - b0) / b0:+.0%} (상대)"
                break
            a, b = D[have[0]][tk], D[r][tk]
            n = min(len(a), len(b))
            pair = [(a[i][1], b[i][1]) for i in range(n)
                    if a[i][0] and b[i][0] and a[i][1] and b[i][1]]
            if len(pair) >= MIN_PAIRS:
                dd = float(np.mean([q - p for p, q in pair]))
                if dd >= 0:
                    why = f"{r:.3f} 에서 스텝이 안 줆 ({dd:+.0f}, n={len(pair)})"
                    break
            ceil = snap(r)
            why = f"{r:.3f} 까지 유지" + (" (짝검정 없음)" if r in nopair else "")
        out[tk] = ceil
        print("%-28s %s | %6.2f  %s" % (
            tk[:28], " ".join(f"{sr[r]:6.2f}" if sr[r] is not None else "     -"
                              for r in have), ceil, why))

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "analysis", f"{bench}_task_ceilings.json")
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print("\n상한 분포:", dict(sorted(collections.Counter(out.values()).items())))
    print(f"-> {os.path.normpath(dst)}")


if __name__ == "__main__":
    main()
