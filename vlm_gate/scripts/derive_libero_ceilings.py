"""libero 태스크별 배속 상한표. 클립만 푼 사다리에서 만든다.

`derive_task_ceilings.py` 와 같은 규칙이다. 상한은 두 조건이 다 성립하는 가장
높은 배속이다.

    1. 성공률이 허용 폭(TOL) 안에서 유지된다
    2. 스텝이 앞 배속보다 더 줄어든다 -- **짝지은 에피소드로만 본다**

두 번째를 짝짓지 않으면 안 된다. `steps_on_success` 는 성공한 에피소드만의
평균이라 조건마다 부분집합이 다르고, 배속을 올려 성공률이 떨어지면 살아남은
쪽이 더 쉬운 집합이라 배속 효과가 아니라 집합이 바뀐 효과가 나온다.

**구동부를 연 실행(`released_*`)은 쓰지 않는다.** 1.429배에서 이미 0.616 으로
무너져 압축이 아니라 제어기 손상을 재고 있다(2026-09-08, 각 2000에피).

띠의 아래끝은 1.0 이고 위끝이 상한이다. 상한이 1.0 이면 그 태스크는 압축하지
않는다는 뜻이다.

    python vlm_gate/scripts/derive_libero_ceilings.py
"""
import collections
import json
import os
from pathlib import Path

import numpy as np

T = Path("/sjw_alinlab2/home/taekwan/Data")
R = Path("/rlwrld-unified-checkpoints/taekwan")
B = Path(os.path.expanduser(
    "~/quantization_agent_workspace/vlm_gate/output/libero"))

# 배속 -> 결과 디렉터리. 전부 **명령 클리핑만** 푼 것이어야 한다.
# 1.5 는 F_level 디코더 눈금이라 반드시 필요하다. 없으면 그 칸을 건너뛴다.
LADDER = [
    (1.000, T / "libero_all_Fine"),
    (1.500, B / "cliponly_K1p5"),
    (1.667, R / "eqs_libero_naive_k2_noclip_repeat"),
    (2.000, R / "eqs_libero_naive_k2_replan6"),
    (2.500, R / "eqs_libero_naive_k4_noclip"),
]
GRID = (1.0, 1.5, 2.0, 2.5)     # 라벨이 쓸 수 있는 값. 1.667 은 진단으로만 본다.
TOL = 0.10                      # 50에피 표준오차 0.07 의 1.4배
MIN_PAIRS = 10

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "libero_task_ceilings.json")


def load(root):
    """태스크 -> [(성공, 스텝), ...]  에피소드 순서 그대로."""
    o = {}
    pats = ("*/*_results.txt", "*/*/*_results.txt")
    for pat in pats:
        for f in Path(root).glob(pat):
            k = f.parent.name.replace("vid_", "") + "/" + f.stem.split("_")[0]
            if k in o:
                continue
            d = []
            for line in open(f):
                q = line.strip().split("\t")
                if len(q) >= 2 and q[1].strip() in ("True", "False"):
                    st = None
                    if len(q) >= 3:
                        try:
                            st = float(q[2])
                        except ValueError:
                            pass
                    d.append((q[1].strip() == "True", st))
            if d:
                o[k] = d
    return o


def main():
    D, have = {}, []
    for r, p in LADDER:
        if not Path(p).exists():
            print(f"[ ] {r:5.3f} 없음 -- {p}")
            continue
        D[r] = load(p)
        have.append(r)
        print(f"[o] {r:5.3f} 태스크 {len(D[r])} -- {p.name}")
    if 1.0 not in D:
        raise SystemExit("기준선(1.0)이 없다")
    base = D[1.0]

    print("\n%-16s %s | %6s  %s" % (
        "task", " ".join(f"{r:>6.3f}" for r in have), "상한", "왜"))
    out = {}
    for tk in sorted(base):
        sr = {}
        for r in have:
            d = D[r].get(tk)
            sr[r] = np.mean([v[0] for v in d]) if d else None
        if sr[1.0] is None:
            continue
        ceil, why = 1.0, "-"
        for r in have[1:]:
            if sr[r] is None:
                why = f"{r:.3f} 결과 없음"
                break
            if sr[r] < sr[1.0] - TOL:
                why = f"{r:.3f} 에서 {sr[r] - sr[1.0]:+.2f}"
                break
            a, b = D[1.0][tk], D[r][tk]
            n = min(len(a), len(b))
            pair = [(a[i][1], b[i][1]) for i in range(n)
                    if a[i][0] and b[i][0] and a[i][1] and b[i][1]]
            if len(pair) >= MIN_PAIRS:
                dd = float(np.mean([q - p for p, q in pair]))
                if dd >= 0:
                    why = f"{r:.3f} 에서 스텝이 안 줆 ({dd:+.0f}, n={len(pair)})"
                    break
            ceil, why = r, f"{r:.3f} 까지 유지"
        # 라벨이 쓸 수 있는 눈금으로 내린다. 1.667 까지 버텨도 눈금엔 1.5 가 있다.
        g = [v for v in GRID if v <= ceil + 1e-9]
        out[tk] = max(g) if g else 1.0
        print("%-16s %s | %6.2f  %s" % (
            tk, " ".join(f"{sr[r]:6.2f}" if sr[r] is not None else "     -"
                         for r in have), out[tk], why))

    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print("\n상한 분포:", dict(sorted(collections.Counter(out.values()).items())))
    print(f"-> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
