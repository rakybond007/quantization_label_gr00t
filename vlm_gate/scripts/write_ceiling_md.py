"""상한표 json 을 사람이 읽는 표로. 기계는 json 을, 사람은 md 를 본다.

    python vlm_gate/scripts/write_ceiling_md.py
    -> vlm_gate/analysis/TASK_CEILINGS.md

값은 `<bench>_task_ceilings.json` 에서 그대로 읽는다. 여기서 계산하지 않는다 --
두 군데서 계산하면 한쪽만 고쳐진다.
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(HERE, "..", "analysis")
OUT = os.path.join(ANA, "TASK_CEILINGS.md")

LADDER = {
    "libero": ("F_level 잡 157729 checkpoint-60000 (`level_ks` 1·1.5·2·2.5) 에서 "
               "level 을 하나씩 고정한 arm. seed 7 · replan 5 · `--args.no-action-clip` · "
               "kp 150 · torque stock. 40태스크 × 50에피.",
               [("1.0", "0.959"), ("1.5", "0.935"),
                ("2.0", "0.886"), ("2.5", "0.785")]),
    "robocasa": ("압축 없이 학습된 GR00T 베이스라인에 균일 고정 배속. 24태스크 × 50에피. "
                 "네 칸을 **클립 푼 조건 하나**로 본다 -- robocasa 는 클립을 풀어도 "
                 "성공률이 안 움직이는 것이 24태스크로 측정돼 있다"
                 "(K=3 +0.002 t +0.16, K=4 −0.022 t −1.52).",
                 [("1.0", "0.657"), ("1.5", "0.647"),
                  ("2.0", "0.598"), ("2.5", "0.545")]),
}

HEAD = """# 태스크별 배속 상한표

시점마다 몇 배속으로 갈지를 정할 때 쓰는 **띠**다. 아래끝과 위끝이 있고,
그 안에서 어디에 앉을지는 VLM 신뢰도가 정한다.

기계가 읽는 정본은 `<bench>_task_ceilings.json` 이다. 이 문서는 그것을 그대로
옮긴 것이므로 값을 여기서 고치지 말 것 -- `derive_ceilings.py` 를 다시 돌린다.

```
python vlm_gate/scripts/derive_ceilings.py libero
python vlm_gate/scripts/derive_ceilings.py robocasa
python vlm_gate/scripts/write_ceiling_md.py
```

## 어떻게 정하나

상한은 두 조건이 다 성립하는 가장 높은 배속이다.

1. 성공률이 허용 폭 안에서 유지된다 -- 절대 0.10(50에피 표준오차 0.07 의 1.4배)
   **그리고** 상대 25%. 절대만 보면 기준선이 낮은 태스크에서 너무 관대하다.
2. 스텝이 줄어든다 -- **짝지은 에피소드로만 본다.** `steps_on_success` 는 성공한
   에피소드만의 평균이라 조건마다 부분집합이 다르고, 배속을 올려 성공률이 떨어지면
   살아남은 쪽이 더 쉬운 집합이라 배속 효과가 아니라 집합이 바뀐 효과가 나온다.

압축 없이도 성공률이 0.35 미만인 태스크는 **판정하지 않는다.** 상한을 논할 자리가
아니다 -- 어느 배속에서든 차이가 잡음이다.

하한은 **망가지는 태스크가 아니면 1.5 로 고정**한다. allex 는 운용자가 태스크마다
하한과 상한을 둘 다 줬고(Bring Box 2.0~3.0), 그래서 "태스크마다 1.0 이 강제로
나오는" 문제가 없었다. 사다리에서 재면 하한 2.0 이 여럿 나오는데 그건 과하다 --
손상 0 은 50에피 점추정이다.

눈금은 `1.0 · 1.5 · 2.0 · 2.5` 다. 디코더가 이산이라 그 칸의 이름으로 적는다.
3배·4배는 안 쓴다 -- 상한이 눈금 최댓값에서 포화하므로 라벨에 아무것도 더해 주지
않는다.
"""


def main():
    L = [HEAD]
    for bench in ("libero", "robocasa"):
        p = os.path.join(ANA, f"{bench}_task_ceilings.json")
        if not os.path.exists(p):
            L.append(f"\n## {bench}\n\n아직 없다 (`derive_ceilings.py {bench}`).\n")
            continue
        d = json.load(open(p))
        note, rungs = LADDER[bench]
        L.append(f"\n## {bench} — {len(d)}태스크\n")
        L.append(note + "\n")
        L.append("\n| 배속 | 전체 성공률 |")
        L.append("|---:|---:|")
        for r, v in rungs:
            L.append(f"| {r} | {v} |")
        band = Counter(f"{v[0]}~{v[1]}" if isinstance(v, list) else f"1.0~{v}"
                       for v in d.values())
        L.append("\n띠 분포: " + " · ".join(
            f"`{k}` {n}개" for k, n in sorted(band.items())) + "\n")
        L.append("\n| 태스크 | 하한 | 상한 |")
        L.append("|---|---:|---:|")
        for t in sorted(d):
            v = d[t]
            lo, hi = (v if isinstance(v, list) else (1.0, v))
            mark = "  ← 압축 안 함" if hi <= 1.0 else ""
            L.append(f"| {t} | {lo:.1f} | {hi:.1f}{mark} |")
        L.append("")
    open(OUT, "w").write("\n".join(L))
    print(f"-> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
