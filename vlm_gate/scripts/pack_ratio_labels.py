"""배속 라벨을 허깅페이스에 올릴 한 벌로 묶는다.

받는 쪽은 이 한 벌과 `apply_ratio_labels.py` 만 있으면 자기 데이터셋 parquet 에
열을 붙일 수 있다. VLM 도 GPU 도 우리 서버 경로도 필요 없다.

    python vlm_gate/scripts/pack_ratio_labels.py <bench> <ratio.parquet> [--out DIR]
    python vlm_gate/scripts/pack_ratio_labels.py <bench> ... --push <repo_id>

무엇이 들어가나
--------------
    labels/<bench>.parquet        등급 A~E · 신뢰도 · 배속 · 접촉 표시
    ceilings/<bench>.json         태스크별 [하한, 상한]
    prompts/<bench>.txt           그 등급을 매긴 문항 전량
    checks/<bench>.py             부호·가중치 (등급 -> 신뢰도 를 다시 만들 때)
    README.md                     붙이는 법과 열의 뜻

**등급을 같이 넣는 것이 핵심이다.** 비싼 것은 VLM 이 매기는 등급뿐이고 그 뒤는
전부 산수다. 등급이 있으면 띠나 규칙을 바꿔 CPU 만으로 몇 초 만에 다시 라벨링할
수 있다 -- 받는 쪽에서도 그렇다.

`--push` 는 `huggingface_hub` 가 있어야 하고 토큰이 필요하다. 없으면 폴더만
만들고 멈춘다 -- 올리는 것은 밖으로 나가는 일이라 조용히 하지 않는다.
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(HERE, "..", "analysis")
PROMPTS = os.path.join(HERE, "..", "prompts")
PROMPT_FILE = {"libero": "libero_v2.txt", "robocasa": "robocasa_phase9.txt"}
CHECKS_FILE = {"libero": "libero_v2_checks.py", "robocasa": "phase9_checks.py"}

README = """---
license: apache-2.0
task_categories:
- robotics
tags:
- action-chunk-compression
- vla
---

# 배속 라벨 (ratio labels)

로봇 정책이 내는 액션 청크를 **시점마다 몇 배속으로 압축해도 되는지** 적은
라벨입니다. `{bench}` 벤치마크용입니다.

## 붙이는 법

```bash
git clone https://github.com/rakybond007/quantization_label_gr00t
python quantization_label_gr00t/vlm_gate/scripts/apply_ratio_labels.py \\
    --labels labels/{bench}.parquet \\
    --dataset /내/경로/lerobot/{bench} \\
    --out     /내/경로/{bench}_with_ratio
```

`pandas` 와 `pyarrow` 만 있으면 됩니다. GPU 도 VLM 도 필요 없습니다.
`--dry-run` 을 먼저 주면 아무것도 쓰지 않고 붙는 비율만 보여 줍니다.

## 붙는 열

| 열 | 뜻 |
|---|---|
| `ratio` | 이 시점에 쓸 배속. **1.0 · 1.5 · 2.0 · 2.5 중 하나** |
| `ratio_conf` | 그 배속을 정한 신뢰도 0~1. 확률이 아니라 **순서** 값 |
| `ratio_fixed` | 접촉이라 1.0 으로 고정된 시점이면 1 |
| `ratio_valid` | 라벨이 있는 시점이면 1 |

**`ratio_valid` 가 0 인 행은 손실에서 빼야 합니다.** 라벨이 없는 시점을
"1배속이 정답" 으로 학습시키면 안 됩니다. `ratio` 자리에 1.0 이 들어가 있는데
그것은 기본값이지 판정이 아닙니다.

## 안에 든 것

```
labels/{bench}.parquet     등급 A~E · 신뢰도 · 배속 · 접촉 표시
ceilings/{bench}.json      태스크별 [하한, 상한]
prompts/{bench}.txt        그 등급을 매긴 문항 전량
checks/{bench}.py          부호·가중치
```

**등급이 같이 들어 있습니다.** 비싼 것은 VLM 이 매기는 등급뿐이고 그 뒤는 전부
산수입니다. 띠나 규칙을 바꿔 CPU 만으로 다시 라벨링할 수 있습니다.

## 어떻게 만들었나

```
접촉이면        1.0배 고정 (그리퍼가 열리거나 닫히는 순간, 쥔 채 느리게 가는 구간)
아니면          신뢰도 순서로 그 태스크의 [하한, 상한] 안에 앉힌다
```

신뢰도는 VLM 이 문항 다섯에 매긴 1~5 등급에서 나옵니다.

```
등급 -> 값    (g − 1) / 4
신뢰도        ( 1 + Σw·가점 − Σw·감점 ) / 2
```

문항과 가중치는 장면 인상으로 지은 것이 아니라 **측정된 손상**에서 나왔습니다.
상한도 문항이 정하지 않고 **평가가 줍니다** -- 태스크마다 배속을 올려 가며
성공률과 스텝을 재서, 성공률이 유지되고 스텝이 실제로 줄어드는 가장 높은 배속이
상한입니다. 문항이 정하는 것은 그 상한을 얼마나 쓸 것인가뿐입니다.

자세한 것은 저장소의 `docs/RATIO_LABELS.md` 와
`vlm_gate/analysis/TASK_CEILINGS.md` 를 보세요.

## 통계

{stats}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=sorted(PROMPT_FILE))
    ap.add_argument("labels", help="ratio_label.py 가 낸 *_ratio.parquet")
    ap.add_argument("--out", default="", help="묶을 곳 (기본: dist/<bench>_ratio)")
    ap.add_argument("--push", default="", help="허깅페이스 repo_id. 주면 올린다")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    out = a.out or os.path.join(HERE, "..", "..", "dist", f"{a.bench}_ratio")
    for sub in ("labels", "ceilings", "prompts", "checks"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    d = pd.read_parquet(a.labels)
    need = {"episode_index", "frame_index", "ratio"}
    if need - set(d.columns):
        raise SystemExit(f"라벨에 열이 없다: {sorted(need - set(d.columns))}")
    d.to_parquet(os.path.join(out, "labels", f"{a.bench}.parquet"), index=False)

    src = os.path.join(ANA, f"{a.bench}_task_ceilings.json")
    if not os.path.exists(src):
        raise SystemExit(f"상한표가 없다: {src}")
    shutil.copy(src, os.path.join(out, "ceilings", f"{a.bench}.json"))
    shutil.copy(os.path.join(PROMPTS, PROMPT_FILE[a.bench]),
                os.path.join(out, "prompts", f"{a.bench}.txt"))
    shutil.copy(os.path.join(HERE, CHECKS_FILE[a.bench]),
                os.path.join(out, "checks", f"{a.bench}.py"))

    g = sorted(d.ratio.unique())
    stats = [f"- 행 {len(d):,} · 에피소드 {d.episode_index.nunique():,}"]
    if "task" in d.columns:
        stats.append(f"- 태스크 {d.task.nunique()}")
    stats.append("- 배속 분포  " + " · ".join(
        f"`{v}x` {(d.ratio == v).mean():.1%}" for v in g))
    stats.append(f"- 평균 배속 {d.ratio.mean():.3f}")
    if "fixed" in d.columns:
        stats.append(f"- 접촉으로 1.0 고정 {d.fixed.mean():.1%}")
    open(os.path.join(out, "README.md"), "w").write(
        README.format(bench=a.bench, stats="\n".join(stats)))
    print("\n".join(stats))
    print(f"-> {os.path.normpath(out)}")

    if not a.push:
        print("\n(--push <repo_id> 를 주면 허깅페이스에 올린다. "
              "올리는 것은 밖으로 나가는 일이라 기본값이 아니다)")
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("huggingface_hub 가 없다: pip install huggingface_hub")
    api = HfApi()
    api.create_repo(a.push, repo_type="dataset", private=a.private,
                    exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=a.push, repo_type="dataset")
    print(f"-> https://huggingface.co/datasets/{a.push}")


if __name__ == "__main__":
    sys.exit(main())
