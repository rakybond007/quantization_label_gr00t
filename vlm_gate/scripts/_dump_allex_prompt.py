"""allex 프롬프트를 실제 데이터로 한 벌 찍는다. 손으로 짜맞추지 않는다."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# **라벨링할 때 쓴 환경으로 맞춘다.** 코드 기본값은 여섯(IDLE 포함)인데
# 실제 라벨은 sbatch_v5tempo_v3.sh 가 다섯으로 줄여 돌렸다. 기본값으로 찍으면
# 라벨에 없는 문항이 딸려 나온다.
os.environ.setdefault("ALLEX_CHECKS", "CLAMP,LOOSE,SHOVE,FLIP,FREE")
import allex_v3_checks as C

# 라벨 기록에서 실제로 쓰인 한 줄을 집는다
F = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/allex_v5tempo_v3/records.jsonl")
rows = [json.loads(l) for l in open(F)][:400]
print(f"기록 {len(rows)}행 · 키 {sorted(rows[0].keys())}\n")
r = rows[0]
print("=== 기록 한 줄 (라벨 결과) ===")
print(json.dumps({k: r[k] for k in list(r)[:14]}, ensure_ascii=False, indent=1)[:900])
print(f"\n=== ACTIVE (실제 문항) === {C.ACTIVE}")
print("\n=== GUIDANCE ===")
print(C.GUIDANCE)
print("\n=== ASK (문항 + 등급표) ===")
print(C.ASK)
print("\n=== SIGN / WEIGHT ===")
for q in C.ACTIVE:
    print(f"  {q:6s} {'감점' if C.SIGN[q] < 0 else '가점'}  {C.WEIGHT[q]:.3f}")
print("\n=== TASK_RANGE (띠) ===")
for k, v in C.TASK_RANGE.items():
    print(f"  {k:18s} {v[0]} ~ {v[1]}")
