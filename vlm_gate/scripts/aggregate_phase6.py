"""phase6(5문항) 라벨 집계 — v6b 와 같은 구조, 문항 역할만 다르다.

phase6 의 다섯 축은 안전 하나와 위험 넷이다:
  A 빈 공간 통과            -> 안전
  B 손잡이·노브·레버·문 모서리 -> 위험 (구속된 일방향 부품)
  C 하중이 옮겨가는 순간      -> 위험 (사건형)
  D 턱·선반·벽을 넘거나 돌아감 -> 위험 (경로 꺾임)
  E 팔이 움직이는 중 베이스 주행 -> 위험 (도달 한계)

v6b(4문항) 는 A·D 를 안전, B·C 를 위험으로 썼다. 스크립트를 고치지 않고 새로 두는
이유는 phase5 라벨을 그 스크립트로 다시 만들 수 있어야 비교가 성립하기 때문이다.
"""
import glob, json, os
import numpy as np, pandas as pd

BASE = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
TAG = os.environ.get("TAG", "v6b_phase6")
OUTP = os.path.expanduser(os.environ.get(
    "OUTP", "~/quantization_agent_workspace/assets/labels/robocasa/v6b_phase6_full.parquet"))

COMPUTED = ("grip_transition", "reversal", "precise_hold", "infeasible_merge")
RISK_Q = "BCDE"     # 위험 문항
SAFE_Q = "A"        # 안전 문항

instr = {}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(l)
    c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""

rows = {}
for p in sorted(glob.glob(f"{BASE}/output/_gate_distill/{TAG}_s16_*.jsonl")):
    for l in open(p):
        try:
            r = json.loads(l)
            if all(k in r for k in ("A", "B", "C", "D", "E", "grip_transition")):
                rows[(r["ep"], r["f"])] = r
        except Exception:
            pass
print(f"수집 {len(rows)}프레임")
if not rows:
    raise SystemExit(f"{TAG}_s16_*.jsonl 에서 읽은 게 없다")

keys = sorted(rows)
CR = np.array([[rows[k][c] for c in COMPUTED] for k in keys])
V = np.array([[rows[k][q] for q in "ABCDE"] for k in keys])

risk = 1 - np.prod(1 - np.column_stack([CR] + [V[:, "ABCDE".index(q)] for q in RISK_Q]), axis=1)
safe = 0.5 + 0.5 * V[:, "ABCDE".index(SAFE_Q)]
raw = (1 - risk) * safe
rank = (np.argsort(np.argsort(raw)) / (len(raw) - 1)).astype(np.float64)

df = pd.DataFrame({"episode_index": [k[0] for k in keys], "frame_index": [k[1] for k in keys],
                   "task": [instr.get(k[0], "") for k in keys], "p_yes": rank, "p_raw": raw,
                   "quantize": (rank >= 0.5).astype(int)})
for i, c in enumerate(COMPUTED):
    df[f"c_{c}"] = CR[:, i]
for i, q in enumerate("ABCDE"):
    df[f"q_{q}"] = V[:, i]
os.makedirs(os.path.dirname(OUTP), exist_ok=True)
df.to_parquet(OUTP, index=False)

print(f"저장 {OUTP}  {len(df)}행  태스크 {df.task.nunique()}종")
print("계산 플래그 발생률: " + "  ".join(f"{c}={CR[:, i].mean():.1%}" for i, c in enumerate(COMPUTED)))
print("VLM 문항 평균: " + "  ".join(f"{q}={V[:, i].mean():.3f}" for i, q in enumerate("ABCDE")))
# 동점률은 noisy-OR 포화의 직접 지표다. 이진 플래그만 쓰면 여기가 크게 부풀고,
# 그러면 순위정규화가 임의 순서를 만들어낸다.
tie = float((raw == raw.max()).mean())
print(f"raw conf 평균 {raw.mean():.3f} std {raw.std():.3f} | 최댓값 동점 {tie:.2%}")
