"""게이트 공동 학습이 백본을 실제로 바꿨는지 가중치로 확인한다.

게이트는 hidden_states[14] 를 읽으므로 그 손실은 레이어 12·13 까지만 흐른다.
액션 손실은 16 을 읽으므로 12~15 전부로 흐른다.
따라서 게이트가 정말 영향을 줬다면 12·13 이 14·15 보다 더 많이 갈라져야 한다.
14·15 는 두 실행 모두 액션 손실만 받으므로 대조군이 된다.
"""
import json, os, sys, numpy as np, torch
from safetensors import safe_open

WS = os.path.expanduser("~/quantization_agent_workspace")
A = f"{WS}/assets/checkpoints/n17_robocasa_gate"
B = os.environ.get("BASELINE", f"{WS}/assets/checkpoints/n17_robocasa_baseline_top4")

def index(d):
    return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]

import glob
BASE_SNAP = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*")))[0]
ia, ib = index(A), index(B)
ic = index(BASE_SNAP)
common = sorted(set(ia) & set(ib))
print(f"게이트본 텐서 {len(ia)} · 베이스라인 {len(ib)} · 공통 {len(common)}")
only_a = sorted(set(ia) - set(ib))
print(f"게이트본에만 있는 텐서 {len(only_a)}개" + (f" 예: {only_a[:3]}" if only_a else ""))

handles = {}
def get(d, idx, name):
    f = idx[name]
    k = (d, f)
    if k not in handles:
        handles[k] = safe_open(f"{d}/{f}", framework="pt")   # 원본은 bf16 — numpy 로는 못 읽는다
    return handles[k].get_tensor(name).float().numpy()

# 원본 사전학습본을 기준으로 각 실행이 얼마나 움직였는지 본다.
# 서로를 비교하면 게이트 효과와 학습 노이즈를 못 가르고, 위층으로의 전파도 섞인다.
ga, gb = {}, {}
for n in common:
    if ".layers." not in n or n not in ic:
        continue
    li = int(n.split(".layers.")[1].split(".")[0])
    if li < 12:
        continue
    base = get(BASE_SNAP, ic, n)
    nb = np.linalg.norm(base) + 1e-12
    ga.setdefault(li, []).append(np.linalg.norm(get(A, ia, n) - base) / nb)
    gb.setdefault(li, []).append(np.linalg.norm(get(B, ib, n) - base) / nb)

print(f"\n원본 대비 이동량 (클수록 많이 학습됨)")
print(f"{'레이어':>6s} {'게이트본':>10s} {'베이스라인':>11s} {'차이':>9s}   비고")
for li in sorted(ga):
    a_, b_ = np.mean(ga[li]), np.mean(gb[li])
    tag = "게이트+액션" if li in (12, 13) else "액션만"
    print(f"{li:6d} {a_:10.6f} {b_:11.6f} {a_-b_:+9.6f}   {tag}")
print("\n게이트가 백본을 실제로 형성했다면 12·13 에서 게이트본이 더 많이 움직여야 한다.")
