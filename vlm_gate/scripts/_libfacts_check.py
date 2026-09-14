"""사실 문장이 **실제** libero 액션에서 어떻게 갈리는지. 한 문장에 붙박이면 사실이 아니다."""
import sys, os, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_descriptors import descriptors, facts_text

DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
import glob
pq = sorted(glob.glob(f"{DS}/data/**/*.parquet", recursive=True))
print("파케이 파일", len(pq))
import pandas as pd
acts, eps = [], 0
for f in pq[:40]:
    d = pd.read_parquet(f)
    col = [c for c in d.columns if "action" in c.lower()]
    if not col:
        print("액션 열 없음:", list(d.columns)[:8]); break
    for ep, g in d.groupby("episode_index"):
        a = np.stack(g[col[0]].to_numpy())
        if a.shape[0] >= 32:
            acts.append(a); eps += 1
    if eps >= 60:
        break
print(f"에피소드 {eps}, 액션 차원 {acts[0].shape if acts else '-'}")

c, nwin = collections.Counter(), 0
for a in acts:
    for f in range(0, a.shape[0] - 16, 8):
        t = facts_text(descriptors(a, f=f))
        nwin += 1
        for part in t.split(": ", 1)[1].rstrip(".").split("; "):
            c[part.split(" (")[0]] += 1
print(f"\n창 {nwin}개")
for k, v in c.most_common():
    print(f"  {v/nwin:6.1%}  {k}")
print("\n--- 예시 셋")
for a in acts[:3]:
    print(" ", facts_text(descriptors(a, f=16)))
