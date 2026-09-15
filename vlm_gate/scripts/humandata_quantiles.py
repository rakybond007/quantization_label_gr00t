"""human_data 두 데이터셋의 병합 점프 분위대 경계. 각자 자기 분포에서 낸다."""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "human_data", "quantiles.json")
H = 16
res = {}
for d in ("pnp_task", "long_horizon_task"):
    dem = {2: [], 3: []}; spd = []; ratio = []; grips = []
    for f in sorted(glob.glob(f"{ROOT}/{d}/lerobot/data/*/*.parquet")):
        J = np.stack(pd.read_parquet(f)["action"].values)[:, :7]
        A = np.stack(pd.read_parquet(f)["action"].values)
        grips.append(A[:, 7])
        for K in (2, 3):
            for s in range(0, max(len(J) - H, 0), H):
                w = J[s:s + H]
                if len(w) > K:
                    dem[K].append(float(np.linalg.norm(w[K:] - w[:-K], axis=1).max()))
        for s in range(0, max(len(J) - H, 0), H):
            st = np.linalg.norm(np.diff(J[s:s + H], axis=0), axis=1)
            if len(st) < 4:
                continue
            spd.append(float(st.mean()))
            h2 = len(st) // 2
            v0 = st[:h2].mean()
            ratio.append(float(st[h2:].mean() / v0) if v0 > 1e-9 else 1.0)
    res[d] = {f"k{K}": [round(float(np.percentile(dem[K], q)), 4) for q in (20, 40, 60, 80)]
              for K in (2, 3)}
    res[d]["n_windows"] = len(dem[2])
    # **문턱은 전부 이 데이터셋 자기 분포에서 낸다.** 손으로 적은 상수를 쓰면
    # 그것이 곧 "내가 추측한 한계" 가 된다. 속도 형용사 3단은 창 평균 스텝의
    # 3분위, 감속·저속유지 판정도 같은 분포의 분위로 잡는다.
    sp = np.array(spd); res[d]["speed_tertile"] = [round(float(np.percentile(sp, q)), 5)
                                                   for q in (33, 67)]
    rt = np.array(ratio); res[d]["decel_q25"] = round(float(np.percentile(rt, 25)), 4)
    gg = np.concatenate(grips)
    # 그리퍼 문턱: 0~1 이 양봉이라 두 봉우리 사이 골을 쓴다(히스토그램 최소).
    h, e = np.histogram(gg, bins=25, range=(0, 1))
    mid = e[1:] - (e[1] - e[0]) / 2
    lo_pk, hi_pk = int(np.argmax(h[:8])), 8 + int(np.argmax(h[8:]))
    res[d]["grip_closed"] = round(float(mid[lo_pk + int(np.argmin(h[lo_pk:hi_pk + 1]))]), 3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
