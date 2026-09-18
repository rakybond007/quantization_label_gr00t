"""openarm 의 분위 경계. **전부 이 데이터셋 자기 분포에서 낸다.**

손으로 적은 상수를 쓰면 그것이 곧 "내가 추측한 한계" 가 된다. human_data 에서
`grip_closed` 만 봐도 데이터셋마다 0.38~0.82 로 달랐다.

openarm 은 양팔(각 7관절) + 양손(각 6관절) 절대값이라 팔과 손을 따로 잰다.

  python openarm_quantiles.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data/openarm/lerobot"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "analysis", "openarm", "quantiles.json")
H = 16
SL = {"left_arm": (0, 7), "right_arm": (7, 14),
      "left_hand": (14, 20), "right_hand": (20, 26)}


def main():
    task = {}
    for ln in open(f"{ROOT}/meta/episodes.jsonl"):
        d = json.loads(ln)
        t = [x for x in d.get("tasks", []) if isinstance(x, str) and len(x.split()) > 3]
        if t:
            task[d["episode_index"]] = "dustpan" if "dustpan" in t[0] else "scrub"
    res = {}
    for grp in ("dustpan", "scrub"):
        eps = {e for e, g in task.items() if g == grp}
        acc = {k: [] for k in SL}
        dem = {2: [], 3: []}
        ratio = []
        hand_vals = {"left_hand": [], "right_hand": []}
        for f in sorted(glob.glob(f"{ROOT}/data/*/*.parquet")):
            ep = int(f.split("episode_")[1].split(".")[0])
            if ep not in eps:
                continue
            a = np.stack(pd.read_parquet(f)["action"].values).astype(float)
            arms = a[:, 0:14]
            for k, (lo, hi) in SL.items():
                seg = a[:, lo:hi]
                st = np.linalg.norm(np.diff(seg, axis=0), axis=1)
                for s in range(0, max(len(seg) - H, 0), H):
                    acc[k].append(float(st[s:s + H - 1].mean()))
                if k.endswith("hand"):
                    hand_vals[k].append(seg.mean(1))
            for K in (2, 3):
                for s in range(0, max(len(arms) - H, 0), H):
                    w = arms[s:s + H]
                    if len(w) > K:
                        dem[K].append(float(np.linalg.norm(w[K:] - w[:-K], axis=1).max()))
            st = np.linalg.norm(np.diff(arms, axis=0), axis=1)
            for s in range(0, max(len(arms) - H, 0), H):
                seg = st[s:s + H - 1]
                if len(seg) < 4:
                    continue
                h2 = len(seg) // 2
                v0 = seg[:h2].mean()
                ratio.append(float(seg[h2:].mean() / v0) if v0 > 1e-9 else 1.0)
        r = {}
        for k in SL:
            v = np.array(acc[k])
            r[f"{k}_tertile"] = [round(float(np.percentile(v, q)), 5) for q in (33, 67)]
        r.update({f"k{K}": [round(float(np.percentile(dem[K], q)), 4) for q in (20, 40, 60, 80)]
                  for K in (2, 3)})
        r["n_windows"] = len(dem[2])
        r["decel_q25"] = round(float(np.percentile(np.array(ratio), 25)), 4)
        # 손 닫힘 문턱: 0~1 양봉의 골. 한쪽 손이 거의 안 움직이면(수세미의 오른손)
        # 골이 없으므로 그 경우는 중앙값을 쓴다 -- 그 손은 어차피 "닫힘" 판정을 안 한다.
        for k in ("left_hand", "right_hand"):
            g = np.concatenate(hand_vals[k])
            h, e = np.histogram(g, bins=25, range=(float(g.min()), float(g.max()) + 1e-9))
            mid = e[1:] - (e[1] - e[0]) / 2
            lo_pk = int(np.argmax(h[:8])); hi_pk = 8 + int(np.argmax(h[8:]))
            th = float(mid[lo_pk + int(np.argmin(h[lo_pk:hi_pk + 1]))]) if hi_pk > lo_pk else float(np.median(g))
            r[f"{k}_closed"] = round(th, 3)
            r[f"{k}_range"] = [round(float(g.min()), 3), round(float(g.max()), 3)]
        res[grp] = r
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
