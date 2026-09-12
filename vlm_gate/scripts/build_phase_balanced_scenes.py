"""국면별로 고르게 뽑은 장면 집합. **파지·해제가 얇아서 판정을 못 믿던 것을 고친다.**

태스크당 무작위로 뽑으면 파지(그리퍼가 닫히는 창)와 해제가 720 장면 중 52개밖에
안 나온다 -- 전체의 7% 다. 그 52개에 맞춰 문항을 깎으면 과적합이다. 국면 정의는
계산만으로 나오므로(judge_ab.phase_of) 국면별 할당량을 정해 채우면 된다.

    python build_phase_balanced_scenes.py [태스크당_국면당] [출력경로]

기본 10 이면 24 태스크 x 4 국면 x 10 = 최대 960 장면, 파지·해제가 각 200 이상 된다.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robocasa_descriptors import descriptors          # noqa: E402
from judge_ab import phase_of, vert_of                # noqa: E402

BASE = os.path.dirname(HERE)
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full"
LAB = f"{BASE}/output/_gate_distill/robocasa_contact_ratio_instr.parquet"
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/_tmp/selfagg_bias/scenes_phase.json"
# 전이를 파지/해제로 쪼갠 6단. "전이(둘다)" 와 "놓기" 는 드물어 할당량을 두지 않고
# 걸리는 대로 받는다.
WANT = ("접근", "파지", "해제", "운반")
CH = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)


def fine_phase(a, f):
    x = descriptors(a, f)
    x["vert"] = vert_of(a, f)
    p = phase_of(x)
    if p == "전이":
        return ("파지" if (x["grip_close"] and not x["grip_open"]) else
                "해제" if (x["grip_open"] and not x["grip_close"]) else "전이(둘다)")
    return p


def main():
    man = []
    for s in range(3):
        man += json.load(open(f"{TILES}/manifest_shard{s}.json"))
    m = pd.DataFrame(man).rename(columns={"ep": "episode_index", "f": "frame_index"})
    m["tile"] = m.path.str.replace(r".*/", "", regex=True).str.replace(".png", "", regex=False)
    lab = pd.read_parquet(LAB, columns=["episode_index", "frame_index", "task"])
    j = m.merge(lab, on=["episode_index", "frame_index"], how="inner")
    print(f"타일 ∩ 라벨 {len(j):,} · 태스크 {j.task.nunique()}")

    rng = np.random.default_rng(11)
    out, stat = [], defaultdict(int)
    for task, g in j.groupby("task"):
        got = defaultdict(list)
        eps = g.episode_index.unique()
        rng.shuffle(eps)
        for ep in eps:
            if all(len(got[p]) >= PER for p in WANT):
                break
            try:
                a = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ep // CH:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            except Exception:
                continue
            rows = g[g.episode_index == ep]
            for r in rows.itertuples():
                f = int(r.frame_index)
                if f + 20 >= len(a):
                    continue
                p = fine_phase(a, f)
                # 같은 에피소드에서 한 국면을 두 개까지만 -- 한 에피가 국면을 독점하면
                # 태스크 안 변동이 아니라 그 에피의 특성을 재게 된다
                if p in WANT and len(got[p]) < PER and \
                        sum(1 for x in got[p] if x["ep"] == ep) < 2:
                    got[p].append({"task": task, "ep": int(ep), "f": f, "tile": r.tile,
                                   "phase": p})
        for p in WANT:
            out += got[p]
            stat[p] += len(got[p])
        print(f"  {task:24s} " + " ".join(f"{p}{len(got[p]):>3}" for p in WANT))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"\n장면 {len(out)} -> {OUT}")
    print("  국면별: " + "  ".join(f"{p}={stat[p]}" for p in WANT))
    d = pd.DataFrame(out)
    print(f"  에피소드 {d.ep.nunique()} · 태스크당 {d.groupby('task').size().min()}"
          f"~{d.groupby('task').size().max()}")


if __name__ == "__main__":
    main()
