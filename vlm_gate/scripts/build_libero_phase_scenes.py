"""LIBERO 국면균등 장면. robocasa build_phase_balanced_scenes.py 의 LIBERO 판.

국면은 계산만으로 나온다(`libero_phase.phase_of`). LIBERO 는 robocasa 보다 국면이
자연스럽게 균형 잡혀 있지만(파지 12.9% 대 robocasa 3%), 그래도 할당량을 정해 뽑아야
파지·해제가 200개 이상 확보된다.

    python build_libero_phase_scenes.py [태스크당_국면당] [출력경로]

한 에피소드가 한 국면을 두 개까지만 차지하게 막는다 -- 태스크 안 변동이 아니라 그
에피의 특성을 재게 되기 때문이다.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from libero_phase import phase_of                      # noqa: E402

BASE = os.path.dirname(HERE)
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "libero_gr00t_delta")
TILES = f"{BASE}/output/_gate_distill/libero_full/tiles"
MAN = f"{BASE}/output/_gate_distill/libero_tiles_manifest.txt"
IDX = f"{BASE}/analysis/libero_task_index.json"
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/_tmp/libero/scenes_phase.json"
WANT = ("접근", "파지", "해제", "운반", "들어올림", "내려놓기")
CH = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)


def main():
    # 타일 매니페스트 -- 공유 마운트에서 재귀 탐색을 하지 않는다
    have = set()
    if os.path.exists(MAN):
        for l in open(MAN):
            nm = l.strip()
            if nm:
                have.add(nm[:-4] if nm.endswith(".png") else nm)
    else:
        have = {f[:-4] for f in os.listdir(TILES) if f.endswith(".png")}
    print(f"타일 {len(have):,}장")

    idx = json.load(open(IDX))
    i2t = {v: f"{s}/{k}" for s, d in idx.items() for k, v in d.items()}
    ep2cell, ep2instr = {}, {}
    for l in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(l)
        ts = d.get("tasks") or []
        if ts and ts[0] in i2t:
            ep2cell[d["episode_index"]] = i2t[ts[0]]
            ep2instr[d["episode_index"]] = ts[0]
    byc = defaultdict(list)
    for ep, c in ep2cell.items():
        byc[c].append(ep)
    print(f"평가 셀 {len(byc)}개에 매핑된 에피 {len(ep2cell)}")

    rng = np.random.default_rng(11)
    out, stat = [], defaultdict(int)
    for cell in sorted(byc):
        eps = np.array(byc[cell])
        rng.shuffle(eps)
        got = defaultdict(list)
        for ep in eps:
            if all(len(got[p]) >= PER for p in WANT):
                break
            try:
                a = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ep // CH:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            except Exception:
                continue
            for f in range(0, max(len(a) - 20, 0), 4):
                nm = f"ep{ep:04d}_f{f:03d}"
                if nm not in have:
                    continue
                p = phase_of(a, f)
                if p in WANT and len(got[p]) < PER and \
                        sum(1 for x in got[p] if x["ep"] == ep) < 2:
                    got[p].append({"cell": cell, "task": cell, "ep": int(ep),
                                   "f": f, "tile": nm, "phase": p,
                                   "instruction": ep2instr[ep]})
        for p in WANT:
            out += got[p]
            stat[p] += len(got[p])
        print(f"  {cell:<20} " + " ".join(f"{p}{len(got[p]):>2}" for p in WANT))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    d = pd.DataFrame(out)
    print(f"\n장면 {len(out)} -> {OUT}")
    print("  국면별: " + "  ".join(f"{p}={stat[p]}" for p in WANT))
    if len(d):
        print(f"  셀 {d.cell.nunique()} · 에피 {d.ep.nunique()}"
              f" · 셀당 {d.groupby('cell').size().min()}~{d.groupby('cell').size().max()}")


if __name__ == "__main__":
    main()
