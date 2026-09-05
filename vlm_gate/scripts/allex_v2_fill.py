"""건너뛰며 라벨링한 결과를 모든 청크로 채운다.

stride 로 라벨링하면 청크의 1/stride 만 값을 갖는다. 나머지는 **같은 서브태스크
구간 안에서** 가장 가까운 라벨을 받는다. 구간을 넘어 가져오지 않는다 -- 상한이
구간마다 다르므로 경계를 넘으면 다른 상한의 값이 새어 들어온다.

이웃이 양쪽에 있으면 **낮은 쪽**을 준다. 배속은 넘치면 에피소드를 잃고 밑돌면
속도만 못 챙기는 비대칭이라, 모르는 자리는 느린 쪽으로 둔다.

  python allex_v2_fill.py            # ALLEX_OUT/records.jsonl -> records_full.jsonl
"""
import json
import os

DS = os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/v5_matched/merged_v5tempo")
OUT = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v5tempo"))
CHUNK = 16
SRC = f"{OUT}/records.jsonl"
DST = f"{OUT}/records_full.jsonl"

info = json.load(open(f"{DS}/meta/info.json"))
CH = int(info["chunks_size"])
lens = {}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(l)
    lens[d["episode_index"]] = d["length"]

segs = {}
for l in open(f"{DS}/meta/subtasks.jsonl"):
    r = json.loads(l)
    segs.setdefault(r["episode_index"], []).append(
        (r["start_frame"], r["end_frame"], r["label"]))
for v in segs.values():
    v.sort()


def seg_of(ep, f):
    mid = f + CHUNK // 2
    for i, (a, b, _) in enumerate(segs.get(ep, ())):
        if a <= mid < b:
            return i
    return -1


have = {}
for l in open(SRC):
    r = json.loads(l)
    have.setdefault(r["ep"], {})[r["f"]] = r

n_lab = sum(len(v) for v in have.values())
n_out = n_copy = n_orphan = 0
with open(DST, "w") as fh:
    for ep in sorted(lens):
        N = lens[ep]
        mine = have.get(ep, {})
        if not mine:
            continue
        # 라벨이 있는 자리를 구간별로 모아 둔다
        by_seg = {}
        for f in sorted(mine):
            by_seg.setdefault(seg_of(ep, f), []).append(f)
        for f in range(0, N - CHUNK, CHUNK):
            if f in mine:
                fh.write(json.dumps(mine[f]) + "\n"); n_out += 1
                continue
            si = seg_of(ep, f)
            cand = by_seg.get(si)
            if not cand:
                n_orphan += 1
                continue
            d = min(abs(c - f) for c in cand)
            near = [c for c in cand if abs(c - f) == d]
            # 같은 거리에 둘이면 느린 쪽
            src = min(near, key=lambda c: mine[c]["K"])
            r = dict(mine[src])
            r["ep"], r["f"] = ep, f
            r["filled_from"] = src
            fh.write(json.dumps(r) + "\n"); n_out += 1; n_copy += 1

print(f"라벨된 청크 {n_lab}  ->  채운 뒤 {n_out}  (복사 {n_copy}, "
      f"구간에 라벨이 없어 버린 것 {n_orphan})")
print(f"-> {DST}")
