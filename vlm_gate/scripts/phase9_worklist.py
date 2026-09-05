"""Work out what is left to label, and split it across N workers up front.

The labeller used to shard by `episode % nshard` and skip what its own shard
file already held. That ties the shard count to the files on disk: changing it
renames the files, the new worker reads none of the old ones, and a run that was
half done starts over. It also means a worker cannot know how much is left until
it has walked its episodes.

So the remaining work is computed once, here, and written out. The frame count
per episode comes from `meta/episodes.jsonl` (`length`), so nothing has to be
decoded; the done set is read from every `labels_*.jsonl` in the output
directory regardless of what shard naming produced it. Episodes are then dealt
out to workers by remaining frames, largest first, which balances them even
though the episodes are unevenly finished.

    python phase9_worklist.py <n_workers> [out_dir]
"""
import glob
import json
import os
import sys

BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/output/_gate_distill/phase9_full"

# The labeller stops 4 frames short of the action array, so that is the universe.
TAIL = 4
total = {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    e = json.loads(line)
    total[e["episode_index"]] = max(0, e["length"] - TAIL)

done = {}
files = sorted(glob.glob(f"{OUT}/labels_*.jsonl"))
ndone = 0
for f in files:
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        done.setdefault(r["ep"], set()).add(r["f"])
        ndone += 1
print(f"기존 라벨 {len(files)}개 파일, {ndone:,}행")

left = {ep: n - len(done.get(ep, ())) for ep, n in total.items()}
left = {ep: v for ep, v in left.items() if v > 0}
print(f"전체 {sum(total.values()):,} 프레임 중 남은 것 {sum(left.values()):,} "
      f"({100*sum(left.values())/sum(total.values()):.1f}%), 에피소드 {len(left)}개")

# 남은 양이 많은 에피소드부터 가장 한가한 일꾼에게
buckets = [[] for _ in range(N)]
load = [0] * N
for ep in sorted(left, key=lambda e: -left[e]):
    i = min(range(N), key=lambda k: load[k])
    buckets[i].append(ep)
    load[i] += left[ep]

os.makedirs(OUT, exist_ok=True)
for i, eps in enumerate(buckets):
    p = f"{OUT}/worklist_{N}_{i}.json"
    json.dump({"episodes": sorted(eps),
               "done": {str(ep): sorted(done.get(ep, ())) for ep in eps},
               "n_left": load[i]}, open(p, "w"))
    print(f"  일꾼 {i}: 에피소드 {len(eps):4d}개, 남은 프레임 {load[i]:,}  -> {os.path.basename(p)}")
