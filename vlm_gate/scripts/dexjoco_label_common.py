"""Shared constants for the DexJoCo labelling harness.

The one thing worth reading here is the episode key.

RoboCasa has ONE episode-index space, so `(ep, f)` identifies a chunk and the
verifier (`qgate labels`) can count duplicates on it.  DexJoCo is six separate
datasets, each numbering its episodes 0..99.  Writing the local index would make
water_plant ep7 and hammer_nail ep7 the same row: the resume set would skip work
that was never done, and the duplicate check would report collisions that are
really different episodes.

So the record's `ep` is GLOBAL:

    ep = TASK_OFFSET * task_id + ep_local            (TASK_OFFSET = 1000)

with `task_id` the fixed index of the task in TASK_ORDER (alphabetical, frozen
below -- never reorder it, the labels on disk depend on it).  100 episodes per
task fit inside the 1000 stride with room to spare, and the mapping is
invertible, so `ep // 1000 -> task`, `ep % 1000 -> episode inside the task`.
The record also carries `task` and `ep_local` explicitly so nothing downstream
has to know the arithmetic.

Sharding stays `ep % NSH == SHARD` on the GLOBAL ep, exactly like RoboCasa.
"""
import os

BASE = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"

# frozen; the `ep` values already written to disk are defined by this order
TASK_ORDER = ["click_mouse", "fold_glasses", "hammer_nail",
              "pick_bucket", "pinch_tongs", "water_plant"]
TASK_OFFSET = 1000
TASK_ID = {t: i for i, t in enumerate(TASK_ORDER)}

DEFAULT_TILES = f"{BASE}/output/_gate_distill/dexjoco_v1"
DEFAULT_MANIFEST = f"{DEFAULT_TILES}/tiles_manifest.txt"

# 640x640 @ 30 fps, two views.  STRIDE 16 = one non-overlapping descriptor window
# (`descriptors(a, f, n=16)` = ~0.53 s), so chunks tile the episode without gaps
# or overlap; 13,348 chunks over the six tasks, the same order of magnitude as
# the RoboCasa pilot.  DOWNSCALE 2 -> 320 px per view (tile 640x320), twice
# RoboCasa's per-view 128 px because the dexterous hand's finger configuration
# is the thing being judged.
# 전 프레임 라벨링이 기본. 이유는 gen_libero_tiles_shard.py 의 같은 자리에 있다.
STRIDE = int(os.environ.get("TILE_STRIDE", "1"))
TILE_VIEW_PX = 320          # per-view square size after downscale (640 -> 320)
CHUNK_N = 16                # descriptor window length, in control steps
TAIL = 16                   # frames at the end of an episode with no full chunk


def global_ep(task, ep_local):
    return TASK_OFFSET * TASK_ID[task] + int(ep_local)


def split_ep(ep):
    """global ep -> (task, ep_local)"""
    return TASK_ORDER[int(ep) // TASK_OFFSET], int(ep) % TASK_OFFSET


def parse_manifest_line(line):
    """'water_plant/ep0007_f00080.png' -> (task, ep_local, f, name)"""
    line = line.strip()
    task, nm = line.split("/", 1)
    stem = os.path.basename(nm).rsplit(".", 1)[0]
    ep, f = stem.split("_f")
    return task, int(ep[2:]), int(f), nm
