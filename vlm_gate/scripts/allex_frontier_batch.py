"""상위 모델(Claude)을 판정기로 쓰는 배치를 만든다.

**파이프라인은 그대로다.** 지금 Cosmos 판정기가 받는 것과 같은 것을 받는다 --
그림 + 계산 사실 + GUIDANCE + 문항 -> 등급. 바뀌는 것은 판정기뿐이다.
논문에서 "오픈소스 VLM 대신 프런티어 모델을 판정기로 두면 어떻게 되는가" 를
말할 수 있게, 비교가 성립하도록 입력을 맞춘다.

계산 사실은 **이 목적에 맞게 다시 쓴다**. 지금 문구("팔이 보통 속도로 움직인다")
는 판정기가 등급으로 옮기기 어렵다. 압축이 무엇을 요구하는지 -- 건너뛸 때 한
스텝에 얼마를 움직여야 하고 그것이 시연 한계에 얼마나 가까운지 -- 를 결론까지
계산해서 준다.

    python allex_frontier_batch.py <에피> <stride> [개수]
        -> _tmp/frontier/sheet_<ep>.png   그림 격자
           _tmp/frontier/facts_<ep>.md    청크마다 계산 사실
           _tmp/frontier/meta_<ep>.json   답을 되붙일 정보
"""
import json, os, sys
import numpy as np, pandas as pd, av
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
from allex_v2_common import descriptors
import allex_v2_common as CM
import allex_v3_checks as TH   # 문턱값은 v3 문항 모듈이 갖고 있다. 같은 값을 쓴다.

D = os.environ.get("ALLEX_DS", "/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
EP = int(sys.argv[1]); STRIDE = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
OUT = f"{BASE}/_tmp/frontier"; os.makedirs(OUT, exist_ok=True)


from allex_facts import facts



d = pd.read_parquet(f"{D}/data/chunk-{EP//1000:03d}/episode_{EP:06d}.parquet")
A = np.stack(d["action"].values)
WR = np.stack(d["action.right_wrist_wrt_base"].values)
WL = np.stack(d["action.left_wrist_wrt_base"].values)
TASK = json.loads(open(f"{D}/meta/tasks.jsonl").readline())["task"]
fr = list(range(0, len(A) - 16, STRIDE))[:N]

want = set(fr); got = {}
with av.open(f"{D}/videos/chunk-{EP//1000:03d}/observation.images.camera_ego_left/"
             f"episode_{EP:06d}.mp4") as c:
    for i, f2 in enumerate(c.decode(video=0)):
        if i in want:
            got[i] = f2.to_image().resize((200, 200))
            if len(got) == len(want): break

tiles, lines, meta = [], [], []
for n, f in enumerate(fr):
    if f not in got: continue
    x = descriptors(A, WR, WL, f, 16)
    im = got[f].copy(); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 40, 16], fill=(0, 0, 0)); dr.text((4, 3), f"#{n+1}", fill=(255, 255, 0))
    tiles.append(im)
    lines.append(f"**#{n+1}**  (ep{EP} f{f})\n{facts(x)}\n")
    meta.append({"ep": EP, "f": f, "n": n + 1,
                 "one_handed": bool(x["one_handed"]),
                 "merge_demand_k2": round(float(x["merge_demand_k2"]), 4)})

cols = 4; rws = (len(tiles) + cols - 1) // cols
sh = Image.new("RGB", (cols * 200, rws * 200), (20, 20, 20))
for i, t in enumerate(tiles): sh.paste(t, ((i % cols) * 200, (i // cols) * 200))
sh.save(f"{OUT}/sheet_{EP}.png")
open(f"{OUT}/facts_{EP}.md", "w").write(
    f"# ep{EP} · stride {STRIDE} · {len(tiles)}청크\n\n"
    f"지시문: \"{TASK}\"\n\n" + "\n".join(lines))
json.dump(meta, open(f"{OUT}/meta_{EP}.json", "w"))
print(f"{len(tiles)}청크 -> sheet_{EP}.png · facts_{EP}.md · meta_{EP}.json")
