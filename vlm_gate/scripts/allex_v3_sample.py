"""개발집합 D 와 보류집합 H 를 층화해서 뽑는다. PROMPT_LOOP.md 의 D/H 규칙.

여기서 뽑아 파일로 남기는 이유는 하나다. 그때그때 "에피소드 0~3" 을 고르면
층별 검정력이 6배씩 차이나고(Rotate PolyBag 41 vs Rotate Box 308), 그런 표본
위에서 나온 "약함" 은 문항이 약한 건지 표본이 작은 건지 알 수 없다.

층은 손상표의 여섯 칸이다. 주석은 회전에만 물체를 적지만(Rotate Box /
Rotate PolyBag) 에피소드 안의 순서가 나머지를 알려준다 -- 한 주기가
가져오기 → (필요하면 뒤집기) → 넘기기 이고, 그 주기에 낀 회전이 그 주기의
물체를 말한다.

    ep0   B rBOX P   B P   B rBOX P   B rBOX P   B rBAG P   B

완성된 주기 393개 중 222개에 Rotate Box 가, 84개에 Rotate PolyBag 이 끼어
있다. 나머지 87개(22.1%)는 뒤집을 필요가 없어 건너뛴 주기라 물체를 알 수
없고, 그 청크는 D/H 에서 빠진다 -- 층을 모르는 표본은 층 간 대비를 재는 데
쓸 수 없다.

    python allex_v3_sample.py [층당 개수]
"""
import json
import os
import random
import sys

DS = os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin")
OUT = os.path.expanduser(os.environ.get(
    "ALLEX_SAMPLE", "~/quantization_agent_workspace/vlm_gate/output/allex_sample"))
CHUNK = 16
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(os.environ.get("ALLEX_SEED", 0))
# 한 에피소드가 한 층을 채워버리면 조명·물체·사람이 그 몇 개에 맞춰진다.
CAP = max(2, PER // 10)
os.makedirs(OUT, exist_ok=True)

segs = [json.loads(l) for l in open(f"{DS}/meta/subtasks.jsonl")]
by_ep = {}
for x in segs:
    by_ep.setdefault(x["episode_index"], []).append(x)
for v in by_ep.values():
    v.sort(key=lambda x: x["start_frame"])

# 주기의 회전이 그 주기의 물체를 정한다. 회전이 없으면 정하지 않는다.
# 층 이름은 TASK_RANGE 의 칸 이름과 글자 그대로 같아야 한다. 주석이 쓰는 말을
# 그대로 쓴다 -- Bring / Pass / Rotate x Box / PolyBag.
OBJ = {"Rotate Box": "Box", "Rotate PolyBag": "PolyBag"}
cell = {}                       # id(seg) -> (행동, 물체)
unknown = 0
for v in by_ep.values():
    for x in v:
        if x["label"] in OBJ:
            cell[x["id"]] = ("Rotate", OBJ[x["label"]])
    i = 0
    while i < len(v):
        if v[i]["label"] != "Bring Object":
            i += 1
            continue
        j2, mid = i + 1, []
        while j2 < len(v) and v[j2]["label"] not in ("Bring Object", "Pass Object"):
            mid.append(v[j2]["label"])
            j2 += 1
        if j2 < len(v) and v[j2]["label"] == "Pass Object":
            objs = {OBJ[m] for m in mid if m in OBJ}
            if len(objs) == 1:
                o = objs.pop()
                cell[v[i]["id"]] = ("Bring", o)
                cell[v[j2]["id"]] = ("Pass", o)
            else:
                unknown += 2
            i = j2 + 1
        else:
            unknown += 1
            i = j2

chunks = {}
for s_ in segs:
    c = cell.get(s_["id"])
    if c is None:
        continue
    a = (s_["start_frame"] // CHUNK) * CHUNK
    for f in range(a, s_["end_frame"] - CHUNK, CHUNK):
        chunks.setdefault(f"{c[0]} {c[1]}", []).append((s_["episode_index"], f))
print(f"  물체를 못 정한 구간 {unknown}개는 제외")

rng = random.Random(SEED)
D, H = {}, {}
for lab in sorted(chunks):
    pool = sorted(set(chunks[lab]))
    rng.shuffle(pool)
    by_ep, taken = {}, []
    for ep, f in pool:                       # 에피소드 상한을 지키며 채운다
        if by_ep.get(ep, 0) >= CAP:
            continue
        by_ep[ep] = by_ep.get(ep, 0) + 1
        taken.append((ep, f))
        if len(taken) >= 2 * PER:
            break
    if len(taken) < 2 * PER:                 # 상한 때문에 모자라면 상한을 푼다
        for ep, f in pool:
            if (ep, f) not in taken:
                taken.append((ep, f))
            if len(taken) >= 2 * PER:
                break
    D[lab], H[lab] = taken[:PER], taken[PER:2 * PER]
    print(f"  {lab:<12} 풀 {len(pool):5d}  D {len(D[lab]):4d}  H {len(H[lab]):4d}  "
          f"에피소드 {len(set(e for e, _ in taken)):3d}개")

for name, split in (("D", D), ("H", H)):
    path = f"{OUT}/{name}.json"
    json.dump({"per_stratum": PER, "seed": SEED, "cap_per_episode": CAP,
               "strata": {k: [list(x) for x in v] for k, v in split.items()}},
              open(path, "w"))
    n = sum(len(v) for v in split.values())
    print(f"{name}: {n} 청크 -> {path}")
