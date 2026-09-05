"""주기에서 물체를 채워 넣은 서브태스크 주석을 **파일로 박는다.**

주석은 회전에만 물체를 적는다(Rotate Box / Rotate PolyBag). 나머지는 "Object"
라고만 쓴다. 그런데 에피소드 안의 순서가 나머지를 알려준다 -- 한 주기가
가져오기 → (필요하면 뒤집기) → 넘기기이고, 그 주기에 낀 회전이 그 주기의
물체를 말한다.

    ep0   B rBOX P   B P   B rBOX P   B rBOX P   B rBAG P   B

이걸 코드 안에서 매번 다시 유도하면 잊어버리고 "주석에 물체가 없다" 고 다시
말하게 된다. 한 번 만들어 파일로 두고, 이후에는 이 파일만 읽는다.

    python allex_make_subtask_object.py [데이터셋]
    -> <데이터셋>/meta/subtasks_with_object.jsonl   (원본은 안 건드린다)
"""
import collections
import json
import os
import sys

DS = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/replay_evaluation10/replay_evaluation_ee_subtask")
OUT = os.environ.get("ALLEX_SUBTASK_FILE", f"{DS}/meta/subtasks_with_object.jsonl")

segs = [json.loads(l) for l in open(f"{DS}/meta/subtasks.jsonl")]
by = collections.defaultdict(list)
for s in segs:
    by[s["episode_index"]].append(s)
for v in by.values():
    v.sort(key=lambda x: x["start_frame"])

# 주석의 말을 그대로 쓴다. Rotate 를 turn 으로, Bring/Pass 를 move 로 바꿔
# 쓴 판이 있었는데 그건 내 임의였고, 특히 Bring 과 Pass 는 서로 다른
# 서브태스크인데 하나로 묶어버렸다.
OBJ = {"Rotate Box": "Box", "Rotate PolyBag": "PolyBag"}
ACT = {"Rotate Box": "Rotate", "Rotate PolyBag": "Rotate",
       "Bring Object": "Bring", "Pass Object": "Pass"}
obj = {}
for v in by.values():
    for x in v:
        if x["label"] in OBJ:
            obj[x["id"]] = OBJ[x["label"]]
    i = 0
    while i < len(v):
        if v[i]["label"] != "Bring Object":
            i += 1
            continue
        j, mid = i + 1, []
        while j < len(v) and v[j]["label"] not in ("Bring Object", "Pass Object"):
            mid.append(v[j]["label"])
            j += 1
        if j < len(v) and v[j]["label"] == "Pass Object":
            o = {OBJ[m] for m in mid if m in OBJ}
            if len(o) == 1:
                got = o.pop()
                obj[v[i]["id"]] = got
                obj[v[j]["id"]] = got
            i = j + 1
        else:
            i = j

n = collections.Counter()
with open(OUT, "w") as fh:
    for s in segs:
        o = obj.get(s["id"])
        a = ACT[s["label"]]
        s2 = dict(s)
        s2["object"] = o                    # "Box" / "PolyBag" / None
        s2["action"] = a                    # "Bring" / "Pass" / "Rotate"
        s2["cell"] = f"{a} {o}" if o else None
        n[s2["cell"]] += 1
        fh.write(json.dumps(s2, ensure_ascii=False) + "\n")

print(f"  {len(segs)}구간 -> {OUT}")
for k, v in sorted(n.items(), key=lambda kv: (kv[0] is None, kv[0])):
    print(f"    {str(k):<12} {v}구간")
