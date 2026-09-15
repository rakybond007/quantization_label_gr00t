"""allex v4 라벨러 -- 서브태스크 라벨 없이, 지시문만 받아 청크마다 conf 를 낸다.

`allex_v3_label.py` 와 두 가지가 다르다.

**서브태스크 라벨을 안 쓴다.** v3 는 `meta/subtasks.jsonl` 로 셀을 정하고 셀마다
상한을 조회해 K 를 놓았다. v4 는 배속이 하나(2.0 또는 2.5)이므로 셀 상한이 필요
없고, 그러면 서브태스크 라벨을 쓸 곳이 없어진다 -- 프롬프트에는 원래 안 들어갔다.
**다른 데이터셋에는 서브태스크 라벨이 없으므로 이쪽이 일반적이다.**

**지시문을 넣는다.** v3 는 뺐는데, 그때 지시문이 서브태스크 이름이라 칸마다 다른
상수였고 모델이 화면 대신 그것을 보고 답했다(Rotate Box 77청크가 전부 한 값).
여기서는 지시문이 데이터셋 전체에 하나여서 구분 정보가 0 이다 -- 상수화를 일으킬
수 없고, 앞으로 어떤 동작이 오는지의 맥락은 된다. robocasa v7 · libero v3c 와도
이쪽이 통일된다.

산출물은 청크마다 한 줄이고 `gp`(등급 분포)를 싣는다. conf 는 기댓값 등급으로
계산된다 -- 정수 등급으로 계산하면 계단이 되어 역치가 안 먹는다(하네스 R4).

    ALLEX_CHECKS_MODULE=allex_v4_checks \
    ALLEX_DS=/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4 \
      python allex_v4_label.py <port> [에피수]
"""
import hashlib
import importlib
import json
import os
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_v2_common import descriptors          # noqa: E402
from vlm_gate import VLMGate                     # noqa: E402

CK = importlib.import_module(os.environ.get("ALLEX_CHECKS_MODULE", "allex_v4_checks"))
Q = tuple(sorted(CK.SIGN))
DS = os.environ.get("ALLEX_DS",
                    "/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
OUT = os.environ.get("ALLEX_OUT", f"{os.getcwd()}/output/allex_v4/records.jsonl")
CHUNK = int(os.environ.get("ALLEX_CHUNK", "16"))
STRIDE = int(os.environ.get("ALLEX_STRIDE", "16"))
BATCH = int(os.environ.get("ALLEX_BATCH", "8"))
PORT = sys.argv[1]
NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 0

os.makedirs(os.path.dirname(OUT), exist_ok=True)


def cdir(ep):
    return f"chunk-{ep // 1000:03d}"


# --- 지시문. 데이터셋 전체에 하나일 수도, task_index 별로 여럿일 수도 있다. ---
TASKS = {}
with open(f"{DS}/meta/tasks.jsonl") as f:
    for line in f:
        r = json.loads(line)
        TASKS[r["task_index"]] = r["task"]
print(f"[i] 지시문 {len(TASKS)}개: " +
      " / ".join(list(TASKS.values())[:2])[:150], flush=True)

EPS = sorted(int(p.split("episode_")[1][:6])
             for p in __import__("glob").glob(f"{DS}/data/*/*.parquet"))
# 에피를 골라 돌릴 수 있다. 앞에서부터 N 개만 보면 표본이 한쪽에 몰린다 --
# ep0~14 는 잡은구간 손바닥 간격이 전부 0.352~0.404 로 넓은 쪽이었고, 그래서
# "늘어진 짐" 문항이 한 번도 안 떴다. 물체가 갈리는 축을 일부러 덮어야 한다.
# 청크를 프레임 단위로 지정할 수 있다: {"52": [0, 144, ...], ...}
PICK = None
_pf = os.environ.get("ALLEX_FRAMES", "")
if _pf:
    PICK = {int(k): set(v) for k, v in json.load(open(_pf)).items()}
    print(f"[i] 프레임 목록 지정: {sum(len(v) for v in PICK.values())}청크 / {len(PICK)}에피", flush=True)

_pick = os.environ.get("ALLEX_EPS", "")
if PICK is not None:
    EPS = [e for e in EPS if e in PICK]
elif _pick:
    want = {int(x) for x in _pick.replace(",", " ").split()}
    EPS = [e for e in EPS if e in want]
elif NEP:
    EPS = EPS[:NEP]
print(f"[i] 에피 {len(EPS)} · 청크 {CHUNK} · stride {STRIDE} · 문항 {''.join(Q)}",
      flush=True)

json.dump({"checks": os.environ.get("ALLEX_CHECKS_MODULE", "allex_v4_checks"),
           "dataset": DS, "chunk": CHUNK, "stride": STRIDE,
           "ngrade": CK.NGRADE,
           "sign": {q: CK.SIGN[q] for q in Q},
           "weight": {q: CK.WEIGHT[q] for q in Q},
           "name": getattr(CK, "NAME", {}),
           "guidance_sha1": hashlib.sha1(CK.GUIDANCE.encode()).hexdigest()[:12],
           "ask_sha1": hashlib.sha1(CK.ASK.encode()).hexdigest()[:12]},
          open(OUT.replace(".jsonl", "_meta.json"), "w"),
          ensure_ascii=False, indent=1)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            done.add((r["ep"], r["f"]))
        except Exception:
            pass
    print(f"[i] 이어서 {len(done)}", flush=True)


def grab(ep, frames, side):
    want, got = set(frames), {}
    path = (f"{DS}/videos/{cdir(ep)}/observation.images.camera_ego_{side}/"
            f"episode_{ep:06d}.mp4")
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want):
                    break
    return got


gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)
out = open(OUT, "a")
ntot = nbad = 0
for ep in EPS:
    d = pd.read_parquet(f"{DS}/data/{cdir(ep)}/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ti = d["task_index"].values
    if PICK is not None:                      # 프레임 목록을 직접 받는 경로.
        # 판정기끼리 비교하려면 **똑같은 청크**를 봐야 한다. stride 로 고르면
        # 판정기마다 다른 청크를 보게 되어 비교가 성립하지 않는다.
        starts = [f for f in sorted(PICK.get(ep, ())) if (ep, f) not in done]
    else:
        starts = [f for f in range(0, len(d) - CHUNK, STRIDE) if (ep, f) not in done]
    if not starts:
        continue
    L = grab(ep, starts, "left")
    R = grab(ep, starts, "right")
    for b0 in range(0, len(starts), BATCH):
        grp = [f for f in starts[b0:b0 + BATCH] if f in L and f in R]
        if not grp:
            continue
        payload, meta = [], []
        for f in grp:
            x = descriptors(A, WR, WL, f, CHUNK)
            seg = ti[f:f + CHUNK]
            instr = TASKS.get(int(np.bincount(seg).argmax()), "")
            payload.append(([L[f], R[f]], CK.facts_v3(x, instr)))
            meta.append((f, instr, x))
        try:
            rs = gate.judge_batch(payload, CK.GUIDANCE, question=CK.ASK,
                                  n_ask=len(Q), n_grade=CK.NGRADE)
        except Exception as e:
            print(f"  ep{ep} f{grp[0]}+: {type(e).__name__}: {e}", flush=True)
            continue
        for (f, instr, x), r in zip(meta, rs):
            picks = r.get("picks")
            if r.get("error") or not picks or len(picks) < len(Q) \
                    or any(p is None for p in picks):
                nbad += 1
                continue
            g = {q: int(picks[i]) for i, q in enumerate(Q)}
            gp = r.get("grade_probs")
            eg = CK.expected_grades(gp) if gp else None
            rec = {"ep": ep, "f": int(f), "instr": instr,
                   **g,
                   "conf": round(CK.confidence(picks, gp), 4),
                   **({"eg": [round(float(v), 3) for v in eg]} if eg else {}),
                   **({"gp": [[round(float(v), 4) for v in row] for row in gp]}
                      if gp and len(gp) == len(Q) else {}),
                   "text": r.get("text", "").replace("\n", " | "),
                   **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in x.items()}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ntot += 1
        out.flush()
    print(f"  ep{ep} 누적 {ntot} (형식실패 {nbad})", flush=True)
out.close()
print(f"완료 {ntot} · 형식실패 {nbad} -> {OUT}")
