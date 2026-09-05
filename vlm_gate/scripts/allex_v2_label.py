"""allex v2 - two-stage variable-ratio labelling of the subtask-labelled dataset.

Per 16-step chunk:
  stage 1  general prompt (allex_common_v5 GUIDANCE/ASK) -> base confidence p
  stage 2  task-specific prompt (allex_v2_common)        -> ceiling K_max
  K = snap(1 + p*(K_max-1)) in {1, 2, 2.5, 3}

Both stages are one Cosmos call each: a single image prefill with four
teacher-forced "X) YES/NO" slots, P(YES) read off the {YES,NO} tokens.

  python allex_v2_label.py <PORT> <SHARD> <NSHARDS> [EP_LIST]
"""
import json, os, sys
import numpy as np, pandas as pd, av
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
from allex_common_v5 import GUIDANCE, ASK
from allex_v2_common import (descriptors, facts_text, stage1_confidence,
                             STAGE2_GUIDANCE, STAGE2_ASK, stage2_facts,
                             ceiling_from_stage2, final_ratio, TASKS)
# STAGE2_MODE=graded replaces the four yes/no checks with one graded question the
# model answers from the scene. Needed where there are no subtask labels to look
# a prior ceiling up from; the shipped path is untouched and stays the default.
STAGE2_MODE = os.environ.get("STAGE2_MODE", "checks")
if STAGE2_MODE == "twosided":
    # five one-thing checks weighted by the hand-set subtask ceilings
    from allex_twosided_ceiling import (ASK as TS_ASK, GUIDANCE as TS_GUIDANCE,
                                        ceiling_from_twosided)
elif STAGE2_MODE == "graded":
    from allex_graded_ceiling import (LEVELS, STAGE2_GRADED_ASK, STAGE2_GRADED_GUIDANCE,
                                      argmax_ceiling, ceiling_from_graded)

# The recording is an argument, not a constant: a second allex capture (v3) is
# labelled with this same prompt, and its output must not land on v1's.
DS = os.environ.get(
    "ALLEX_DS",
    "/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin")
OUTDIR = os.path.expanduser(os.environ.get(
    "ALLEX_OUT", "~/quantization_agent_workspace/vlm_gate/output/allex_v2"))
os.makedirs(OUTDIR, exist_ok=True)
CHUNK = 16

PORT = sys.argv[1]
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSH = int(sys.argv[3]) if len(sys.argv) > 3 else 1
EPS = [int(e) for e in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
TAG = os.environ.get("TAG", f"s{NSH}_{SHARD}")
OUT = f"{OUTDIR}/labels_{TAG}.jsonl"

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)

# 데이터셋이 1000 에피소드마다 chunk-000, chunk-001 로 나뉜다. 앞 판은 하나뿐이라
# chunk-000 을 박아 뒀는데, merged_v5tempo 는 1280 개라 뒤 280 개를 못 찾는다.
try:
    CHUNKS_SIZE = int(json.load(open(f"{DS}/meta/info.json"))["chunks_size"])
except Exception:
    CHUNKS_SIZE = 1000


def cdir(ep):
    return f"chunk-{ep // CHUNKS_SIZE:03d}"


# 서브태스크 이름을 어디서 읽나. 앞 판은 parquet 의 task_index 가 서브태스크를
# 가리켰는데, merged_v5tempo 는 tasks.jsonl 에 항목이 하나뿐이고 task_index 가
# 전부 0 이다. 서브태스크는 meta/subtasks.jsonl 의 구간에만 있다. 상한이 이
# 이름에서 나오므로 구간에서 읽는다 -- 있으면 그쪽이 우선이다.
SEGS = {}
_sp = f"{DS}/meta/subtasks.jsonl"
if os.path.exists(_sp) and len(TASKS) < 4:
    for _l in open(_sp):
        _r = json.loads(_l)
        SEGS.setdefault(_r["episode_index"], []).append(
            (_r["start_frame"], _r["end_frame"], _r["label"]))
    for _v in SEGS.values():
        _v.sort()
    print(f"  서브태스크 구간 {sum(len(v) for v in SEGS.values())}개를 "
          f"{len(SEGS)} 에피소드에서 읽음", flush=True)


def seg_label(ep, f, n=CHUNK):
    """청크 한가운데 프레임이 든 구간의 이름."""
    mid = f + n // 2
    for a, b, lab in SEGS.get(ep, ()):
        if a <= mid < b:
            return lab
    return None

episodes = []
for l in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(l); episodes.append((d["episode_index"], d["length"]))
# 청크 표본만 라벨링하는 길. v3 루프의 참고 지표([7])는 v2 의 stage-1 확신과
# 견주는 것인데, 층화 표본이 에피소드 70여 개에 흩어져 있어 에피소드 단위로
# 돌리면 표본의 수십 배를 라벨링하게 된다. 표본 파일이 주어지면 그 청크만 한다.
WANT = None
_sf = os.environ.get("ALLEX_SAMPLE_FILE", "")
if _sf:
    _d = json.load(open(os.path.expanduser(_sf)))
    WANT = {}
    for _v in _d["strata"].values():
        for _ep, _f in _v:
            WANT.setdefault(int(_ep), set()).add(int(_f))
    episodes = [(e, n) for e, n in episodes if e in WANT and e % NSH == SHARD]
    print(f"  표본 {sum(len(v) for v in WANT.values())} 청크, "
          f"에피소드 {len(WANT)}개 중 이 샤드 {len(episodes)}개", flush=True)
elif EPS is not None:
    episodes = [(e, n) for e, n in episodes if e in EPS]
else:
    episodes = [(e, n) for e, n in episodes if e % NSH == SHARD]

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l); done.add((r["ep"], r["f"]))
        except Exception:
            pass


def grab(ep, frames, side):
    """Decode the requested frame indices from one ego camera."""
    want = set(frames); got = {}
    path = (f"{DS}/videos/{cdir(ep)}/observation.images.camera_ego_{side}"
            f"/episode_{ep:06d}.mp4")
    with av.open(path) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = Image.fromarray(fr.to_ndarray(format="rgb24"))
                if len(got) == len(want):
                    break
    return got


out = open(OUT, "a")
ntot = nerr = 0
for ep, N in episodes:
    d = pd.read_parquet(f"{DS}/data/{cdir(ep)}/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ti = d["task_index"].values
    starts = [f for f in range(0, len(A) - CHUNK, CHUNK) if (ep, f) not in done
              and (WANT is None or f in WANT[ep])]
    if not starts:
        print(f"ep{ep}: already done", flush=True); continue
    L = grab(ep, starts, "left"); R = grab(ep, starts, "right")
    n0 = ntot
    for f in starts:
        try:
            x = descriptors(A, WR, WL, f, CHUNK)
            seg = ti[f:f + CHUNK]
            task = (seg_label(ep, f) if SEGS else None) or \
                TASKS[int(np.bincount(seg, minlength=len(TASKS)).argmax())]
            views = [L[f], R[f]]
            # ---- stage 1: general "is this moment safe to compress at all"
            r1 = gate.judge(views, f"{task}\n{facts_text(x)}", GUIDANCE,
                            question=ASK, n_ask=4)
            c1 = r1.get("confidences")
            if not c1 or len(c1) != 4:
                raise ValueError(f"stage1 parse: {r1.get('error','')}")
            p = stage1_confidence(c1, x, task)
            # ---- stage 2: the ceiling
            if STAGE2_MODE == "twosided":
                from allex_twosided_ceiling import CEILING as TS_CEIL
                n_ts = len(TS_CEIL)
                r2 = gate.judge(views, f"{task}\n{stage2_facts(task, x)}",
                                TS_GUIDANCE, question=TS_ASK, n_ask=n_ts,
                                n_grade=5, mode="text")
                picks = r2.get("picks") or [None] * n_ts
                if all(p is None for p in picks):
                    raise ValueError(f"stage2 twosided parse: {r2.get('error', r2.get('text',''))[:80]}")
                K_max = ceiling_from_twosided(picks)
                c2 = [float(p) if p is not None else 0.0 for p in picks]
                r2["_pick"] = K_max
            elif STAGE2_MODE == "graded":
                r2 = gate.judge(views, f"{task}\n{stage2_facts(task, x)}",
                                STAGE2_GRADED_GUIDANCE, question=STAGE2_GRADED_ASK,
                                n_ask=1, n_grade=len(LEVELS))
                d2 = (r2.get("dist") or [[]])[0]
                if len(d2) != len(LEVELS):
                    raise ValueError(f"stage2 graded parse: {r2.get('error','')}")
                K_max = ceiling_from_graded(d2)
                c2 = d2                       # stored per level, not per check
                r2["_pick"] = argmax_ceiling(d2)
            else:
                r2 = gate.judge(views, f"{task}\n{stage2_facts(task, x)}", STAGE2_GUIDANCE,
                                question=STAGE2_ASK, n_ask=4)
                c2 = r2.get("confidences")
                if not c2 or len(c2) != 4:
                    raise ValueError(f"stage2 parse: {r2.get('error','')}")
                K_max = ceiling_from_stage2(task, *c2)
            K_pre = final_ratio(p, K_max, task)   # 하한이 태스크마다 다르다
            rec = {"ep": ep, "f": f, "task": task, "p": float(p), "K_max": float(K_max),
                   "K_pre": float(K_pre),
                   **{f"s1_{k}": float(v) for k, v in zip("ABCD", c1)},
                   # slot count follows the mode: checks/graded write four, the
                   # two-sided set writes five. Hardcoding "ABCD" silently dropped
                   # the fifth answer while K_max was still computed from all five,
                   # so the stored record could not reproduce its own ceiling.
                   **{f"s2_{k}": float(v) for k, v in zip("ABCDE", c2)},
                   **({"K_max_pick": r2.get("_pick")}
                      if STAGE2_MODE in ("graded", "twosided") else {}),
                   "ans1": r1.get("answer", ""),
                   # text mode returns "text"; "answer" is the logit path's key,
                   # so reading only that stored 115 empty strings and hid what
                   # the model actually wrote.
                   "ans2": r2.get("text") or r2.get("answer", ""),
                   **{k: (int(v) if isinstance(v, bool) else float(v)) for k, v in x.items()}}
            out.write(json.dumps(rec) + "\n"); ntot += 1
            if ntot % 100 == 0:
                out.flush(); print(f"  {TAG}: {ntot} chunks", flush=True)
        except Exception as e:
            nerr += 1
            if nerr <= 5:
                print("ERR", type(e).__name__, str(e)[:160], flush=True)
    out.flush()
    print(f"ep{ep}: {ntot-n0} chunks (total {ntot}, err {nerr})", flush=True)
out.close()
print(f"{TAG} done: {ntot} chunks, {nerr} errors -> {OUT}")
