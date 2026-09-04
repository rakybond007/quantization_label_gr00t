"""Where phase9 labelling time actually goes, and whether decoding ahead wins it back.

The labeller runs decode -> judge -> decode on one thread, so the GPU idles
through every read and the CPU idles through every forward. The training
loader does not do this: gr00t_finetune_moh.py sets dataloader_num_workers=8
with persistent_workers=True, and CachedLeRobotSingleDataset goes further and
decodes whole trajectories up front. This mirrors the first of those.

decord readers are not safe to share, so the split is by episode -- each
decoder thread opens its own readers and walks its own episodes, exactly how a
DataLoader worker owns its samples. Blocks land in a bounded queue; the judge
loop drains it. Row order stops being episode order, which costs nothing
because every row carries its own (ep, f).

    python phase9_prefetch_test.py <port> seq|pf [n_ep] [n_thread]
"""
import json
import os
import queue
import sys
import threading
import time

import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase9_checks import ASK, GUIDANCE, NGRADE  # noqa: E402
from robocasa_descriptors import descriptors, facts_text  # noqa: E402
from vlm_gate import VLMGate  # noqa: E402

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
PORT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "seq"
NEP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
NTHREAD = int(sys.argv[4]) if len(sys.argv) > 4 else 4
BLOCK = 64
BATCH = int(os.environ.get("PHASE9_BATCH", 8))
DEPTH = int(os.environ.get("PHASE9_DEPTH", 4))   # blocks held ready

info = json.load(open(f"{DS}/meta/info.json"))
vks = [k for k in info["features"] if info["features"][k].get("dtype") == "video"]
VK = ([k for k in vks if "left" in k]
      + [k for k in vks if "right" in k and "wrist" not in k]
      + [k for k in vks if "wrist" in k])

instr = {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(line)
    c = [t for t in d.get("tasks", []) if isinstance(t, str)
         and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)
EPS = sorted(instr)[:NEP]
t_decode = 0.0      # seconds spent inside get_batch, summed over threads
t_judge = 0.0
N_NEW, TXT = [], []


def open_ep(ep):
    """Readers plus the action array for one episode, or None if either is missing."""
    ch = ep // info["chunks_size"]
    try:
        a = np.stack(pd.read_parquet(
            f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        vrs = [VideoReader(f"{DS}/" + info["video_path"].format(
            episode_chunk=ch, episode_index=ep, video_key=k)) for k in VK]
    except Exception:
        return None
    n = min(min(len(v) for v in vrs), len(a) - 4)
    return a, vrs, list(range(max(n, 0)))


def judge(ep, a, idx, blocks):
    """One block: BATCH frames per forward, same call the labeller makes."""
    global t_judge
    ins = instr.get(ep, "")
    out = 0
    for b1 in range(0, len(idx), BATCH):
        grp = list(enumerate(idx))[b1:b1 + BATCH]
        payload = [([Image.fromarray(bl[row]) for bl in blocks],
                    f"{ins}\n{facts_text(descriptors(a, f))}") for row, f in grp]
        t0 = time.time()
        rs = gate.judge_batch(payload, GUIDANCE, question=ASK, n_ask=5, n_grade=NGRADE)
        t_judge += time.time() - t0
        for r in rs:
            N_NEW.append(r.get("n_new", 0))
            TXT.append(len(r.get("text", "")))
        out += sum(1 for r in rs if r.get("text", "").strip())
    return out


def run_seq():
    """What the running job does: one thread, decode and judge in turn."""
    global t_decode
    n = 0
    for ep in EPS:
        o = open_ep(ep)
        if not o:
            continue
        a, vrs, frames = o
        for b0 in range(0, len(frames), BLOCK):
            idx = frames[b0:b0 + BLOCK]
            t0 = time.time()
            blocks = [v.get_batch(idx).asnumpy() for v in vrs]
            t_decode += time.time() - t0
            n += judge(ep, a, idx, blocks)
    return n


def run_pf():
    """Decoder threads run ahead; the judge loop drains a bounded queue."""
    global t_decode
    q = queue.Queue(maxsize=DEPTH)
    lock = threading.Lock()

    def worker(my_eps):
        global t_decode
        for ep in my_eps:
            o = open_ep(ep)
            if not o:
                continue
            a, vrs, frames = o
            for b0 in range(0, len(frames), BLOCK):
                idx = frames[b0:b0 + BLOCK]
                t0 = time.time()
                try:
                    blocks = [v.get_batch(idx).asnumpy() for v in vrs]
                except Exception:
                    break
                with lock:
                    t_decode += time.time() - t0
                q.put((ep, a, idx, blocks))     # blocks when DEPTH ahead

    ts = [threading.Thread(target=worker, args=(EPS[i::NTHREAD],), daemon=True)
          for i in range(NTHREAD)]
    for t in ts:
        t.start()
    n = 0
    alive = True
    while alive or not q.empty():
        try:
            ep, a, idx, blocks = q.get(timeout=1.0)
        except queue.Empty:
            alive = any(t.is_alive() for t in ts)
            continue
        n += judge(ep, a, idx, blocks)
        alive = any(t.is_alive() for t in ts)
    return n


def run_pp():
    """Keep a request always in flight.

    The server-side profile says preprocess+generate is 1.63s of a 2.06s round
    trip; the missing 0.43s is this side -- building PIL images, base64, the
    socket -- and the GPU is idle for all of it. Threads here each own their
    episodes end to end, so while one waits on the judge the next has its
    payload already built. The judge serialises the forwards itself, which is
    what we want: it should never be the one waiting.
    """
    lock = threading.Lock()
    tot = [0]

    def worker(my_eps):
        n = 0
        for ep in my_eps:
            o = open_ep(ep)
            if not o:
                continue
            a, vrs, frames = o
            for b0 in range(0, len(frames), BLOCK):
                idx = frames[b0:b0 + BLOCK]
                try:
                    blocks = [v.get_batch(idx).asnumpy() for v in vrs]
                except Exception:
                    break
                n += judge(ep, a, idx, blocks)
        with lock:
            tot[0] += n

    ts = [threading.Thread(target=worker, args=(EPS[i::NTHREAD],))
          for i in range(NTHREAD)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return tot[0]


t_all = time.time()
nlab = {"seq": run_seq, "pf": run_pf, "pp": run_pp}[MODE]()
t_all = time.time() - t_all

print(f"\n  방식      {MODE}" + (f" (스레드 {NTHREAD}, 큐 {DEPTH})" if MODE == "pf" else ""))
print(f"  라벨      {nlab} 프레임")
print(f"  전체      {t_all:.1f}초   프레임당 {t_all / max(nlab, 1):.3f}초")
print(f"  디코딩    {t_decode:.1f}초 (스레드 합)   판정 {t_judge:.1f}초")
if N_NEW:
    print(f"  생성토큰  블록최대 평균 {np.mean(N_NEW):.0f}  최대 {max(N_NEW)}"
          f"   답변 길이 평균 {np.mean(TXT):.0f}자")
print(f"  겹친양    {t_decode + t_judge - t_all:+.1f}초"
      f"   GPU 놀린 비율 {max(t_all - t_judge, 0) / max(t_all, 1e-9):.1%}")
