"""phase9's five checks over EVERY frame, decoded from the videos.

The tile directory holds one PNG per 8th frame -- 260,031 of them -- and the
pilot labelled a shuffled 1,898 of those. Labelling every frame through that
directory would mean writing about two million more files into a tree the
infrastructure team already watches, so nothing is written: the three camera
views are decoded straight out of the episode's videos and tiled in memory, the
same way the allex labeller works.

Sharding is by episode so each worker opens each video once and walks it in
order; random access across a whole shard would decode far more than it reads.
Resume is per shard file, so a preemption on the background partition costs the
frames of one episode at most.

    python phase9_label_full.py <port> <shard> <nshard>
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 문항 판을 CHECKS 로 고른다. 기본은 v1 이라 기존 실행이 그대로 재현된다.
import importlib  # noqa: E402
_CK = importlib.import_module(os.environ.get("CHECKS", "phase9_checks"))
ASK, GUIDANCE, NGRADE = _CK.ASK, _CK.GUIDANCE, _CK.NGRADE

# **집계를 모델에 맡기는 판.** SELFAGG=1 이면 문항 다섯을 따로 묻지 않고, 어느
# 문항이 감점/가산점인지 알려 주고 최종 판단 하나를 받는다. 역치가 없어지는 대신
# 해상도가 다섯 칸이고 운용점을 못 고른다 -- 어느 쪽이 나은지 재려고 두 판을 낸다.
SELFAGG = os.environ.get("SELFAGG", "") not in ("", "0")
if SELFAGG:
    ASK = open(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
               f"/prompts/robocasa_phase9_selfagg.txt").read().strip()
    N_ASK = 1
else:
    N_ASK = 5
from robocasa_descriptors import descriptors, facts_text  # noqa: E402
from vlm_gate import VLMGate  # noqa: E402

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
OUT = os.environ.get("PHASE9_OUT", f"{BASE}/output/_gate_distill/phase9_full")
PORT = sys.argv[1]
# Two ways to be told what to do. A worklist file (phase9_worklist.py) names the
# episodes and the frames already labelled, so the shard count is no longer tied
# to the file names on disk and a half-finished run survives changing it.
WORKLIST = os.environ.get("PHASE9_WORKLIST", "")
REVERSE = os.environ.get("PHASE9_REVERSE", "") == "1"
# The forward worker's log, so a reverse pass can see where it has got to and
# stop when the two meet instead of redoing the whole front half.
FRONT_LOG = os.environ.get("PHASE9_FRONT_LOG", "")
SHARD = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NSHARD = int(sys.argv[3]) if len(sys.argv) > 3 else 1
BLOCK = 64                      # frames decoded per read
BATCH = int(os.environ.get('PHASE9_BATCH', 8))   # frames per forward
# 라벨을 전 프레임 대신 stride 로 뽑는다. 창이 16프레임이라 인접 라벨이 15/16
# 겹치므로, stride 4 로 뽑아 학습 때 펼쳐도 이진 라벨 재현율 95.1% (반반 균형,
# 에피 800개 22.7만 프레임 실측). 1 이 15% 아래로 희귀해지면 91% 밑으로 떨어진다.
STRIDE = int(os.environ.get('PHASE9_STRIDE', 1))

info = json.load(open(f"{DS}/meta/info.json"))
vks = [k for k in info["features"] if info["features"][k].get("dtype") == "video"]
VK = ([k for k in vks if "left" in k]
      + [k for k in vks if "right" in k and "wrist" not in k]
      + [k for k in vks if "wrist" in k])

instr = {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(line)
    c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""

os.makedirs(OUT, exist_ok=True)
if WORKLIST:
    wl = json.load(open(WORKLIST))
    eps = list(wl["episodes"])
    done = {(int(ep), f) for ep, fs in wl["done"].items() for f in fs}
    # A second pair of workers can walk the SAME worklist backwards while the
    # first pair is still going forwards, without re-dealing the worklist --
    # re-dealing would hand out episodes the running job is mid-way through.
    # They meet in the middle; only the meeting episode is done twice, and
    # rows are keyed (ep, f) so the merge drops it. What the forward worker
    # has already written is read in below, so the reverse pass starts past it.
    if REVERSE:
        eps = eps[::-1]
        FP = f"{OUT}/labels_r{SHARD}.jsonl"
        fw = f"{OUT}/labels_w{SHARD}.jsonl"
        if os.path.exists(fw):
            for line in open(fw):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done.add((r["ep"], r["f"]))
    else:
        FP = f"{OUT}/labels_w{SHARD}.jsonl"
    print(f"worker {SHARD}{' (역순)' if REVERSE else ''}: {len(eps)} episodes, "
          f"{wl['n_left']:,} frames left, {len(done):,} already done", flush=True)
else:
    FP = f"{OUT}/labels_s{NSHARD}_{SHARD}.jsonl"
    done = set()
    if os.path.exists(FP):
        for line in open(FP):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["ep"], r["f"]))
    eps = [e for e in range(7200) if e % NSHARD == SHARD]
    print(f"shard {SHARD}/{NSHARD}: {len(eps)} episodes, resuming past {len(done)} frames",
          flush=True)
# A restart of THIS worker still needs to skip what it wrote itself.
if os.path.exists(FP):
    for line in open(FP):
        try:
            r = json.loads(line)
        except Exception:
            continue
        done.add((r["ep"], r["f"]))

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)

json.dump({"batch": BATCH, "shard": SHARD, "nshard": NSHARD, "decode_block": BLOCK,
           "worklist": WORKLIST or None, "reverse": REVERSE,
           "note": "batch width changes borderline answers; rerun with the same "
                   "batch to reproduce this file"},
          open(f"{OUT}/meta_{'r' if REVERSE else 'w'}{SHARD}.json" if WORKLIST else
               f"{OUT}/meta_s{NSHARD}_{SHARD}.json", "w"))
fh = open(FP, "a")
nlab = nfull = nempty = 0
for ei, ep in enumerate(eps):
    ch = ep // info["chunks_size"]
    try:
        a = np.stack(pd.read_parquet(
            f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        vrs = [VideoReader(f"{DS}/" + info["video_path"].format(
            episode_chunk=ch, episode_index=ep, video_key=k)) for k in VK]
    except Exception:
        continue
    n = min(min(len(v) for v in vrs), len(a) - 4)
    todo = [f for f in range(0, max(n, 0), STRIDE) if (ep, f) not in done]
    if not todo:
        continue
    ins_ep = instr.get(ep, "")
    for b0 in range(0, len(todo), BLOCK):
        idx = todo[b0:b0 + BLOCK]
        try:
            blocks = [v.get_batch(idx).asnumpy() for v in vrs]
        except Exception:
            break
        # BATCH is part of the label. Each way of running is reproducible on its
        # own -- three single passes agreed 32/32, two batch-8 passes agreed
        # 32/32 -- but single and batch-8 disagree on 2 of 32, because a
        # different batch width picks different kernels and the last bits of
        # bfloat16 move. Those are chunks sitting between two grades either way.
        # The size is recorded in meta.json so a rerun can reproduce this file.
        for b1 in range(0, len(idx), BATCH):
            grp = list(enumerate(idx))[b1:b1 + BATCH]      # (row in blocks, frame)
            payload = [([Image.fromarray(bl[row]) for bl in blocks],
                        f"{ins_ep}\n{facts_text(descriptors(a, f))}") for row, f in grp]
            try:
                rs = gate.judge_batch(payload, GUIDANCE, question=ASK,
                                      n_ask=N_ASK, n_grade=NGRADE)
            except Exception as e:
                print(f"  ep{ep} f{grp[0]}+: {type(e).__name__}", flush=True)
                continue
            for (row, f), r in zip(grp, rs):
                # A dead judge answers every call the same; writing rows through
                # it yields a file of the right length and no content, which
                # counting outputs cannot tell from success.
                if r.get("error") or not r.get("text", "").strip():
                    nempty += 1
                    if nempty >= 5 and nlab < 5:
                        raise SystemExit(f"judge not answering: {r.get('error', 'empty')}")
                    continue
                nempty = 0
                picks = r.get("picks") or [None] * N_ASK
                nfull += int(all(p is not None for p in picks))
                rec = ({"ep": ep, "f": f, "Z": picks[0]} if SELFAGG else
                       {"ep": ep, "f": f,
                        **{q: picks[i] for i, q in enumerate("ABCDE")}})
                # **등급 분포를 버리지 않는다.** ratio_label.confidence 는 gp 가
                # 있으면 정수 등급 대신 기댓값 Σ(i+1)·p(i) 를 쓰는데, 라벨러가
                # 이것을 적지 않아서 2.04M 행 전부가 정수 등급으로 만들어졌다.
                # P(3)=0.9 와 P(2)=.3/P(3)=.35/P(4)=.3 은 전혀 다른 상태다.
                _gp = r.get("grade_probs")
                if _gp:
                    rec["gp"] = _gp
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                nlab += 1
        fh.flush()
    if ei % 20 == 0:
        tag = "역순 " if REVERSE else ""
        print(f"  shard{SHARD}: {tag}ep {ei}/{len(eps)}  {nlab} frames  full {nfull}",
              flush=True)
        # Stop where the forward worker has already been. Its log's last
        # progress line names the index it is on, in the un-reversed list; this
        # pass is at len(eps)-1-ei of that same list. Without this the reverse
        # pass would carry on into ground the forward pass has covered and
        # relabel all of it.
        if REVERSE and FRONT_LOG and os.path.exists(FRONT_LOG):
            front = -1
            try:
                for line in open(FRONT_LOG):
                    m = re.search(r"ep (\d+)/", line)
                    if m:
                        front = int(m.group(1))
            except Exception:
                pass
            if front >= 0 and (len(eps) - 1 - ei) <= front:
                print(f"  shard{SHARD}: 앞선 워커와 만남 (앞 {front}, 뒤 "
                      f"{len(eps) - 1 - ei}) -- 중단", flush=True)
                break

fh.close()
print(f"shard{SHARD} done: {nlab} frames, {nfull} complete -> {FP}")
