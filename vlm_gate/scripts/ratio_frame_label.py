"""배속 문항 라벨러. 평가 롤아웃 초기 장면 -> 문항 등급 + 등급 분포.

    python ratio_label.py <port> [dev|hold|all]

CHECKS 로 문항 판을 고른다. 등급 분포를 반드시 저장한다 -- 정수 등급만 남기면
배율 사상이 계단이 되고, 문항 판을 견줄 때 표본 잡음과 구분되지 않는다.
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
from vlm_gate import VLMGate                                        # noqa: E402

CK = importlib.import_module(os.environ.get("CHECKS", "ratio_checks_v1"))
TAG = os.environ.get("TAG", os.environ.get("CHECKS", "ratio_checks_v1"))
# **관측을 바꾸는 손잡이 둘.** 지시문은 태스크마다 상수라 모델이 그것만 읽고
# 그림을 안 볼 수 있다(1라운드에서 태스크 안 점수 표준편차가 태스크 간의 0.08
# 배였다). 확대판은 원본과 작업면 확대를 나란히 붙인 것이다.
FR = os.environ.get("RATIO_FRAMES", f"{BASE}/output/_gate_distill/ratio_frames")
NO_INSTR = os.environ.get("NO_INSTR", "0") == "1"
OUT = f"{BASE}/analysis/ratio_prompt/{TAG}.jsonl"
BATCH = int(os.environ.get("RATIO_BATCH", "8"))
PORT = sys.argv[1]
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "dev"

T = pd.read_parquet(f"{BASE}/analysis/ratio_prompt/episode_split.parquet")
T.columns = [c if c != 0 else "max_ratio" for c in T.columns]
if SPLIT != "all":
    T = T[T.split == SPLIT]
INSTR = {r["task"]: r["instruction"] for r in
         json.load(open(f"{BASE}/analysis/ratio_prompt/task_instruction.json"))}
NQ = len(CK.SIGN)

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l); done.add((r["task"], r["ep"]))
        except Exception:
            pass
    print(f"이어서: {len(done)} 완료", flush=True)

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=300)
out = open(OUT, "a")
json.dump({"checks": os.environ.get("CHECKS"), "ngrade": CK.NGRADE,
           "sign": CK.SIGN, "weight": CK.WEIGHT, "split": SPLIT, "batch": BATCH, "frames": FR, "no_instr": NO_INSTR},
          open(OUT.replace(".jsonl", "_meta.json"), "w"), ensure_ascii=False, indent=1)

buf, n, miss = [], 0, 0
rows = [r for r in T.itertuples() if (r.task, int(r.ep)) not in done]
print(f"{SPLIT}: {len(rows)}에피 · 문항 {NQ} · {TAG}", flush=True)
for i, r in enumerate(rows + [None]):
    if r is not None:
        p = f"{FR}/{r.task}__ep{int(r.ep):03d}.png"
        if not os.path.exists(p):
            miss += 1; continue
        im = Image.open(p).convert("RGB")
        buf.append((r.task, int(r.ep), float(r.max_ratio),
                    ([im], "" if NO_INSTR else f"The robot was told: {INSTR.get(r.task,'')}")))
        if len(buf) < BATCH:
            continue
    if not buf:
        continue
    try:
        rs = gate.judge_batch([b[3] for b in buf], CK.GUIDANCE, question=CK.ASK,
                              n_ask=NQ, n_grade=CK.NGRADE, mode="text")
    except Exception as e:
        print(f"batch {type(e).__name__}: {e}", flush=True); buf = []; continue
    for (tk, ep, y, _), res in zip(buf, rs):
        c = res.get("picks")
        if res.get("error") or not c or len(c) != NQ or any(v is None for v in c):
            miss += 1; continue
        rec = {"task": tk, "ep": ep, "max_ratio": y,
               **{q: int(v) for q, v in zip(sorted(CK.SIGN), c)},
               "text": res.get("text", "")}
        gp = res.get("grade_probs")
        if gp and len(gp) == NQ:
            rec["gp"] = [[round(float(v), 4) for v in row] for row in gp]
        out.write(json.dumps(rec) + "\n"); n += 1
    out.flush(); buf = []
    if n % 80 < BATCH:
        print(f"  {n}/{len(rows)} (miss {miss})", flush=True)
out.close()
print(f"완료 {n} · 실패 {miss} -> {OUT}")
