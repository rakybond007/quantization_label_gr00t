"""판정기 둘을 **같은 장면·같은 프롬프트**로 돌려 비교한다.

    # 1) 장면을 고른다 (모델 안 씀, CPU)
    python vlm_gate/scripts/judge_ab.py pick --out /tmp/ab_scenes.json

    # 2) 한 모델로 그 장면들을 답하게 한다 (판정 서버가 떠 있어야 한다)
    python vlm_gate/scripts/judge_ab.py run --port 8120 --tag cosmos \\
        --scenes /tmp/ab_scenes.json --out /tmp/ab_cosmos.jsonl

    # 3) 둘을 나란히 놓는다 (CPU)
    python vlm_gate/scripts/judge_ab.py cmp /tmp/ab_cosmos.jsonl /tmp/ab_qwen.jsonl

**장면은 계산으로 고른다.** 내가 그림을 보고 고르면 내 기대가 그 그림에 맞춰진다.
한 태스크에서 네 국면을 뽑는다 -- 접근 · 전이 · 운반 · 놓기. 국면을 가르는 것은
그리퍼 상태와 수직 성분과 감속뿐이고, 어느 것도 문항의 답을 미리 알려주지 않는다.

기대 등급은 `analysis/judge_ab_expected.md` 에 **모델을 돌리기 전에** 적어 두었다.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
MAN = f"{BASE}/output/_gate_distill/tiles_manifest.txt"
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
SLOTS = "ABCDE"

# 문항 다섯이 서로 다르게 떠야 두 모델이 갈리는 자리가 보인다.
TASKS = ["TurnOnStove", "CoffeePressButton", "OpenDrawer",
         "CloseDoubleDoor", "PnPCounterToCab", "PnPCounterToStove"]
PHASES = ("접근", "전이", "운반", "놓기")


def vert_of(a, f, n=16):
    """수직 성분. robocasa 액션은 위치 델타가 5:8 이고 z 가 그 마지막이다.
    `robocasa_descriptors` 에는 이 값이 없어 여기서 낸다."""
    w = np.asarray(a[f:f + n], dtype=float)
    if len(w) < 2:
        return 0.0
    pos = w[:, 5:8].sum(axis=0)
    span = float(np.linalg.norm(pos)) + 1e-9
    return float(pos[2] / span)


def phase_of(x):
    """계산만으로 국면을 부른다. 문항의 답을 미리 알려주지 않는 값만 쓴다."""
    if x["grip_change"]:
        return "전이"
    closed = x["gripper_closed"] > 0.5
    down = x.get("vert", 0.0) < -0.35
    slow = x.get("decel", 0.0) > 0.5
    if down and slow:
        return "놓기" if closed else "접근"
    if closed and not down:
        return "운반"
    return "접근" if not closed else "운반"


def pick(args):
    import pandas as pd
    from robocasa_descriptors import descriptors

    ep2t, ep2n = {}, {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        t = [v for v in d.get("tasks", [])[1:]
             if isinstance(v, str) and " " not in v and v != "Valid"]
        if t:
            ep2t[d["episode_index"]] = t[0]
            ep2n[d["episode_index"]] = d.get("length", 0)

    have = set()
    for l in open(MAN):
        nm = l.strip()
        if nm:
            have.add(nm)

    out = []
    for task in TASKS:
        eps = sorted(e for e, t in ep2t.items() if t == task)[:args.episodes]
        got = defaultdict(list)
        for ep in eps:
            try:
                a = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ep // 1000:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            except Exception:
                continue
            for f in range(0, len(a) - 20, 4):
                nm = f"ep{ep:04d}_f{f:03d}"
                if nm not in have:
                    continue
                x = descriptors(a, f)
                x["vert"] = vert_of(a, f)
                ph = phase_of(x)
                if len(got[ph]) < args.per_phase:
                    got[ph].append({"task": task, "phase": ph, "ep": ep, "f": f,
                                    "tile": nm})
            if all(len(got[p]) >= args.per_phase for p in PHASES):
                break
        for p in PHASES:
            out += got[p]
        miss = [p for p in PHASES if not got[p]]
        print(f"  {task:22s} " + " ".join(f"{p}{len(got[p])}" for p in PHASES)
              + (f"   [!] {miss} 없음" if miss else ""))

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n장면 {len(out)}개 -> {args.out}")


def run(args):
    from PIL import Image
    from vlm_gate import VLMGate
    from robocasa_descriptors import descriptors, facts_text
    import pandas as pd

    scenes = json.load(open(args.scenes))
    G = open(f"{BASE}/analysis/_evolver/guidance_phase9.txt").read().strip() \
        if os.path.exists(f"{BASE}/analysis/_evolver/guidance_phase9.txt") else ""
    src = open(os.path.join(HERE, "..", "prompts", "robocasa_phase9.txt")).read()
    if not G:
        G = src.split("### GUIDANCE\n", 1)[1].split("\n### QUESTION", 1)[0].strip()
    ASK = src.split("### QUESTION\n", 1)[1].split("\n### SIGN", 1)[0].strip()

    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
        instr[d["episode_index"]] = c[0] if c else ""

    gate = VLMGate(f"http://127.0.0.1:{args.port}", timeout=600)
    acts, n, bad = {}, 0, 0
    with open(args.out, "w") as fh:
        for s in scenes:
            ep = s["ep"]
            if ep not in acts:
                acts.clear()
                acts[ep] = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ep // 1000:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            x = descriptors(acts[ep], s["f"])
            im = np.array(Image.open(f"{TILES}/{s['tile']}.png").convert("RGB"))
            h, w, _ = im.shape
            views = [Image.fromarray(im[:, k * w // 3:(k + 1) * w // 3])
                     for k in range(3)]
            r = gate.judge(views, f"{instr.get(ep, '')}\n{facts_text(x)}", G,
                           question=ASK, n_ask=5, n_grade=5, mode="text")
            picks = r.get("picks")
            ok = bool(picks) and len(picks) == 5 and all(p is not None for p in picks)
            if not ok:
                bad += 1
            fh.write(json.dumps({**s, "tag": args.tag, "ok": ok,
                                 "picks": picks, "gp": r.get("grade_probs"),
                                 "text": r.get("text", ""),
                                 "err": r.get("error", "")}, ensure_ascii=False) + "\n")
            n += 1
            if n % 8 == 0:
                print(f"  {n}/{len(scenes)}  형식실패 {bad}", flush=True)
    print(f"[{args.tag}] {n}장면, 형식실패 {bad} -> {args.out}")


def _eg(gp):
    if not gp:
        return None
    return [sum((i + 1) * float(q) for i, q in enumerate(row)) for row in gp]


def cmp_(args):
    runs = {}
    for p in args.files:
        for l in open(p):
            r = json.loads(l)
            runs.setdefault(r["tag"], {})[(r["task"], r["phase"], r["ep"], r["f"])] = r
    tags = list(runs)
    keys = sorted(set.intersection(*(set(v) for v in runs.values())))
    print(f"두 판 공통 장면 {len(keys)}개 · 판 {tags}\n")

    for t in tags:
        rs = [runs[t][k] for k in keys]
        okn = sum(r["ok"] for r in rs)
        print(f"[{t}] 형식 {okn}/{len(rs)} = {okn/len(rs):.0%}")
        grid = np.array([r["picks"] for r in rs if r["ok"]], dtype=float)
        if len(grid):
            for i, q in enumerate(SLOTS):
                c = np.bincount(grid[:, i].astype(int), minlength=6)[1:6]
                dead = [g + 1 for g in range(5) if c[g] == 0]
                print(f"    {q}  " + " ".join(f"{g+1}:{c[g]:2d}" for g in range(5))
                      + (f"   {dead} 등급 안 씀" if dead else ""))
            # 시점을 가르는가 -- 태스크 안에서 국면별 평균이 움직이는가
            spread = []
            for task in sorted({k[0] for k in keys}):
                sub = [runs[t][k] for k in keys if k[0] == task and runs[t][k]["ok"]]
                if len(sub) < 2:
                    continue
                by = defaultdict(list)
                for r in sub:
                    by[r["phase"]].append(r["picks"])
                m = {p: np.mean(v, axis=0) for p, v in by.items() if v}
                if len(m) >= 2:
                    spread.append(float(np.mean(np.ptp(np.stack(list(m.values())), axis=0))))
            if spread:
                print(f"    국면 간 등급 폭 평균 {np.mean(spread):.2f}  "
                      f"(0 이면 태스크마다 상수 — 시점을 못 가른다)")
        print()

    if len(tags) == 2:
        a, b = tags
        A = np.array([runs[a][k]["picks"] for k in keys
                      if runs[a][k]["ok"] and runs[b][k]["ok"]], dtype=float)
        B = np.array([runs[b][k]["picks"] for k in keys
                      if runs[a][k]["ok"] and runs[b][k]["ok"]], dtype=float)
        if len(A):
            print(f"두 판이 같은 등급을 준 칸 {int((A == B).sum())}/{A.size} "
                  f"= {(A == B).mean():.1%}")
            for i, q in enumerate(SLOTS):
                print(f"    {q}  일치 {(A[:, i] == B[:, i]).mean():5.1%}  "
                      f"평균 {a} {A[:, i].mean():.2f} · {b} {B[:, i].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pick"); p1.add_argument("--out", default="/tmp/ab_scenes.json")
    p1.add_argument("--episodes", type=int, default=6)
    p1.add_argument("--per-phase", type=int, default=1)
    p1.set_defaults(fn=pick)
    p2 = sub.add_parser("run")
    p2.add_argument("--port", type=int, required=True)
    p2.add_argument("--tag", required=True)
    p2.add_argument("--scenes", default="/tmp/ab_scenes.json")
    p2.add_argument("--out", required=True)
    p2.set_defaults(fn=run)
    p3 = sub.add_parser("cmp"); p3.add_argument("files", nargs="+")
    p3.set_defaults(fn=cmp_)
    a = ap.parse_args()
    a.fn(a)
