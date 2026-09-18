"""openarm 문항 판을 Gemini 로 표본 판정한다.

정답(배속 사다리)이 없다. 그래서 확인하는 것은 둘이다:
  1. 국면 순서가 conf 에 나타나는가 (analysis/openarm/PREDICTION.md 에 미리 적었다)
  2. 문항이 죽는가 -- **4등급 이상 비율이 균일하게 낮으면** 죽은 문항이고,
     해당 국면/태스크에 몰려 있으면 정상이다. libero 에서 이 구분을 놓쳐 잘 도는
     문항을 지울 뻔했다.

국면은 액션에서 유도한다(왼손 개폐 = 도구 파지/해제). 두 태스크 모두 왼손이 도구를
쥐므로 왼손을 쓴다. 오른손은 수세미 태스크에서 거의 안 움직여 기준이 못 된다.

  CHECKS=openarm_v1_checks OA_N=300 python openarm_gemini_probe.py out.jsonl
"""
import base64
import importlib
import io
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pandas as pd
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")
CK = importlib.import_module(os.environ.get("CHECKS", "openarm_v1_checks"))
D = importlib.import_module(os.environ.get("DESCRIPTORS", "openarm_descriptors"))

ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data/openarm/lerobot"
Q = tuple(sorted(CK.SIGN))
CHUNK = 16
STRIDE = int(os.environ.get("OA_STRIDE", "16"))
N = int(os.environ.get("OA_N", "300"))
CONC = int(os.environ.get("OA_CONC", "12"))
MODEL = os.environ.get("OA_MODEL", "gemini/gemini-3.8-flash")
KEY = open(os.path.expanduser("~/.config/litellm/key")).read().strip()
URL = (open(os.path.expanduser("~/.config/litellm/base")).read().strip()
       + "/v1/chat/completions")
_lock = threading.Lock()
STAT = {"n": 0, "bad": 0, "cost": 0.0, "t0": time.time()}


def ask_one(imgs, fx, instr):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(i).decode()}}
               for i in imgs]
    content.append({"type": "text",
                    "text": f"{CK.GUIDANCE}\n\n{CK.VIEW}\n\nThe robot was told: {instr}\n\n"
                            f"{fx}\n\n{CK.ASK}"})
    body = {"model": MODEL, "max_tokens": 1024, "temperature": 0,
            "reasoning_effort": os.environ.get("OA_EFFORT", "low"),
            "messages": [{"role": "user", "content": content}]}
    data = json.dumps(body).encode()
    for a in range(5):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {KEY}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                cost = float(resp.headers.get("x-litellm-response-cost") or 0.0)
                j = json.load(resp)
            return j["choices"][0]["message"]["content"], cost
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < 4:
                time.sleep(4 * (a + 1)); continue
            return None, 0.0
        except Exception:
            if a < 4:
                time.sleep(4 * (a + 1)); continue
            return None, 0.0


def parse(t):
    g = {}
    for line in (t or "").splitlines():
        s = line.strip().lstrip("*# ")
        if len(s) >= 3 and s[0] in Q and s[1] == ")":
            d = "".join(c for c in s[2:] if c.isdigit())
            if d:
                g[s[0]] = max(1, min(CK.NGRADE, int(d[0])))
    return g if len(g) == len(Q) else None


def png(arr, px=448):
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()


def phase_map(a, group):
    """왼손 개폐로 국면 구간. 문턱은 이 에피의 진폭에서 정규화한다."""
    g = a[:, 14:20].mean(1)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 0.15:
        return {}
    th = lo + 0.5 * (hi - lo)
    closed = g > th
    tr = np.flatnonzero(np.diff(closed.astype(int)))
    cl = [x for x in tr if closed[x + 1]]
    op = [x for x in tr if not closed[x + 1]]
    m = {}
    n = len(a)
    for c0 in cl:
        nx = [o for o in op if o > c0]
        o0 = nx[0] if nx else n - 1
        if o0 - c0 < 30:
            continue
        for ph, x0, x1 in (("1접근", max(0, c0 - 30), c0),
                           ("2도구파지", c0, min(o0, c0 + 15)),
                           ("3작업", min(o0, c0 + 15), max(c0 + 16, o0 - 15)),
                           ("4마무리", max(c0 + 16, o0 - 15), o0),
                           ("5해제", o0, min(n, o0 + 30))):
            for f in range(int(x0), int(x1)):
                m.setdefault(f, ph)
        break
    return m


def main():
    out = sys.argv[1]
    info = json.load(open(f"{ROOT}/meta/info.json")); CH = info["chunks_size"]
    task, instr = {}, {}
    for ln in open(f"{ROOT}/meta/episodes.jsonl"):
        d = json.loads(ln)
        t = [x for x in d.get("tasks", []) if isinstance(x, str) and len(x.split()) > 3]
        if t:
            task[d["episode_index"]] = "dustpan" if "dustpan" in t[0] else "scrub"
            instr[d["episode_index"]] = t[0]
    eps = sorted(task)
    random.Random(0).shuffle(eps)
    jobs = []
    for ep in eps:
        a = np.stack(pd.read_parquet(
            f"{ROOT}/data/chunk-{ep//CH:03d}/episode_{ep:06d}.parquet")["action"].values)
        grp = task[ep]; pm = phase_map(a, grp)
        want = list(range(0, max(len(a) - CHUNK, 0) + 1, STRIDE))
        random.Random(ep).shuffle(want)
        want = sorted(want[:6])
        frames = {}
        for key, slot in (("ego_left", 0), ("ego_right", 1)):
            p = f"{ROOT}/videos/chunk-{ep//CH:03d}/observation.images.{key}/episode_{ep:06d}.mp4"
            need = set(want)
            try:
                c = av.open(p)
                for i, fr in enumerate(c.decode(video=0)):
                    if i in need:
                        frames[(slot, i)] = png(fr.to_ndarray(format="rgb24")); need.discard(i)
                        if not need: break
                c.close()
            except Exception:
                pass
        for f in want:
            if (0, f) in frames and (1, f) in frames:
                x = D.descriptors(a, f, group=grp)
                jobs.append((ep, f, grp, pm.get(f), [frames[(0, f)], frames[(1, f)]],
                             D.facts_text(x), instr[ep]))
        if len(jobs) >= N:
            break
    jobs = jobs[:N]
    from collections import Counter
    print(f"[openarm] 청크 {len(jobs)} · 문항 {''.join(Q)} · "
          f"태스크 {dict(Counter(j[2] for j in jobs))} · "
          f"국면 {dict(Counter(j[3] for j in jobs))}", flush=True)

    def run(job):
        ep, f, grp, ph, imgs, fx, ins = job
        txt, cost = ask_one(imgs, fx, ins)
        g = parse(txt)
        with _lock:
            STAT["n"] += 1; STAT["cost"] += cost
            if g is None: STAT["bad"] += 1
            if STAT["n"] % 100 == 0:
                print(f"  {STAT['n']} · ${STAT['cost']:.2f}", flush=True)
        return {"ep": int(ep), "f": int(f), "group": grp, "phase": ph, **(g or {})}

    with ThreadPoolExecutor(CONC) as ex:
        recs = list(ex.map(run, jobs))
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"완료 {STAT['n']} · 실패 {STAT['bad']} · ${STAT['cost']:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
