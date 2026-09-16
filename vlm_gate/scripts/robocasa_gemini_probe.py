"""robocasa 문항 판을 Gemini 로 표본 판정한다. **국면 정답이 있는 타일 위에서** 돈다.

`analysis/ratio_prompt/phase_class.jsonl` 에 사람이 찍은 국면이 762타일 있다
(접근 240 · 파지 180 · 운반 180 · 해제 162). 그 위에서 판을 비교하면 국면이
의도한 순서로 나오는지 볼 수 있다.

타일은 384x128 한 장이고 **3등분해서 3장으로 보낸다** -- 라벨러(phase9_two_sided.py)
가 하는 것과 같다.

  CHECKS=phase9_checks_v8 python robocasa_gemini_probe.py out.jsonl
  CHECKS=phase9_checks_v9 RC_N=380 python robocasa_gemini_probe.py out.jsonl
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

import numpy as np
import pandas as pd
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")

CK = importlib.import_module(os.environ.get("CHECKS", "phase9_checks_v8"))
# 계산 사실: v8 이상은 기간 문구를 뺀 판을 쓴다
_D = ("robocasa_v8_descriptors"
      if os.environ.get("CHECKS", "phase9_checks_v8") != "phase9_checks_v7"
      else "robocasa_descriptors")
D = importlib.import_module(_D)

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
PHASE = f"{BASE}/analysis/ratio_prompt/phase_class.jsonl"
VIEW = ("You are shown 3 camera views: agentview-left, agentview-right, "
        "and a wrist (eye-in-hand) close-up. The wrist camera is mounted on "
        "the gripper, so objects normally look close in it — general closeness "
        "is normal. Use the wrist view only to spot the actual grasp-closure or "
        "fine-insertion instant.")

Q = tuple(sorted(CK.SIGN))
MODEL = os.environ.get("RC_MODEL", "gemini/gemini-3.8-flash")
N = int(os.environ.get("RC_N", "380"))
CONC = int(os.environ.get("RC_CONC", "16"))
KEY = open(os.path.expanduser("~/.config/litellm/key")).read().strip()
URL = (open(os.path.expanduser("~/.config/litellm/base")).read().strip()
       + "/v1/chat/completions")
_lock = threading.Lock()
STAT = {"n": 0, "bad": 0, "cost": 0.0, "t0": time.time()}


def ask_one(imgs, fx, instr):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64,"
                                     + base64.b64encode(i).decode()}}
               for i in imgs]
    content.append({"type": "text",
                    "text": f"{CK.GUIDANCE}\n\n{VIEW}\n\nThe robot was told: "
                            f"{instr}\n\n{fx}\n\n{CK.ASK}"})
    body = {"model": MODEL, "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}]}
    if not any(k in MODEL for k in ("opus-5", "opus-4-7", "opus-4-8", "fable")):
        body["temperature"] = 0
    if "gemini" in MODEL or "gpt" in MODEL:
        body["reasoning_effort"] = os.environ.get("RC_EFFORT", "low")
    data = json.dumps(body).encode()
    for a in range(5):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {KEY}",
                "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                cost = float(resp.headers.get("x-litellm-response-cost") or 0.0)
                j = json.load(resp)
            return j["choices"][0]["message"]["content"], cost
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < 4:
                time.sleep(4 * (a + 1))
                continue
            return None, 0.0
        except Exception:
            if a < 4:
                time.sleep(4 * (a + 1))
                continue
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


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(b, format="PNG")
    return b.getvalue()


def collect():
    info = json.load(open(f"{DS}/meta/info.json"))
    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        c = [t for t in d.get("tasks", [])
             if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
        if c:
            instr[d["episode_index"]] = c[0]
    rows = []
    seen = set()
    for line in open(PHASE):
        r = json.loads(line)
        t = r["tile"]
        if t in seen:
            continue
        seen.add(t)
        rows.append((t, r["true"], r.get("task")))
    random.Random(0).shuffle(rows)
    rows = rows[:N]
    acts = {}
    jobs = []
    for tile, ph, task in rows:
        ep = int(tile.split("ep")[1].split("_")[0])
        fr = int(tile.split("_f")[1])
        if ep not in acts:
            ch = ep // info["chunks_size"]
            try:
                acts[ep] = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet"
                )["action"].values)
            except Exception:
                acts[ep] = None
        a = acts[ep]
        if a is None or fr >= len(a) - 4:
            continue
        try:
            im = np.array(Image.open(f"{TILES}/{tile}.png").convert("RGB"))
        except Exception:
            continue
        h, w, _ = im.shape
        views = [png(im[:, k * w // 3:(k + 1) * w // 3]) for k in range(3)]
        x = D.descriptors(a, fr)
        # 그리퍼가 아직 닫혀 있나 -- "쥔 채 내려놓는 중" 과 "이미 놓음" 을 가른다.
        # 사람 국면 라벨은 그 둘을 "해제" 하나로 묶는다.
        held = bool((a[fr:fr + 16, -1] > 0.5).mean() > 0.5)
        jobs.append((tile, ep, fr, ph, task, held, views,
                     D.facts_text(x), instr.get(ep, "")))
    return jobs


def run(job):
    tile, ep, fr, ph, task, held, views, fx, instr = job
    txt, cost = ask_one(views, fx, instr)
    g = parse(txt)
    with _lock:
        STAT["n"] += 1
        STAT["cost"] += cost
        if g is None:
            STAT["bad"] += 1
        if STAT["n"] % 100 == 0:
            el = time.time() - STAT["t0"]
            print(f"  {STAT['n']} · ${STAT['cost']:.2f} · {STAT['n'] / el:.1f} req/s",
                  flush=True)
    return {"tile": tile, "ep": ep, "f": fr, "phase": ph, "task": task,
            "held": held, **(g or {})}


def main():
    out = sys.argv[1]
    jobs = collect()
    from collections import Counter
    print(f"[robocasa] 청크 {len(jobs)} · 문항 {''.join(Q)} · "
          f"국면 {dict(Counter(j[3] for j in jobs))}", flush=True)
    with ThreadPoolExecutor(CONC) as ex:
        recs = list(ex.map(run, jobs))
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    el = time.time() - STAT["t0"]
    print(f"완료 {STAT['n']} · 실패 {STAT['bad']} · ${STAT['cost']:.2f} · "
          f"{el / 60:.1f}분 -> {out}", flush=True)


if __name__ == "__main__":
    main()
