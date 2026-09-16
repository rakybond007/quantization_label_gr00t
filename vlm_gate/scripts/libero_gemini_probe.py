"""libero 문항 판을 Gemini 로 표본 판정한다.

**국면 정답이 없다.** robocasa 는 사람이 찍은 762타일이 있지만 libero 는 없다. 그래서
human_data 에서 한 것처럼 **그리퍼 전이로 국면을 유도**한다. 문턱은 에피소드별로
정규화한다 -- 전역 상수를 쓰면 물체마다 쥐는 폭이 달라 놓치는 에피가 생긴다.

정답은 `analysis/subaction/libero_limits.json` 의 실측 국면 상한이다:

    carry 378.94 · place_catcher 8.53 · place_precise 2.58
    grasp_open 2.26 · grasp_handle 2.00 · grasp_confined/between 1.67

`place_open` 은 7.5e9 로 상한을 못 찾은 퇴화값이라 쓰지 않는다.

  CHECKS=libero_v3c_checks python libero_gemini_probe.py out.jsonl
  CHECKS=libero_v4_checks  LB_N=380 python libero_gemini_probe.py out.jsonl
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

CK = importlib.import_module(os.environ.get("CHECKS", "libero_v3c_checks"))
# 계산 사실도 판본에 따라 고른다. v3d 는 속도를 백분위로 주고 단위 없는 raw 숫자를
# 뺀 판이다(libero 절대 임계 세 구간의 실측 분포가 3.6/50.1/46.3 으로 뭉개져 있었다).
_D = os.environ.get("DESCRIPTORS",
                    "libero_v3d_descriptors" if os.environ.get("CHECKS") == "libero_v3d_checks"
                    else "libero_descriptors")
D = importlib.import_module(_D)

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "libero_gr00t_delta")
# **VIEW 는 판본이 가지고 있으면 그것을 쓴다.** 아래 문장은 v3c 까지 쓰던 것이고
# robocasa 에 그대로 넘어가 있었다(거기엔 삽입 태스크가 없다). v3d 는 판본이 든다.
_VIEW_LEGACY = ("You are shown 2 camera views of this one moment: a scene view and a wrist "
        "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
        "normally look close in it -- general closeness is normal. Use the wrist view "
        "only to spot the actual grasp-closure or fine-insertion instant.")
VIEW = getattr(CK, "VIEW", _VIEW_LEGACY)

# v3c 는 문항 상수가 파일에 있고 모듈에는 SIGN/WEIGHT 만 있다.
_EV = f"{BASE}/analysis/_evolver/_libero"
GUID = getattr(CK, "GUIDANCE", None) or open(f"{_EV}/libero_guidance_v3c.txt").read().strip()
ASK = getattr(CK, "ASK", None) or open(f"{_EV}/libero_questions_v3c.txt").read().strip()

Q = tuple(sorted(CK.SIGN))
MODEL = os.environ.get("LB_MODEL", "gemini/gemini-3.8-flash")
N = int(os.environ.get("LB_N", "380"))
CONC = int(os.environ.get("LB_CONC", "16"))
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
                    "text": f"{GUID}\n\n{VIEW}\n\nThe robot was told: {instr}\n\n"
                            f"{fx}\n\n{ASK}"})
    body = {"model": MODEL, "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}]}
    if not any(k in MODEL for k in ("opus-5", "opus-4-7", "opus-4-8", "fable")):
        body["temperature"] = 0
    if "gemini" in MODEL or "gpt" in MODEL:
        body["reasoning_effort"] = os.environ.get("LB_EFFORT", "low")
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


def png(arr, px=448):
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def phases(a):
    """그리퍼 전이로 국면을 찍는다. 문턱은 이 에피의 진폭에서 정규화한다.

    libero 그리퍼 부호는 데이터셋마다 다를 수 있어 **닫힘을 진폭으로 정의**한다 --
    값이 중앙에서 멀리 간 쪽을 닫힘으로 본다.
    """
    g = a[:, -1]
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return {}
    mid = (lo + hi) / 2
    # 소수 쪽이 닫힘이다(대부분의 시간은 열려 있다)
    up = g > mid
    closed = up if up.mean() < 0.5 else ~up
    tr = np.flatnonzero(np.diff(closed.astype(int)))
    cl = [x for x in tr if closed[x + 1]]
    op = [x for x in tr if not closed[x + 1]]
    cyc = []
    for c in cl:
        nx = [o for o in op if o > c]
        if nx and (not cyc or c > cyc[-1][1]) and nx[0] - c >= 24:
            cyc.append((c, nx[0]))
    m = {}
    for c0, o0 in cyc:
        m[max(0, c0 - 12)] = "1집으러"
        m[c0] = "2파지"
        m[(c0 + o0) // 2] = "3운반"
        m[max(c0 + 1, o0 - 12)] = "4내려놓으러"
        m[o0] = "5물러남"
    return m


def collect():
    info = json.load(open(f"{DS}/meta/info.json"))
    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        t = [x for x in d.get("tasks", []) if isinstance(x, str) and len(x.split()) > 2]
        if t:
            instr[d["episode_index"]] = t[0]
    eps = sorted(instr)
    random.Random(0).shuffle(eps)
    jobs = []
    for ep in eps:
        ch = ep // info["chunks_size"]
        try:
            a = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            continue
        pm = phases(a)
        want = [f for f in sorted(pm) if f < len(a) - 20]
        if not want:
            continue
        frames = {}
        for key, slot in (("front_view", "scene"), ("left_wrist_view", "wrist")):
            p = (f"{DS}/videos/chunk-{ch:03d}/observation.images.{key}/"
                 f"episode_{ep:06d}.mp4")
            need = set(want)
            try:
                c = av.open(p)
                for i, fr in enumerate(c.decode(video=0)):
                    if i in need:
                        frames[(slot, i)] = png(fr.to_ndarray(format="rgb24"))
                        need.discard(i)
                        if not need:
                            break
                c.close()
            except Exception:
                pass
        for f in want:
            if ("scene", f) in frames and ("wrist", f) in frames:
                x = D.descriptors(a, f)
                jobs.append((ep, f, pm[f], [frames[("scene", f)], frames[("wrist", f)]],
                             D.facts_text(x), instr[ep]))
        if len(jobs) >= N:
            break
    return jobs[:N]


def run(job):
    ep, f, ph, imgs, fx, instr = job
    txt, cost = ask_one(imgs, fx, instr)
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
    return {"ep": int(ep), "f": int(f), "phase": ph, "instr": instr, **(g or {})}


def main():
    out = sys.argv[1]
    jobs = collect()
    from collections import Counter
    print(f"[libero] 청크 {len(jobs)} · 문항 {''.join(Q)} · "
          f"국면 {dict(Counter(j[2] for j in jobs))}", flush=True)
    with ThreadPoolExecutor(CONC) as ex:
        recs = list(ex.map(run, jobs))
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"완료 {STAT['n']} · 실패 {STAT['bad']} · ${STAT['cost']:.2f} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
