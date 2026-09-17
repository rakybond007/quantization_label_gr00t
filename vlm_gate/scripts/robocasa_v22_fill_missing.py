"""타일이 없어 빠진 robocasa 에피소드를 **영상에서 직접** 라벨한다.

`robocasa_label_full_gemini.py` 는 미리 만들어 둔 타일 매니페스트
(`output/_gate_distill/tiles_manifest.txt`)를 작업 목록으로 쓴다. 그런데 그 매니페스트가
**7,200 에피 중 6,993 만 덮는다** -- 207 에피에 타일이 아예 없다. 매니페스트가 전량을
덮는지 확인하지 않고 그대로 목록으로 쓴 것이 원인이다.

빠진 207 에피는 세 태스크에 몰려 있어 **태스크 편향**이다:

    CloseSingleDoor  300 중 75 (25%)
    CoffeeServeMug   300 중 75 (25%)
    OpenDoubleDoor   300 중 57 (19%)

태스크별 가중 재적합(R5)에 직접 영향이 가므로 메꾼다. 여기서는 타일 대신 세 영상
(left_view · right_view · wrist_view)을 직접 디코딩해 같은 3장을 보낸다 -- 타일은
384x128 을 3등분한 것이므로 내용이 같다.

**에피소드 하나씩 처리한다.** 전 에피 프레임을 모아 두면 메모리가 는다.

  CHECKS=phase9_checks_v22 python robocasa_v22_fill_missing.py <출력디렉터리>
"""
import base64
import importlib
import io
import json
import os
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

CK = importlib.import_module(os.environ.get("CHECKS", "phase9_checks_v22"))
D = importlib.import_module(os.environ.get("DESCRIPTORS", "robocasa_v22_descriptors"))
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
MAN = f"{BASE}/output/_gate_distill/tiles_manifest.txt"
VIEW = CK.VIEW
Q = tuple(sorted(CK.SIGN))
STRIDE = int(os.environ.get("RC_STRIDE", "16"))
MODEL = os.environ.get("RC_MODEL", "gemini/gemini-3.8-flash")
CONC = int(os.environ.get("RC_CONC", "48"))
KEY = open(os.path.expanduser("~/.config/litellm/key")).read().strip()
URL = (open(os.path.expanduser("~/.config/litellm/base")).read().strip()
       + "/v1/chat/completions")
_lock = threading.Lock()
STAT = {"n": 0, "bad": 0, "cost": 0.0, "t0": time.time()}
VKEYS = ("left_view", "right_view", "wrist_view")


def ask_one(imgs, fx, instr):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64,"
                                     + base64.b64encode(i).decode()}}
               for i in imgs]
    content.append({"type": "text",
                    "text": f"{CK.GUIDANCE}\n\n{VIEW}\n\nThe robot was told: "
                            f"{instr}\n\n{fx}\n\n{CK.ASK}"})
    body = {"model": MODEL, "max_tokens": 1024, "temperature": 0,
            "reasoning_effort": os.environ.get("RC_EFFORT", "low"),
            "messages": [{"role": "user", "content": content}]}
    data = json.dumps(body).encode()
    for a in range(6):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {KEY}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                cost = float(resp.headers.get("x-litellm-response-cost") or 0.0)
                j = json.load(resp)
            return j["choices"][0]["message"]["content"], cost
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < 5:
                time.sleep(3 * (a + 1))
                continue
            return None, 0.0
        except Exception:
            if a < 5:
                time.sleep(3 * (a + 1))
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


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    info = json.load(open(f"{DS}/meta/info.json"))
    CH = info["chunks_size"]
    have = set()
    for l in open(MAN):
        s = l.strip()
        if not s:
            continue
        s = s[:-4] if s.endswith(".png") else s
        try:
            have.add(int(s.split("ep")[1].split("_")[0]))
        except Exception:
            pass
    missing = sorted(set(range(info["total_episodes"])) - have)
    print(f"타일 없는 에피 {len(missing):,}", flush=True)

    instr = {}
    for ln in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(ln)
        c = [t for t in d.get("tasks", [])
             if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
        if c:
            instr[d["episode_index"]] = c[0]

    fp = f"{out}/labels.jsonl"
    done = set()
    if os.path.exists(fp):
        for l in open(fp):
            try:
                r = json.loads(l)
                done.add((r["ep"], r["f"]))
            except Exception:
                pass
    print(f"재개 {len(done):,}", flush=True)
    open(f"{out}/PROMPT.txt", "w").write(
        "[USER]\n<3 images>\n" + CK.GUIDANCE + "\n\n" + VIEW + "\n\n"
        "The robot was told: {instruction}\n\n{computed facts}\n\n" + CK.ASK + "\n")
    json.dump({"checks": CK.__name__, "descriptors": D.__name__, "model": MODEL,
               "stride": STRIDE, "conc": CONC, "ngrade": CK.NGRADE,
               "note": "타일이 없어 빠졌던 207 에피를 영상에서 직접 라벨",
               "sign": CK.SIGN, "weight": CK.WEIGHT},
              open(f"{out}/meta.json", "w"), indent=1, ensure_ascii=False)

    fh = open(fp, "a")
    fh_lock = threading.Lock()
    ex = ThreadPoolExecutor(CONC)

    def run(job):
        ep, f, imgs, fx, ins = job
        txt, cost = ask_one(imgs, fx, ins)
        g = parse(txt)
        rec = {"ep": int(ep), "f": int(f), **(g or {})}
        with fh_lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        with _lock:
            STAT["n"] += 1
            STAT["cost"] += cost
            if g is None:
                STAT["bad"] += 1
            if STAT["n"] % 500 == 0:
                el = time.time() - STAT["t0"]
                print(f"  {STAT['n']:,} · ${STAT['cost']:.2f} · {STAT['n']/el:.1f} req/s"
                      f" · 실패 {STAT['bad']}", flush=True)

    for i, ep in enumerate(missing):
        ch = ep // CH
        try:
            a = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            continue
        want = [f for f in range(0, len(a) - 4, STRIDE) if (ep, f) not in done]
        if not want:
            continue
        frames = {}
        for key in VKEYS:
            p = (f"{DS}/videos/chunk-{ch:03d}/observation.images.{key}/"
                 f"episode_{ep:06d}.mp4")
            need = set(want)
            try:
                c = av.open(p)
                for k, fr in enumerate(c.decode(video=0)):
                    if k in need:
                        frames[(key, k)] = png(fr.to_ndarray(format="rgb24"))
                        need.discard(k)
                        if not need:
                            break
                c.close()
            except Exception:
                pass
        jobs = [(ep, f, [frames[(k, f)] for k in VKEYS],
                 D.facts_text(D.descriptors(a, f)), instr.get(ep, ""))
                for f in want if all((k, f) in frames for k in VKEYS)]
        if jobs:
            list(ex.map(run, jobs))
        frames.clear()
        if (i + 1) % 25 == 0:
            print(f"  에피 {i+1}/{len(missing)}", flush=True)
    ex.shutdown()
    fh.close()
    el = time.time() - STAT["t0"]
    print(f"완료 {STAT['n']:,} · 실패 {STAT['bad']} · ${STAT['cost']:.2f} · "
          f"{el/60:.1f}분 -> {fp}", flush=True)


if __name__ == "__main__":
    main()
