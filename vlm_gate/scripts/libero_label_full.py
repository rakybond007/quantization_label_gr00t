"""libero 전량 라벨링 (stride 16, 청크 길이와 같아 겹침 없음).

**에피 LB_BATCH(기본 24)개씩 모아 던진다.** 프로브(`libero_gemini_probe.py`)는 전
청크의 프레임을 먼저 다 모아 두고 시작하는데, 전량이면 16,288 청크 x 2장 = 5GB 가
넘는다(human_data 전량이 같은 구조로 죽었다). 반대로 **에피 하나씩** 던지면 이번엔
너무 잘게 나뉜다 -- libero 는 평균 길이 162프레임이라 에피당 청크가 10개뿐이고,
동시성 48 을 줘도 풀이 10개만 받고 나머지 시간을 디코딩 대기로 쓴다(실측 1.5 req/s).
24에피(~240청크)씩 모으면 풀이 계속 차고 메모리는 240x2장 정도만 든다.

**국면은 그리퍼 전이로 유도해 같이 적는다.** 문턱은 에피별 진폭에서 정규화한다 --
전역 상수를 쓰면 물체마다 쥐는 폭이 달라 놓치는 에피가 생긴다. libero 는 그리퍼
부호가 데이터셋마다 다를 수 있어 **닫힘을 진폭으로 정의**한다(값이 중앙에서 멀리
간 쪽, 소수인 쪽이 닫힘).

**나간 전문과 부호·가중을 출력 폴더에 남긴다.** 없으면 어떤 프롬프트로 만든
라벨인지 복구할 수 없다(robocasa 에서 실제로 그 일이 있었다).

  CHECKS=libero_v3d_checks python libero_label_full.py <출력디렉터리>
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

_CK_NAME = os.environ.get("CHECKS", "libero_v3d_checks")
CK = importlib.import_module(_CK_NAME)
_D_NAME = os.environ.get("DESCRIPTORS",
                         "libero_v3d_descriptors" if _CK_NAME == "libero_v3d_checks"
                         else "libero_descriptors")
D = importlib.import_module(_D_NAME)

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "libero_gr00t_delta")
GUID, VIEW, ASK = CK.GUIDANCE, CK.VIEW, CK.ASK
Q = tuple(sorted(CK.SIGN))
CHUNK = 16
STRIDE = int(os.environ.get("LB_STRIDE", "16"))
MODEL = os.environ.get("LB_MODEL", "gemini/gemini-3.8-flash")
CONC = int(os.environ.get("LB_CONC", "32"))
LIMIT = int(os.environ.get("LB_LIMIT", "0"))
BATCH = int(os.environ.get("LB_BATCH", "24"))
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


def png(arr, px=448):
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def phase_map(a):
    """그리퍼 전이로 국면을 **구간으로** 붙인다. 문턱은 이 에피의 진폭에서 정규화한다.

    점으로 찍으면 stride 16 격자에 거의 안 맞아 전량 라벨의 국면이 전부 null 이 된다
    (human_data 에서 눈 기록 54장 중 4장만 격자에 맞았던 것과 같은 문제).
    프레임 -> 국면 dict 를 돌려주고, 주기 밖은 넣지 않는다(= null).
    """
    g = a[:, -1]
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return {}
    mid = (lo + hi) / 2
    up = g > mid
    closed = up if up.mean() < 0.5 else ~up
    tr = np.flatnonzero(np.diff(closed.astype(int)))
    cl = [x for x in tr if closed[x + 1]]
    op = [x for x in tr if not closed[x + 1]]
    cyc = []
    for c in cl:
        nx = [o for o in op if o > c]
        if nx and (not cyc or c > cyc[-1][1]) and nx[0] - c >= 16:
            cyc.append((c, nx[0]))
    m = {}
    n = len(a)
    for c0, o0 in cyc:
        spans = [("1집으러", max(0, c0 - 16), c0),
                 ("2파지", c0, min(o0, c0 + 8)),
                 ("3운반", min(o0, c0 + 8), max(c0 + 9, o0 - 12)),
                 ("4내려놓으러", max(c0 + 9, o0 - 12), o0),
                 ("5물러남", o0, min(n, o0 + 16))]
        for ph, x0, x1 in spans:
            for f in range(int(x0), int(x1)):
                m.setdefault(f, ph)
    return m


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    open(f"{out}/PROMPT.txt", "w").write(
        "[USER]\n<2 images>\n" + GUID + "\n\n" + VIEW + "\n\n"
        "The robot was told: {instruction}\n\n{computed facts}\n\n" + ASK + "\n")
    json.dump({"checks": CK.__name__, "descriptors": D.__name__, "model": MODEL,
               "stride": STRIDE, "conc": CONC, "ngrade": CK.NGRADE,
               "questions": list(Q), "sign": CK.SIGN, "weight": CK.WEIGHT,
               "name": CK.NAME},
              open(f"{out}/meta.json", "w"), indent=1, ensure_ascii=False)

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

    info = json.load(open(f"{DS}/meta/info.json"))
    CH = info["chunks_size"]
    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        t = [x for x in d.get("tasks", []) if isinstance(x, str) and len(x.split()) > 2]
        if t:
            instr[d["episode_index"]] = t[0]
    eps = sorted(instr)
    print(f"에피 {len(eps):,}", flush=True)

    fh = open(fp, "a")
    fh_lock = threading.Lock()
    ex = ThreadPoolExecutor(CONC)

    def run(job):
        ep, f, ph, imgs, fx, ins = job
        txt, cost = ask_one(imgs, fx, ins)
        g = parse(txt)
        rec = {"ep": int(ep), "f": int(f), "phase": ph, **(g or {})}
        with fh_lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
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

    nep = 0
    buf = []
    for ep in eps:
        if LIMIT and STAT["n"] >= LIMIT:
            break
        ch = ep // CH
        try:
            a = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            continue
        want = [f for f in range(0, max(len(a) - CHUNK, 0) + 1, STRIDE)
                if (ep, f) not in done]
        if not want:
            continue
        pm = phase_map(a)
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
        buf += [(ep, f, pm.get(f), [frames[("scene", f)], frames[("wrist", f)]],
                 D.facts_text(D.descriptors(a, f)), instr[ep])
                for f in want
                if ("scene", f) in frames and ("wrist", f) in frames]
        frames.clear()
        nep += 1
        if nep % BATCH == 0 and buf:
            list(ex.map(run, buf))
            buf = []
            print(f"  에피 {nep:,}/{len(eps):,} 처리", flush=True)
    if buf:
        list(ex.map(run, buf))
    ex.shutdown()
    fh.close()
    el = time.time() - STAT["t0"]
    print(f"완료 {STAT['n']:,} · 실패 {STAT['bad']} · ${STAT['cost']:.2f} · "
          f"{el/3600:.2f}시간 -> {fp}", flush=True)


if __name__ == "__main__":
    main()
