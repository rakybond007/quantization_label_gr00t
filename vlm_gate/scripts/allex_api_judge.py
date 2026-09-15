"""Anthropic API 를 판정기로 쓴다. 파이프라인은 그대로, 판정기만 바뀐다.

Cosmos 판정 서버와 같은 것을 받는다 -- 청크 하나의 좌우 두 뷰 + 그 청크의 계산
사실 + GUIDANCE + 문항 -> 등급 네 개. 청크마다 한 번 부르므로 앞뒤 맥락이 없다
(격자로 묶어 주면 앞뒤가 보여 유리해진다 -- 그러면 비교가 안 된다).

**키는 레포에 두지 않는다.** `~/.config/anthropic/key` (chmod 600) 에서 읽는다.

    python allex_api_judge.py <장면목록.json> <출력.jsonl> [모델]
"""
import base64, io, json, os, sys, time, urllib.error, urllib.request
import numpy as np, pandas as pd, av
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
from allex_v2_common import descriptors
import allex_v4c_checks as CK
from allex_facts import facts as _facts


D = os.environ.get("ALLEX_DS", "/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
SCENES, OUT = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "claude-sonnet-4-5"
KEY = open(os.path.expanduser("~/.config/anthropic/key")).read().strip()
Q = tuple(sorted(CK.SIGN))


def b64(im):
    b = io.BytesIO(); im.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def ask(imgs, facts):
    content = [{"type": "image", "source": {"type": "base64",
                "media_type": "image/png", "data": b64(i)}} for i in imgs]
    content.append({"type": "text", "text":
        f"{CK.GUIDANCE}\n\nThe two images are the left and right camera views of "
        f"this one moment.\n\n{facts}\n\n{CK.ASK}"})
    body = json.dumps({"model": MODEL, "max_tokens": 64,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["content"][0]["text"], r.get("usage", {})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 529) and attempt < 3:
                time.sleep(4 * (attempt + 1)); continue
            return f"ERR {e.code} {e.read().decode()[:160]}", {}
        except Exception as e:
            if attempt < 3: time.sleep(4); continue
            return f"ERR {type(e).__name__}", {}


def parse(text):
    g = {}
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= 3 and s[0] in Q and s[1] == ")":
            d = "".join(c for c in s[2:] if c.isdigit())
            if d: g[s[0]] = max(1, min(CK.NGRADE, int(d[0])))
    return g if len(g) == len(Q) else None


scenes = json.load(open(SCENES))
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(tuple(json.loads(l)["key"]))
        except Exception: pass
cache = {}
out = open(OUT, "a")
tin = tout = nbad = n = 0
t0 = time.time()
for sc in scenes:
    ep, f = sc["ep"], sc["f"]
    if (ep, f) in done: continue
    if ep not in cache:
        d = pd.read_parquet(f"{D}/data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet")
        cache[ep] = (np.stack(d["action"].values),
                     np.stack(d["action.right_wrist_wrt_base"].values),
                     np.stack(d["action.left_wrist_wrt_base"].values))
    A, WR, WL = cache[ep]
    imgs = []
    for side in ("left", "right"):
        p = (f"{D}/videos/chunk-{ep//1000:03d}/observation.images.camera_ego_{side}/"
             f"episode_{ep:06d}.mp4")
        with av.open(p) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i == f:
                    imgs.append(fr.to_image().resize((512, 512))); break
    if len(imgs) < 2: continue
    x = descriptors(A, WR, WL, f, 16)
    txt, us = ask(imgs, _facts(x))
    tin += us.get("input_tokens", 0); tout += us.get("output_tokens", 0)
    g = parse(txt)
    if not g:
        nbad += 1; continue
    rec = {"key": [ep, f], "ep": ep, "f": f, "model": MODEL, **g,
           "conf": round(0.5 * (1 + sum(CK.SIGN[q] * CK.WEIGHT[q] * ((g[q] - 1) / 4)
                                        for q in Q)), 4),
           "one_handed": bool(x["one_handed"]),
           "merge_demand_k2": round(float(x["merge_demand_k2"]), 4)}
    out.write(json.dumps(rec) + "\n"); out.flush(); n += 1
    if n % 10 == 0:
        print(f"  {n} · 입력 {tin} 출력 {tout} 토큰 · {time.time()-t0:.0f}s", flush=True)
out.close()
print(f"완료 {n} · 형식실패 {nbad} · 입력 {tin} 출력 {tout} 토큰 · {time.time()-t0:.0f}s -> {OUT}")
