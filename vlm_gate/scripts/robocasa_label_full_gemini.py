"""robocasa 전량 라벨링 (v10 문항, Gemini, stride 16).

**타일을 그대로 쓴다.** `output/_gate_distill/luna_robocasa_full/tiles` 에 8프레임마다
384x128 타일이 260,031장 있고, 그 중 f%16==0 인 131,768장이 stride 16 을 덮는다.
영상을 다시 디코딩하면 에피당 두 번씩 훑어야 해서 훨씬 느리다.

타일은 3등분해서 3장으로 보낸다 -- 라벨러(phase9_two_sided.py)와 같다.

**나간 전문을 출력 폴더에 남긴다.** 없으면 어떤 프롬프트로 만든 라벨인지 복구할 수
없다 -- 09-05 의 204만 청크가 그래서 복구 불가다.

  CHECKS=phase9_checks_v10 python robocasa_label_full_gemini.py <출력디렉터리>
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

import numpy as np
import pandas as pd
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")

CK = importlib.import_module(os.environ.get("CHECKS", "phase9_checks_v10"))
# v22 는 속도 줄의 단위 없는 raw 숫자를 뺀 계산 사실을 쓴다.
_DEF_D = ("robocasa_v22_descriptors"
          if os.environ.get("CHECKS") == "phase9_checks_v22"
          else "robocasa_v8_descriptors")
D = importlib.import_module(os.environ.get("DESCRIPTORS", _DEF_D))

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
MAN = f"{BASE}/output/_gate_distill/tiles_manifest.txt"
# **VIEW 는 판본이 가지고 있으면 그것을 쓴다.** 여기 하드코딩된 문장은
# libero 의 것이 그대로 넘어온 것이다 -- robocasa 에 삽입 태스크가 없는데
# "fine-insertion instant" 를 찾으라 하고, 손목 뷰를 파지 닫힘에만 쓰라고
# 묶어 내려놓는 순간(F)을 배제한다. v22 부터 판본이 고친 VIEW 를 들고 있다.
# 옛 판을 다시 돌리면 아래 문장이 그대로 쓰여 프로브 결과가 재현된다.
_VIEW_LEGACY = ("You are shown 3 camera views: agentview-left, agentview-right, "
        "and a wrist (eye-in-hand) close-up. The wrist camera is mounted on "
        "the gripper, so objects normally look close in it — general closeness "
        "is normal. Use the wrist view only to spot the actual grasp-closure or "
        "fine-insertion instant.")
VIEW = getattr(CK, "VIEW", _VIEW_LEGACY)

Q = tuple(sorted(CK.SIGN))
STRIDE = int(os.environ.get("RC_STRIDE", "16"))
MODEL = os.environ.get("RC_MODEL", "gemini/gemini-3.8-flash")
CONC = int(os.environ.get("RC_CONC", "32"))
LIMIT = int(os.environ.get("RC_LIMIT", "0"))
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
    for a in range(6):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {KEY}",
                "content-type": "application/json"})
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


def worklist(done):
    """f%STRIDE==0 인 타일만. **에피소드를 섞어 돈다** -- 한 태스크만 먼저 끝나면
    중간에 멈췄을 때 태스크 편향이 남는다."""
    names = [ln.strip()[:-4] if ln.strip().endswith(".png") else ln.strip()
             for ln in open(MAN) if ln.strip()]
    out = []
    for nm in names:
        try:
            ep = int(nm.split("ep")[1].split("_")[0])
            f = int(nm.split("_f")[1])
        except Exception:
            continue
        if f % STRIDE == 0 and (ep, f) not in done:
            out.append((nm, ep, f))
    import random
    random.Random(0).shuffle(out)
    return out[:LIMIT] if LIMIT else out


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    open(f"{out_dir}/PROMPT.txt", "w").write(
        "[USER]\n<3 images>\n" + CK.GUIDANCE + "\n\n" + VIEW + "\n\n"
        "The robot was told: {instruction}\n\n{computed facts}\n\n" + CK.ASK + "\n")
    json.dump({"checks": CK.__name__,
               "descriptors": D.__name__,
               "model": MODEL, "stride": STRIDE, "conc": CONC,
               "sign": CK.SIGN, "weight": CK.WEIGHT, "ngrade": CK.NGRADE},
              open(f"{out_dir}/meta.json", "w"), indent=1)

    fp = f"{out_dir}/labels.jsonl"
    done = set()
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                r = json.loads(ln)
                done.add((r["ep"], r["f"]))
            except Exception:
                pass
    print(f"재개 {len(done):,}", flush=True)

    info = json.load(open(f"{DS}/meta/info.json"))
    instr = {}
    for ln in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(ln)
        c = [t for t in d.get("tasks", [])
             if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
        if c:
            instr[d["episode_index"]] = c[0]

    jobs = worklist(done)
    print(f"남은 청크 {len(jobs):,}", flush=True)
    acts = {}
    acts_lock = threading.Lock()

    def actions(ep):
        with acts_lock:
            if ep in acts:
                return acts[ep]
        ch = ep // info["chunks_size"]
        try:
            a = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            a = None
        with acts_lock:
            # 에피 7천 개의 액션을 다 들고 있으면 메모리가 는다. 최근 것만 남긴다.
            if len(acts) > 400:
                acts.clear()
            acts[ep] = a
        return a

    fh = open(fp, "a")
    fh_lock = threading.Lock()

    def run(job):
        nm, ep, f = job
        a = actions(ep)
        if a is None or f >= len(a) - 4:
            return None
        try:
            im = np.array(Image.open(f"{TILES}/{nm}.png").convert("RGB"))
        except Exception:
            return None
        h, w, _ = im.shape
        views = [png(im[:, k * w // 3:(k + 1) * w // 3]) for k in range(3)]
        x = D.descriptors(a, f)
        txt, cost = ask_one(views, D.facts_text(x), instr.get(ep, ""))
        g = parse(txt)
        rec = {"ep": int(ep), "f": int(f), **(g or {}),
               "speed_mean": round(float(x.get("speed_mean", 0.0)), 4)}
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
                rate = STAT["n"] / el
                left = (len(jobs) - STAT["n"]) / max(rate, 1e-9) / 3600
                print(f"  {STAT['n']:,}/{len(jobs):,} · ${STAT['cost']:.2f} · "
                      f"{rate:.1f} req/s · 실패 {STAT['bad']} · 남은 {left:.1f}시간",
                      flush=True)
        return None

    with ThreadPoolExecutor(CONC) as ex:
        list(ex.map(run, jobs))
    fh.close()
    el = time.time() - STAT["t0"]
    print(f"완료 {STAT['n']:,} · 실패 {STAT['bad']} · ${STAT['cost']:.2f} · "
          f"{el / 3600:.2f}시간 -> {fp}", flush=True)


if __name__ == "__main__":
    main()
