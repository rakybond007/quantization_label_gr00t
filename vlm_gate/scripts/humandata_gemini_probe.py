"""libero 문항을 human_data 두 데이터셋에 그대로 걸어 Gemini 로 표본 판정.

예측은 `analysis/human_data/PREDICTION.md` 에 **돌리기 전에** 적어 두었다.
여기서는 그 예측과 대조할 숫자만 만든다.

프롬프트는 libero 정본을 그대로 쓴다(문항·GUIDANCE 손대지 않음). 다른 것은 둘뿐:
  - 계산 사실이 humandata_descriptors (액션이 절대 관절이라 병합 식이 다르다)
  - 뷰 문구가 2장짜리 (이 데이터셋도 카메라가 2대라 그대로 맞는다)
"""
import base64, io, json, os, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import humandata_descriptors as D  # noqa: E402

ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data"
EV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                  "analysis", "_evolver", "_libero")
GUID = open(f"{EV}/libero_guidance_v3c.txt").read().strip()
ASK = open(f"{EV}/libero_questions_v3c.txt").read().strip()

# HD_C_WIDTH=1 이면 C 한 문항만 바꾼다. **한 번에 하나만**(R2) -- A 까지 같이 바꾸면
# 무엇 때문에 달라졌는지 못 가린다. 지금 C 의 컷은 "받아주느냐" 인데 그 기준에서
# 넓은 접시와 좁은 컵이 같은 편에 떨어진다(72청크에서 pnp 2.43 vs long 2.37).
# 새 컷은 "입구가 물건에 비해 넉넉하냐" 다.
_C_OLD = ("C) Is an object being put into something that would catch it -- a basket, a bin,\n"
          "   a bowl -- so that landing off-centre changes nothing?")
_C_NEW = ("C) Is the place this object has to end up ROOMY FOR IT -- a wide bowl, an open\n"
          "   basket, a broad plate with slack all round -- rather than a cup, a jar or a\n"
          "   slot barely wider than the thing itself?")
if os.environ.get("HD_C_WIDTH") == "1":
    assert _C_OLD in ASK, "C 문항 원문을 못 찾았다 -- 파일이 바뀌었나"
    ASK = ASK.replace(_C_OLD, _C_NEW, 1)
VIEW = ("You are shown 2 camera views of this one moment: a scene view and a wrist "
        "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
        "normally look close in it -- general closeness is normal. Use the wrist view "
        "only to spot the actual grasp-closure or fine-insertion instant.")
Q = tuple("ABCDE")
MODEL = os.environ.get("HD_MODEL", "gemini/gemini-3.8-flash")
EFFORT = os.environ.get("HD_EFFORT", "low")
PER = int(os.environ.get("HD_PER_DS", "30"))
# HD_NO_MERGE=1 이면 압축 요구 두 문장을 뺀다. libero 사실에는 없던 문장이라
# 그것을 더한 것이 답을 흔드는지 따로 봐야 한다 (allex 에서 사실 두 문장이
# 네 문항 답의 36~51% 를 바꾼 전례가 있다).
NO_MERGE = os.environ.get("HD_NO_MERGE") == "1"
KEY = open(os.path.expanduser("~/.config/litellm/key")).read().strip()
URL = open(os.path.expanduser("~/.config/litellm/base")).read().strip() + "/v1/chat/completions"
_lock = threading.Lock()
STAT = {"n": 0, "bad": 0, "cost": 0.0}


def ask_one(imgs, fx, instr):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(i).decode()}}
               for i in imgs]
    content.append({"type": "text",
                    "text": f"{GUID}\n\n{VIEW}\n\nThe robot was told: {instr}\n\n{fx}\n\n{ASK}"})
    body = {"model": MODEL, "max_tokens": 1024, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    if EFFORT:
        body["reasoning_effort"] = EFFORT
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
                g[s[0]] = max(1, min(5, int(d[0])))
    return g if len(g) == len(Q) else None


def png(arr, px=448):
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO(); im.save(b, format="PNG")
    return b.getvalue()


def collect(ds):
    """국면이 고루 섞이도록 에피 여러 개에서 창의 여러 지점을 뽑는다."""
    import av
    R = f"{ROOT}/{ds}/lerobot"
    info = json.load(open(f"{R}/meta/info.json"))
    instr = {}
    for l in open(f"{R}/meta/episodes.jsonl"):
        d = json.loads(l)
        t = [x for x in d.get("tasks", []) if isinstance(x, str) and len(x.split()) > 1]
        if t:
            instr[d["episode_index"]] = t[0]
    eps = sorted(instr)[:PER // 3 + 2]
    jobs = []
    for ep in eps:
        ch = ep // info["chunks_size"]
        a = np.stack(pd.read_parquet(
            f"{R}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        n = len(a) - 20
        if n <= 0:
            continue
        # **국면은 비율이 아니라 그리퍼 신호로 찍는다.** 앞 판에서 0.15/0.45/0.80 을
        # 썼더니 pnp 의 0.80 은 아직 닫힌 운반 중이었고 long 은 표본의 67% 가 그리퍼가
        # 내내 열린 창이었다 -- 놓는 순간을 한 번도 안 뽑아서 A·C·D 를 판정할 수 없었다.
        gz = a[:, 7] > D._th(ds)[2]        # 문턱은 이 데이터셋 분포에서
        tr = np.flatnonzero(np.diff(gz.astype(int)))
        close = [t for t in tr if gz[t + 1]]          # 열림 -> 닫힘 = 파지
        openi = [t for t in tr if not gz[t + 1]]      # 닫힘 -> 열림 = 놓기
        picks = []
        if close:
            picks += [max(0, close[0] - 12), close[0]]                 # 접근 · 파지
        if close and openi:
            mid = [o for o in openi if o > close[0]]
            if mid:
                picks += [(close[0] + mid[0]) // 2, mid[0]]            # 운반 · 놓기
        if not picks:
            picks = [int(n * r) for r in (0.15, 0.45, 0.80)]
        picks = sorted({min(max(0, p), n - 1) for p in picks})
        frames = {}
        for key, slot in (("exterior_image_1_left", "scene"), ("wrist_image_left", "wrist")):
            p = f"{R}/videos/chunk-{ch:03d}/observation.images.{key}/episode_{ep:06d}.mp4"
            want = set(picks); c = av.open(p)
            for i, fr in enumerate(c.decode(video=0)):
                if i in want:
                    frames[(slot, i)] = png(fr.to_ndarray(format="rgb24"))
                    want.discard(i)
                    if not want:
                        break
            c.close()
        for f in picks:
            if ("scene", f) in frames and ("wrist", f) in frames:
                x = D.descriptors(a, f, dataset=ds)
                fx = D.facts_text(x)
                if NO_MERGE:
                    fx = "; ".join(fx.rstrip(".").split("; ")[:-2]) + "."
                jobs.append((ds, ep, f, [frames[("scene", f)], frames[("wrist", f)]],
                             fx, instr[ep], x))
        if len(jobs) >= PER:
            break
    return jobs[:PER]


def run(job):
    ds, ep, f, imgs, fx, instr, x = job
    txt, cost = ask_one(imgs, fx, instr)
    g = parse(txt)
    with _lock:
        STAT["n"] += 1; STAT["cost"] += cost
        if g is None:
            STAT["bad"] += 1
    return {"ds": ds, "ep": int(ep), "f": int(f), **(g or {}), "ans": txt,
            "merge_k2": float(x["merge_k2"]), "grip_closed_frac": float(x["grip_closed_frac"]),
            "facts": fx}


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = []
    for ds in ("pnp_task", "long_horizon_task"):
        j = collect(ds); print(f"[{ds}] 청크 {len(j)}", flush=True)
        jobs += j
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(run, jobs))
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"완료 {STAT['n']} · 파싱실패 {STAT['bad']} · 비용 ${STAT['cost']:.4f} -> {out}")
