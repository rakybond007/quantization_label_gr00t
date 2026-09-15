"""Zero-shot VLM action-quantization gate (model-agnostic).

A small HTTP service that, given the current camera frame + task instruction,
decides whether the upcoming action chunk is in a COARSE phase (free-space
transit / approach / retract where temporally downsampling the actions is safe)
or a PRECISE phase (grasp / align / insert / press, contact-rich, every control
step matters).  Returns a binary "quantize OK" decision.

Two surfaces in one file:
  * `python vlm_gate.py --serve --model google/gemma-4-12b-it --port 8120`
      Loads a VLM via transformers (AutoProcessor + AutoModelForImageTextToText)
      and serves POST /judge.  Model-agnostic: swap --model for any HF
      image-text-to-text model (Gemma 4, Qwen3.5-VL, ...).
  * `from vlm_gate import VLMGate`  -> client used by the eval loop.

This is an inference-time gate only; no training. Single-stream (one request at a
time from the control loop), so plain transformers.generate is sufficient.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Shared: image normalization + prompt
# -----------------------------------------------------------------------------

def to_pil(img) -> Image.Image:
    """Coerce an arbitrary obs image (np array / PIL) to an RGB PIL.Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    while arr.ndim > 3:          # drop leading time/batch dims, e.g. (1,H,W,3)
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))   # CHW -> HWC
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3], mode="RGB")


SYSTEM = (
    "You are a gate deciding whether the next ~1 second of a robot arm's motion can "
    "run at HALF the control rate (its consecutive steps averaged together) WITHOUT "
    "changing the outcome. Compressing saves time, so it is generally preferred for "
    "gross motion; judge honestly whether fine, high-rate control is needed now.\n"
    "Decision test: would averaging consecutive control steps noticeably change what "
    "happens? If no -> YES (coarse). If yes -> NO (precise).\n"
    "YES (coarse, compressible) — gross motion where small timing differences do not "
    "matter: reaching or approaching before contact, carrying or transporting an "
    "object, retracting or pulling back, broad arm sweeps, and pushing or pressing a "
    "large target such as a button.\n"
    "NO (precise, not compressible) — only the brief moments of delicate fine "
    "adjustment: actively closing the gripper onto a small or thin object to grasp "
    "it, or inserting / threading / aligning something into a tight target where "
    "millimeters matter.\n"
    "Mere proximity to objects, or steady contact with a large or forgiving object, "
    "does NOT require precision. Reserve NO for true fine-manipulation; for genuinely "
    "borderline moments answer with real uncertainty rather than forcing a side.\n"
    "Answer with exactly one word: YES or NO."
)

# Ablation arm (Codex R2): a NEUTRAL system prompt with the phase heuristics
# (reach/carry=YES, grasp/insert=NO rules) stripped out — only the task
# definition and the answer format remain. Activated via env so both judge
# backends (gemma here, cosmos importing from this module) inherit it.
NEUTRAL_SYSTEM = (
    "You are a gate deciding whether the next ~1 second of a robot arm's motion can "
    "run at HALF the control rate (its consecutive steps averaged together) WITHOUT "
    "changing the outcome. Decide from the camera views and the task. "
    "Answer with exactly one word: YES (can run at half rate) or NO (needs full rate)."
)
# 등급 문항 경로용 SYSTEM. **기존 라벨을 보존하기 위한 최소 수정이다.**
#
# 원래 SYSTEM 은 이진 게이트용이라 두 가지가 등급 경로와 맞지 않았다.
#   1) "Answer with exactly one word: YES or NO." -- 문항은 다섯 줄 등급을 요구한다
#   2) "reaching/carrying = YES, closing the gripper = NO" -- 문항이 재려던 것의
#      정답표를 미리 준다
#
# 여기서는 **틀은 그대로 남기고 그 둘만 걷어낸다.** 무엇을 판단하는 일인지(압축이
# 무엇을 빼앗는지)는 유지하므로 판정이 크게 흔들리지 않을 것으로 기대하고, 실제로
# 얼마나 흔들리는지는 표본으로 재서 기록한다. 벤치마크 이름(kitchen/table-top)과
# 배율("HALF the control rate")을 넣지 않아 셋이 같은 글을 쓸 수 있다.
GRADED_SYSTEM = (
    "You are judging one moment of a robot arm's motion, to decide how much of the "
    "next stretch of it could be thinned out -- how many of its commanded poses "
    "could be dropped, letting the arm travel further between the ones that "
    "remain, without changing the outcome.\n"
    "What thinning takes away is the arm's chance to correct itself on the way. "
    "Judge the moment in front of you, not the task as a whole: one task passes "
    "through several kinds of motion from one second to the next.\n"
    "Below you are given measurements computed from the planned motion, then a "
    "list of checks. Answer the checks in the form the checks themselves ask for, "
    "and nothing else. Do not decide the compression yourself -- each check is one "
    "piece of evidence, and they are combined afterwards."
)

# 최소 수정판. **틀린 줄 하나만 고친다.**
#
# 원래 SYSTEM 의 마지막 줄 "Answer with exactly one word: YES or NO." 는 등급 다섯 줄을
# 요구하는 문항과 정면으로 모순된다. 점수와 무관하게 틀린 지시다. 여기서는 그 한 줄만
# 문항에 형식을 넘기는 문장으로 바꾸고 **나머지는 글자 하나 건드리지 않는다** --
# 국면 규칙도 그대로 남긴다.
#
# 두 개를 같이 걷어낸 GRADED_SYSTEM 은 robocasa 사다리 상관을 -0.512 -> -0.195 로
# 무너뜨렸다. 그래서는 어느 변경이 원인인지 가릴 수 없다(하네스 R2: 한 번에 하나).
_WRONG_LINE = "Answer with exactly one word: YES or NO."
_RIGHT_LINE = ("Answer in the form the checks below ask for, and nothing else -- "
               "each check is one piece of evidence, and they are combined afterwards.")
assert _WRONG_LINE in SYSTEM, "SYSTEM 의 마지막 줄이 바뀌었다 -- 최소 수정판을 다시 확인하라"
MINFIX_SYSTEM = SYSTEM.replace(_WRONG_LINE, _RIGHT_LINE)

if os.environ.get("JUDGE_MINFIX_SYSTEM") == "1":
    SYSTEM = MINFIX_SYSTEM

if os.environ.get("JUDGE_GRADED_SYSTEM") == "1":
    SYSTEM = GRADED_SYSTEM

if os.environ.get("JUDGE_NEUTRAL_SYSTEM") == "1":
    SYSTEM = NEUTRAL_SYSTEM

# 등급 문항 경로에서는 SYSTEM 을 비울 수 있어야 한다.
#
# SYSTEM 은 이진 게이트(YES/NO)를 위해 쓴 것이고 마지막 줄이
# "Answer with exactly one word: YES or NO." 다. 그런데 등급 문항(A~E 를 1~5 로)
# 경로도 build_messages 를 거치므로 그 문장이 같이 나간다 -- 한 프롬프트 안에서
# 한 단어를 요구하고 다섯 줄을 요구한다. 형식은 마지막 지시를 따라 깨지지 않았지만,
# SYSTEM 이 "reaching/carrying = YES, closing the gripper = NO" 라는 **답 매핑을
# 미리 알려준다.** 문항이 재려던 것을 SYSTEM 이 먼저 말해 주므로 문항 설계 실험이
# 독립적이지 못하다.
#
# JUDGE_NO_SYSTEM=1 이면 SYSTEM 블록 자체가 빠진다 (build_messages 가 빈 문자열이면
# 블록을 넣지 않는다). 등급 경로의 기본값으로 삼아야 하는지는 측정으로 정한다.
if os.environ.get("JUDGE_NO_SYSTEM") == "1":
    SYSTEM = ""


def build_messages(pil_imgs, instruction, guidance="", question="", user_only=False):
    if not isinstance(pil_imgs, (list, tuple)):
        pil_imgs = [pil_imgs]
    sys_text = SYSTEM
    if guidance:
        sys_text = SYSTEM + "\n\nAdditional learned guidance (from prior evaluations):\n" + guidance.strip()
    if len(pil_imgs) == 6:
        # **두 시점을 함께 보여 주는 판.** 한 순간만 주면 국면(접근·파지·운반·정렬)을
        # 볼 근거가 이미지에 없어서, 문항을 국면 기준으로 써 두어도 모델이 태스크
        # 정체성으로 떨어진다 -- 자체집계 라벨은 태스크 다수결 하나로 86% 가 맞았고
        # 6/24 태스크는 전 장면이 같은 답이었다. 그래서 t 와 t+16 을 같이 준다.
        view_note = ("You are shown 6 images: the SAME 3 camera views at two moments. "
                     "Images 1-3 are NOW (agentview-left, agentview-right, wrist "
                     "close-up). Images 4-6 are the SAME three views ~1 second LATER, "
                     "at the end of the motion you are judging. Compare them to see "
                     "what this motion actually does and which stage of the task it is "
                     "-- approaching, closing the grasp, carrying, or lining up to "
                     "release. The wrist camera is mounted on the gripper, so objects "
                     "normally look close in it; use it to spot the grasp-closure or "
                     "fine-insertion instant.")
    elif len(pil_imgs) >= 3:
        view_note = ("You are shown 3 camera views: agentview-left, agentview-right, "
                     "and a wrist (eye-in-hand) close-up. The wrist camera is mounted on "
                     "the gripper, so objects normally look close in it — general "
                     "closeness is normal. Use the wrist view only to spot the actual "
                     "grasp-closure or fine-insertion instant.")
    elif len(pil_imgs) == 2:
        # **libero 와 dexjoco 는 카메라가 2대다.** 전에는 여기가 "the current camera
        # view(s)" 였고, 한편 libero 전문에는 3대(agentview-left/right/wrist)라고
        # 적혀 있었다 -- libero 에는 그 두 이름의 카메라가 없다(front_view,
        # left_wrist_view 뿐). 있는 것만 말한다.
        view_note = ("You are shown 2 camera views of this one moment: a scene view "
                     "and a wrist (eye-in-hand) close-up. The wrist camera is mounted "
                     "on the gripper, so objects normally look close in it -- general "
                     "closeness is normal. Use the wrist view only to spot the actual "
                     "grasp-closure or fine-insertion instant.")
    else:
        view_note = "You are shown the current camera view(s)."
    user_content = [{"type": "image", "image": im} for im in pil_imgs]
    tail_q = question.strip() if question else (
        "Can the next ~1 second of motion be compressed (run at half rate)? "
        "Answer YES (compress) or NO (needs precise full-rate control).")

    if user_only:
        # **allex(Gemini) 배치.** system 메시지를 쓰지 않고 GUIDANCE 가 user 본문의
        # 머리로 들어간다. 이진 게이트용 SYSTEM 을 등급 문항과 같이 보내면 한
        # 프롬프트가 서로 다른 답 형식을 요구하고, 그 SYSTEM 이 "reaching/carrying
        # = YES, closing the gripper = NO" 라는 답 매핑을 미리 준다. 자세한 규격은
        # `prompts/FORMAT.md`.
        #
        # 호출부는 지시문과 계산 사실을 `f"{instr}\n{facts}"` 로 붙여 넘긴다
        # (phase9_two_sided.py, label_chunks.py). allex 처럼 두 자리로 떼어 놓기
        # 위해 첫 줄까지를 지시문으로 본다 -- 시그니처를 네 층 고치는 대신.
        instr, _, facts = str(instruction).partition("\n")
        parts = [guidance.strip(), view_note,
                 f"The robot was told: {instr.strip()}", facts.strip(), tail_q]
        user_content.append({"type": "text",
                             "text": "\n\n".join(x for x in parts if x)})
        return [{"role": "user", "content": user_content}]

    user_content.append({"type": "text", "text": (
        f"Task: {instruction}\n{view_note}\n" + tail_q)})
    msgs = []
    if sys_text.strip():                      # 빈 SYSTEM이면 블록 자체를 넣지 않는다
        msgs.append({"role": "system", "content": [{"type": "text", "text": sys_text}]})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def parse_decision(text):
    """Return (quantize: bool, matched: str). Conservative default = NO (raw)."""
    t = (text or "").upper()
    m = re.search(r"\b(YES|NO)\b", t)
    if not m:
        return False, ""
    return (m.group(1) == "YES"), m.group(1)


# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------

def run_server(model_id, port, host, max_new_tokens, dtype):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    print(f"[judge] loading {model_id} (dtype={dtype}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=td, device_map="cuda", trust_remote_code=True
    ).eval()
    print("[judge] model loaded", flush=True)

    # Resolve candidate first-token ids for YES / NO so we can read a calibrated
    # confidence from the next-token logits instead of a hard 1-bit decode.
    tok = processor.tokenizer

    def _first_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    YES_IDS = _first_ids(["YES", "Yes", "yes", " YES", " Yes", " yes"])
    NO_IDS = _first_ids(["NO", "No", "no", " NO", " No", " no"])
    print(f"[judge] YES ids={YES_IDS} NO ids={NO_IDS}", flush=True)

    # Optional torch.compile (JUDGE_COMPILE=1). Measured on A100-80G bf16:
    # 2-view 175.9 -> 154.9 ms (-12%), 3-view unchanged. Any failure falls back
    # to eager. (The cosmos judge stays eager: its vision-embed split breaks
    # under dynamo, measured.) A new instruction-length shape triggers a brief
    # one-time re-capture during serving.
    if os.environ.get("JUDGE_COMPILE", "0") == "1":
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            _cm = torch.compile(model, mode="reduce-overhead")
            _in = processor.apply_chat_template(
                build_messages([Image.new("RGB", (256, 256))] * 2, "warmup", ""),
                add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                for _ in range(3):
                    _cm(**_in).logits
            model = _cm
            print("[judge] torch.compile enabled (warmed up)", flush=True)
        except Exception as e:
            print(f"[judge] torch.compile failed -> eager fallback: {e}", flush=True)

    def judge(pil_imgs, instruction, guidance=""):
        """Single forward; return P(YES) over the {YES,NO} answer tokens."""
        messages = build_messages(pil_imgs, instruction, guidance)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()   # next-token dist
        lp = torch.log_softmax(logits, dim=-1)
        ly = torch.logsumexp(lp[YES_IDS], dim=0) if YES_IDS else lp.new_tensor(-1e9)
        ln = torch.logsumexp(lp[NO_IDS], dim=0) if NO_IDS else lp.new_tensor(-1e9)
        conf = torch.softmax(torch.stack([ly, ln]), dim=0)[0].item()  # P(YES|{YES,NO})
        return {
            "confidence": float(conf),                 # client thresholds this
            "quantize": bool(conf >= 0.5),             # default 0.5 (client may override)
            "answer": "YES" if conf >= 0.5 else "NO",
            "p_yes_marginal": float(ly.exp().item()),  # full-vocab mass on YES (diagnostic)
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": model_id})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/judge":
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n))
                if "images_b64" in req:
                    imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
                            for b in req["images_b64"]]
                else:
                    imgs = [Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")]
                res = judge(imgs, req.get("instruction", ""), req.get("guidance", ""))
                self._send(200, res)
            except Exception as e:  # noqa
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    # Single-threaded server: requests serialize naturally (mirrors the GR00T
    # ZMQ REP server), so concurrent eval clients are safe without a lock.
    srv = HTTPServer((host, port), Handler)
    print(f"[judge] JUDGE READY listening on {host}:{port}", flush=True)
    srv.serve_forever()


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------

class VLMGate:
    """Client for the judge service. .judge(img, instruction) -> dict.

    On any transport/parse error returns a fail-safe decision (quantize=False =
    execute raw) so a flaky judge never silently corrupts the rollout.
    """

    def __init__(self, url, timeout=30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout


    def judge_batch(self, items, guidance="", question="", n_ask=0, n_grade=0,
                    max_new_tokens=192, mode="text"):
        """Judge many frames in one forward. items: [(imgs, instruction), ...]

        `mode` is accepted so a caller can state the path it wants. The batch
        endpoint only has one: the server routes every batch to
        `judge_text_batch`, where the model writes the answer and the server
        parses it. Anything other than "text" is refused rather than quietly
        served as text -- a caller asking for forced slot logits must find out
        it cannot have them here (CLAUDE.md 되돌리지 말 것 1 forbids that path).
        """
        if mode not in ("", "text"):
            return [{"error": f"judge_batch is text-only, got mode={mode!r}"}] * len(items)
        try:
            bat = []
            for imgs, instruction in items:
                if not isinstance(imgs, (list, tuple)):
                    imgs = [imgs]
                b64s = []
                for im in imgs:
                    buf = io.BytesIO()
                    to_pil(im).save(buf, format="PNG")
                    b64s.append(base64.b64encode(buf.getvalue()).decode())
                bat.append({"images_b64": b64s, "instruction": instruction})
            payload = json.dumps({"batch": bat, "guidance": guidance,
                                  "question": question, "n_ask": n_ask,
                                  "n_grade": n_grade, "mode": "text",
                                  "max_new_tokens": max_new_tokens}).encode()
            req = urllib.request.Request(
                self.url + "/judge", data=payload,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
            return out.get("results") or [{"error": out.get("error", "no results")}] * len(items)
        except Exception as e:  # noqa
            return [{"error": f"{type(e).__name__}: {e}"}] * len(items)

    def judge(self, imgs, instruction, guidance="", question="", n_ask=0, n_grade=0,
              mode=""):
        try:
            if not isinstance(imgs, (list, tuple)):
                imgs = [imgs]
            b64s = []
            for im in imgs:
                buf = io.BytesIO()
                to_pil(im).save(buf, format="PNG")
                b64s.append(base64.b64encode(buf.getvalue()).decode())
            payload = json.dumps({
                "images_b64": b64s,
                "instruction": instruction,
                "guidance": guidance,
                "question": question,
                "n_ask": n_ask,            # >0: answer n checks off ONE image prefill
                "n_grade": n_grade,        # >0: score each slot over 1..n_grade instead of YES/NO
                "mode": mode,              # "text": model writes the answer, server parses it
            }).encode()
            req = urllib.request.Request(
                self.url + "/judge", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa
            return {"quantize": False, "answer": "", "raw": "", "error": str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true")
    p.add_argument("--model", type=str, default="google/gemma-4-12b-it")
    p.add_argument("--port", type=int, default=8120)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", type=str, default="bfloat16")
    # quick CLI self-test against a running server: --ping <image_path>
    p.add_argument("--ping", type=str, default=None)
    p.add_argument("--url", type=str, default="http://127.0.0.1:8120")
    p.add_argument("--instruction", type=str, default="pick up the object")
    args = p.parse_args()

    if args.serve:
        run_server(args.model, args.port, args.host, args.max_new_tokens, args.dtype)
    elif args.ping:
        g = VLMGate(args.url)
        print(g.judge(Image.open(args.ping), args.instruction))
    else:
        p.error("use --serve or --ping <image>")


if __name__ == "__main__":
    main()
