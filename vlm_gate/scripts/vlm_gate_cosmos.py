"""Cosmos 3 (nvidia/Cosmos3-Nano) action-quantization gate — Transformers path.

Drop-in alternative to the Gemma judge (vlm_gate.py). Same POST /judge contract
and VLMGate client, so eval clients / run_scripts only change --judge-url (and
which env runs the server).

Why Transformers (not vLLM): the official NVIDIA/cosmos "Reasoner with
Transformers" recipe loads ONLY the reasoner tower of the unified checkpoint via
`Cosmos3OmniForConditionalGeneration`, runs in-process, and lets us pin a CUDA
build matching the node driver (cu124). The vLLM 0.23 path forces a CUDA-13
torch that this cluster's 12.4 driver (550.x) cannot initialize. The previous
vLLM shim is kept at vlm_gate_cosmos.py.vllm.bak for CUDA-13 nodes.

Confidence: identical to the Gemma gate — a single forward, read the {YES,NO}
first-token logits, softmax -> calibrated P(YES) in [0,1] (NOT a 1-bit decode).

  python vlm_gate_cosmos.py --serve --model nvidia/Cosmos3-Nano --port 8120
  python vlm_gate_cosmos.py --ping <image> --url http://127.0.0.1:8120
"""
from __future__ import annotations

import argparse
import os
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

# Reuse the EXACT prompt + image coercion + client from the Gemma judge so the
# two backends are directly comparable (same SYSTEM text, same message layout).
from vlm_gate import to_pil, build_messages, parse_decision, VLMGate  # noqa: F401
import vlm_gate as _vg

# SYSTEM 극성 문제: 기본 SYSTEM은 YES=압축가능(안전)으로 정의하는데, 분해 문항들은
# YES=위험 신호를 뜻한다. 같은 컨텍스트에서 YES가 반대 의미를 가지면 판정이 뒤집힌다
# (allex에서 AUC 0.756 -> 0.40으로 역전되는 것을 확인). GATE_SYSTEM으로 전환한다.
#   default : 기존 로보카사 SYSTEM (하위호환)
#   neutral : 역할만 알리고 YES/NO 의미를 규정하지 않음
#   none    : system 메시지 자체를 제거
_MODE = os.environ.get("GATE_SYSTEM", "default")
NEUTRAL = ("You are inspecting a robot's camera views and its planned short-horizon motion. "
           "Answer each question that follows on its own merits, in the exact format requested. "
           "Do not add commentary.")
ALIGNED = ("You are inspecting a robot arm to judge whether the next ~1 second of its motion could run "
 "at HALF the control rate (consecutive steps averaged together) without changing the outcome. "
 "Compression saves time and is appropriate for gross motion; it fails only where fine, high-rate "
 "control is needed.\n"
 "COARSE, safe to compress — gross motion where small timing differences do not matter: reaching or "
 "approaching before contact, carrying or transporting an object, retracting or pulling back, broad "
 "arm sweeps, and pushing or pressing a large target such as a button.\n"
 "DELICATE, not safe to compress — only the brief moments of fine adjustment: actively closing the "
 "gripper onto a small or thin object to grasp it, or inserting / threading / aligning something into "
 "a tight target where millimetres matter.\n"
 "Mere proximity to objects, or steady contact with a large or forgiving object, does NOT require "
 "precision. Treat a moment as DELICATE only for true fine-manipulation; where a moment is genuinely "
 "borderline, let your answer carry that uncertainty.\n"
 "IMPORTANT: each question below asks about one specific observation. YES and NO refer only to the "
 "question being asked — they do not mean compress or do-not-compress. Answer each question on its "
 "own merits, in the exact format requested.")
if _MODE == "aligned":
    _vg.SYSTEM = ALIGNED
elif _MODE == "neutral":
    _vg.SYSTEM = NEUTRAL
elif _MODE == "none":
    _vg.SYSTEM = ""
SYSTEM = _vg.SYSTEM


def run_server(model_id, port, host, max_new_tokens, dtype):
    import torch
    from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    print(f"[judge] loading {model_id} (Cosmos3 reasoner, dtype={dtype}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        model_id, dtype=td, device_map="auto",
    ).eval()
    print("[judge] model loaded", flush=True)

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

    Y_IDS = _first_ids(["Y", " Y", "y", " y"])
    N_IDS = _first_ids(["N", " N", "n", " n"])
    print(f"[judge] Y ids={Y_IDS} N ids={N_IDS}", flush=True)

    def _pyes(lp, yes_ids, no_ids):
        ly = torch.logsumexp(lp[yes_ids], dim=0) if yes_ids else lp.new_tensor(-1e9)
        ln = torch.logsumexp(lp[no_ids], dim=0) if no_ids else lp.new_tensor(-1e9)
        return torch.softmax(torch.stack([ly, ln]), dim=0)[0].item()

    def judge_multi(pil_imgs, instruction, guidance="", question="", n_ask=4):
        """ONE image prefill -> per-question P(YES) for n_ask labelled slots.

        The 8-call protocol paid a full image prefill per question. Here the
        answer scaffold "A) <ans>\nB) <ans>..." is teacher-forced through the KV
        cache and P(YES|{YES,NO}) is read at each slot. Two details matter for
        the numbers to line up with the 8-call path:
          - each slot is anchored by its own "X) " label, so the model always
            knows which question it is answering (an unlabelled letter string
            collapses to a positional prior);
          - slots are scored over the SAME {YES,NO} word tokens the single
            question path uses, not single "Y"/"N" characters, which live in a
            differently-calibrated part of the distribution.
        """
        labels = [chr(ord("A") + i) for i in range(n_ask)]
        messages = build_messages(pil_imgs, instruction, guidance, question)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        confs, answers = [], []

        with torch.inference_mode():
            out = model(**inputs, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1].float()

            def feed(text):
                nonlocal past, logits
                ids = tok.encode(text, add_special_tokens=False)
                if not ids:
                    return
                step = model(input_ids=torch.tensor([ids], device=model.device),
                             past_key_values=past, use_cache=True)
                past, logits = step.past_key_values, step.logits[0, -1].float()

            for i, lab in enumerate(labels):
                feed(("" if i == 0 else "\n") + f"{lab}) ")
                lp = torch.log_softmax(logits, dim=-1)
                ly = torch.logsumexp(lp[YES_IDS], dim=0) if YES_IDS else lp.new_tensor(-1e9)
                ln = torch.logsumexp(lp[NO_IDS], dim=0) if NO_IDS else lp.new_tensor(-1e9)
                c = torch.softmax(torch.stack([ly, ln]), dim=0)[0].item()
                confs.append(float(c))
                ans = "YES" if c >= 0.5 else "NO"
                answers.append(ans)
                feed(ans)                       # force the answer, never a free argmax
        return {"confidences": confs, "answer": ",".join(answers),
                "labels": labels}

    def judge(pil_imgs, instruction, guidance="", question=""):
        """Single forward; return P(YES) over the {YES,NO} answer tokens."""
        messages = build_messages(pil_imgs, instruction, guidance, question)
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
            "quantize": bool(conf >= 0.5),
            "answer": "YES" if conf >= 0.5 else "NO",
            "p_yes_marginal": float(ly.exp().item()),  # full-vocab mass on YES (diag)
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
                n_ask = int(req.get("n_ask", 0))
                fn = judge_multi if n_ask > 0 else judge          # n_ask>0: one prefill, n answers
                res = fn(imgs, req.get("instruction", ""), req.get("guidance", ""),
                         req.get("question", ""), **({"n_ask": n_ask} if n_ask > 0 else {}))
                self._send(200, res)
            except Exception as e:  # noqa
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    srv = HTTPServer((host, port), Handler)
    print(f"[judge] JUDGE READY listening on {host}:{port}", flush=True)
    srv.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", action="store_true")
    p.add_argument("--model", type=str, default="nvidia/Cosmos3-Nano")
    p.add_argument("--port", type=int, default=8120)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", type=str, default="bfloat16")
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
