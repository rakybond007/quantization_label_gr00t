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
import re
import time
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
    from transformers import AutoProcessor

    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    # The class was pinned to Cosmos3Omni. Everything else in this file -- the
    # message scaffold, the forced answer slots, the graded and text paths -- is
    # generic across image-text models, so the class is chosen from the id and a
    # different judge can be served without touching any of it.
    if "cosmos" in model_id.lower():
        from transformers import Cosmos3OmniForConditionalGeneration as _Cls
        _tag = "Cosmos3 reasoner"
    else:
        from transformers import AutoModelForImageTextToText as _Cls
        _tag = "auto image-text"
    print(f"[judge] loading {model_id} ({_tag}, dtype={dtype}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = _Cls.from_pretrained(model_id, dtype=td, device_map="auto").eval()
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

    # Graded slots. The binary path reads two token groups at the answer slot;
    # nothing about it is specific to two. Reading N grade tokens instead gives
    # the model a scale to choose on AND keeps the full distribution over that
    # scale, which an API cannot do — top_logprobs truncates at 5, which is why
    # a 20-level API run collapsed while a 10-level one held.
    # The slot is fed as "A) ", so the next token is a bare digit — no leading
    # space variant. Including " 1" would take the tokenizer's space token as
    # the first id and put that SAME id in every grade group, which collapses
    # the distribution. Single digits only: "10" tokenizes as "1","0" and would
    # collide with grade 1.
    GRADE_IDS = [_first_ids([str(g)]) for g in range(1, 10)]
    _bad = [i for i, g in enumerate(GRADE_IDS) if not g]
    _dup = len({tuple(g) for g in GRADE_IDS}) != len(GRADE_IDS)
    print(f"[judge] grade ids 1..9={GRADE_IDS} distinct={not _dup} empty={_bad}", flush=True)

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

    def judge_multi_graded(pil_imgs, instruction, guidance="", question="",
                           n_ask=5, n_grade=5):
        """Same scaffold as judge_multi, but each slot is scored over 1..n_grade.

        Returns both readings of the same forward, so the aggregation can be
        chosen afterwards rather than baked in here:
          - `grades`  the level the model actually picks (argmax), which is what
            a text answer would have given;
          - `expected` the probability-weighted mean over the levels, which is
            what the binary path's continuous confidence is the two-level case of.
        Both are normalized to [0,1] so they drop into the existing aggregation.
        """
        labels = [chr(ord("A") + i) for i in range(n_ask)]
        ids = GRADE_IDS[:n_grade]
        messages = build_messages(pil_imgs, instruction, guidance, question)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        grades, expected, dists, answers = [], [], [], []

        with torch.inference_mode():
            out = model(**inputs, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1].float()

            def feed(text):
                nonlocal past, logits
                t = tok.encode(text, add_special_tokens=False)
                if not t:
                    return
                step = model(input_ids=torch.tensor([t], device=model.device),
                             past_key_values=past, use_cache=True)
                past, logits = step.past_key_values, step.logits[0, -1].float()

            for i, lab in enumerate(labels):
                feed(("" if i == 0 else "\n") + f"{lab}) ")
                lp = torch.log_softmax(logits, dim=-1)
                per = torch.stack([
                    torch.logsumexp(lp[g], dim=0) if g else lp.new_tensor(-1e9)
                    for g in ids])
                pr = torch.softmax(per, dim=0)
                k = int(pr.argmax().item())
                lev = torch.arange(n_grade, device=pr.device, dtype=pr.dtype)
                grades.append(k / (n_grade - 1))
                expected.append(float((pr * lev).sum().item()) / (n_grade - 1))
                dists.append([round(float(x), 5) for x in pr.tolist()])
                answers.append(str(k + 1))
                feed(str(k + 1))                # force the chosen level
        return {"grades": grades, "expected": expected, "dist": dists,
                "answer": ",".join(answers), "labels": labels}

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

    def _parse_text(text, n_ask=0, n_grade=0, n_new=0):
        """Pull the answers out of what the model wrote.

        Nothing is defaulted on a failure: a slot the model skipped stays None
        and the raw text comes back with it, because how often the model fails
        to answer in the requested form IS the measurement, and filling it in
        would destroy the number this path exists to produce.
        """
        if n_grade > 0:
            pat, conv = r"([A-Z])\)\s*([1-9])", int
            ok = lambda v: 1 <= v <= n_grade                       # noqa: E731
        else:
            pat = r"([A-Z])\)\s*(YES|NO|Yes|No|yes|no)"
            conv = lambda t: t.upper()                            # noqa: E731
            ok = lambda v: v in ("YES", "NO")                     # noqa: E731
        found = {}
        for lab, val in re.findall(pat, text):
            v = conv(val)
            if lab not in found and ok(v):
                found[lab] = v
        n = n_ask if n_ask > 0 else 1
        picks = [found.get(chr(ord("A") + i)) for i in range(n)]
        if n == 1 and picks[0] is None:
            m = (re.search(r"\b([1-9])\b", text) if n_grade > 0
                 else re.search(r"\b(YES|NO|Yes|No|yes|no)\b", text))
            if m:
                v = conv(m.group(1))
                if ok(v):
                    picks = [v]
        return {"text": text, "picks": picks, "pick": picks[0],
                "n_parsed": sum(p is not None for p in picks), "n_new": int(n_new)}

    def judge_text_batch(items, guidance="", question="", max_new_tokens=192,
                         n_ask=0, n_grade=0):
        """One forward for many frames.

        Per-frame calls leave a small model idle on a big card: the batch is one
        deep, and the cost is dominated by launching the forward rather than by
        the arithmetic in it. Labelling every frame of 7,200 episodes is about
        two million forwards, so the batch is where the hours are.

        `items` is [{"imgs": [...], "instruction": str}]; guidance and question
        are shared, since every frame is asked the same thing. Prompts differ in
        length (the measurements differ), so they are LEFT-padded -- generation
        reads from the end, and right padding would make the model continue from
        pad tokens.
        """
        _t_pre = time.time()
        seqs, n_in = [], []
        for it in items:
            msg = build_messages(it["imgs"], it.get("instruction", ""), guidance, question)
            enc = processor.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
            seqs.append(enc)
            n_in.append(enc["input_ids"].shape[1])

        width = max(n_in)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        # EVERY per-token field has to be padded alongside input_ids, not just the
        # attention mask. The processor also returns mm_token_type_ids, which says
        # for each position whether it is text or an image slot. Leaving it at its
        # own length made torch.cat fail, the old code then silently fell back to
        # the FIRST sample's copy, and the image slots no longer lined up with the
        # ids -- the model emitted EOS immediately and 17 of 32 answers came back
        # empty. Anything else the processor returns (pixel_values, image_grid_thw)
        # is per-image, not per-token, and simply concatenates.
        PER_TOKEN = ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids")
        batch, extra = {}, {}
        for enc in seqs:
            padw = width - enc["input_ids"].shape[1]
            for k, v in enc.items():
                if k in PER_TOKEN:
                    fill = pad_id if k == "input_ids" else 0
                    v = torch.cat([torch.full((1, padw), fill, dtype=v.dtype), v], dim=1)
                    extra.setdefault(k, []).append(v)
                else:
                    extra.setdefault(k, []).append(v)
            if "attention_mask" not in enc:
                am = torch.cat([torch.zeros((1, padw), dtype=torch.long),
                                torch.ones((1, enc["input_ids"].shape[1]), dtype=torch.long)], dim=1)
                extra.setdefault("attention_mask", []).append(am)
        for k, vs in extra.items():
            batch[k] = torch.cat(vs).to(model.device)

        _t_gen = time.time()
        with torch.inference_mode():
            gen = model.generate(**batch, max_new_tokens=int(max_new_tokens),
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True)
        out = gen.sequences
        # The training dataloader runs its transforms in workers, not in the
        # step. Here apply_chat_template -- which resizes and patchifies three
        # images per item -- runs serially in this loop before any of the GPU
        # work starts, so it is worth knowing what share it is before moving it.
        if os.environ.get("JUDGE_PROFILE"):
            _now = time.time()
            print(f"[prof] n={len(items)} pre={_t_gen - _t_pre:.2f}s "
                  f"gen={_now - _t_gen:.2f}s "
                  f"pre_share={(_t_gen - _t_pre) / max(_now - _t_pre, 1e-9):.0%}",
                  flush=True)
        # 등급 자리의 분포도 같이 돌려준다. 모델은 여전히 텍스트로 답하고 -- 이건
        # 강제된 슬롯의 로짓을 argmax 로 읽어 답을 대신 짓던 옛 방식이 아니다 --
        # 그 답이 얼마나 확실했는지를 덧붙일 뿐이다. 다섯 등급이 한 칸에 뭉쳐
        # 보여도 P(3)=0.9 와 P(2)=.3/P(3)=.35/P(4)=.3 은 전혀 다른 상태다.
        gid = [g[0] for g in GRADE_IDS[:n_grade]] if n_grade > 0 else []
        probs_all = None
        if gid and getattr(gen, "scores", None):
            import torch.nn.functional as _F
            sc = torch.stack(gen.scores, dim=1)          # (B, T, V)
            sel = sc[:, :, gid]                          # 등급 토큰만
            step_p = _F.softmax(sel.float(), dim=-1)     # 등급들 사이의 분포
            emitted = out[:, width:]
            is_grade = torch.zeros_like(emitted, dtype=torch.bool)
            for g in gid:
                is_grade |= (emitted == g)
            probs_all = []
            for i in range(emitted.shape[0]):
                pos = torch.nonzero(is_grade[i]).flatten().tolist()[:max(1, n_ask)]
                probs_all.append([[round(float(v), 4) for v in step_p[i, t]] for t in pos])
        res = []
        for i in range(len(items)):
            text = tok.decode(out[i][width:], skip_special_tokens=True).strip()
            r = _parse_text(text, n_ask, n_grade, int(out.shape[1] - width))
            if probs_all is not None:
                r["grade_probs"] = probs_all[i]
            res.append(r)
        return res

    def judge_text(pil_imgs, instruction, guidance="", question="",
                   max_new_tokens=192, n_ask=0, n_grade=0):
        """Let the model WRITE its answers, then parse what it wrote.

        Every other path here reads token logits at a forced answer slot. That
        scaffold came from the binary YES/NO case and was carried into the
        graded case unchanged, so the model has never actually been asked to
        answer -- an argmax has been answering for it. This path asks.

        Handles both answer scales, because both are in use: a graded question
        is parsed as "A) 3", a phase6-style question as "A) YES". With n_ask > 0
        the answers are returned positionally, so a slot the model skipped stays
        None instead of shifting the ones after it.

        Nothing is defaulted on a parse failure. How often the model declines to
        answer in the requested form IS the measurement, and filling it in would
        destroy exactly the number this path exists to produce.
        """
        messages = build_messages(pil_imgs, instruction, guidance, question)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=int(max_new_tokens),
                                 do_sample=False)
        text = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
        return _parse_text(text, n_ask, n_grade, int(out.shape[1] - n_in))

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
                n_ask = int(req.get("n_ask", 0))
                n_grade = int(req.get("n_grade", 0))
                _dec = lambda b: Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")

                # A batch request carries its images per item, so it has to be
                # taken before the single-request image fields are read -- those
                # are absent here, and reading them raised KeyError('image_b64')
                # before this branch was ever reached.
                if "batch" in req:                                # many frames, one forward
                    bat = [{"imgs": [_dec(b) for b in it["images_b64"]],
                            "instruction": it.get("instruction", "")} for it in req["batch"]]
                    self._send(200, {"results": judge_text_batch(
                        bat, req.get("guidance", ""), req.get("question", ""),
                        int(req.get("max_new_tokens", 192)), n_ask, n_grade)})
                    return

                imgs = ([_dec(b) for b in req["images_b64"]] if "images_b64" in req
                        else [_dec(req["image_b64"])])
                args = (imgs, req.get("instruction", ""), req.get("guidance", ""),
                        req.get("question", ""))
                if req.get("mode") == "text":                     # model writes it
                    res = judge_text(*args, n_ask=n_ask, n_grade=n_grade,
                                     max_new_tokens=int(req.get("max_new_tokens", 192)))
                elif n_grade > 0:                                   # graded slots
                    res = judge_multi_graded(*args, n_ask=n_ask or 5, n_grade=n_grade)
                elif n_ask > 0:                                   # one prefill, n answers
                    res = judge_multi(*args, n_ask=n_ask)
                else:
                    res = judge(*args)
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
