"""Micro-benchmark: per-call judge() inference latency for the Gemma vs Cosmos3
VLM quantization gate. Loads the model exactly like vlm_gate*.run_server, then
times N single-forward judge() calls at 2 views (LIBERO) and 3 views (robocasa).

Run with the matching interpreter/env:
  gemma : ~/miniconda3/envs/vlm_judge/bin/python bench_judge_latency.py --backend gemma
  cosmos: ~/quantization_agent_workspace/cosmos_judge_venv/bin/python bench_judge_latency.py --backend cosmos
"""
import argparse, os, sys, time, statistics as st, glob
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/quantization_agent_workspace/vlm_gate/scripts"))
from vlm_gate import build_messages  # same message layout for both backends

GUIDANCE = ("Compression is the default: answer YES when the arm is moving through free "
            "space — reaching, transporting a grasped object, retracting, or broad sweeps. "
            "Answer NO only when the gripper is in delicate contact.")
INSTR = "pick up the object and place it in the basket"


def load_real_imgs(n):
    """Prefer real saved rollout frames (representative resolution); else random."""
    base = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/libero")
    pngs = sorted(glob.glob(os.path.join(base, "**", "*_img.png"), recursive=True))[:1] \
        + sorted(glob.glob(os.path.join(base, "**", "*_wrist.png"), recursive=True))[:1]
    imgs = [Image.open(p).convert("RGB") for p in pngs if os.path.isfile(p)]
    if not imgs:
        imgs = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))]
    while len(imgs) < n:
        imgs.append(imgs[len(imgs) % len(imgs)].copy())
    return imgs[:n]


def build_judge(backend, model_id, dtype="bfloat16", attn=None, compile_=False):
    import torch
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    extra = {}
    if attn:
        extra["attn_implementation"] = attn
    print(f"[bench] loading {backend} {model_id} attn={attn} compile={compile_} ...", flush=True)
    if backend == "cosmos":
        from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            model_id, dtype=td, device_map="auto", **extra).eval()
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=td, device_map="cuda", trust_remote_code=True, **extra).eval()
    if compile_:
        # reduce-overhead = CUDA graphs; static shapes in this bench, so the
        # steady-state numbers are the best case for a fixed-shape serving path.
        # suppress_errors: untraceable parts (e.g. cosmos vision split with
        # dynamic sizes) silently fall back to eager, the rest still compiles.
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="reduce-overhead")
    print("[bench] model loaded", flush=True)
    tok = processor.tokenizer

    def _first_ids(words):
        ids = set()
        for w in words:
            enc = tok.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)
    YES_IDS, NO_IDS = _first_ids(["YES", "Yes", "yes"]), _first_ids(["NO", "No", "no"])

    probed = set()

    def judge(pil_imgs):
        messages = build_messages(pil_imgs, INSTR, GUIDANCE)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        k = len(pil_imgs)
        if k not in probed:  # one-time probe: where do the tokens go?
            probed.add(k)
            shapes = {kk: tuple(v.shape) for kk, v in inputs.items() if hasattr(v, "shape")}
            print(f"[probe] views={k} seq_len={inputs['input_ids'].shape[-1]} shapes={shapes}", flush=True)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        ly = torch.logsumexp(lp[YES_IDS], dim=0)
        ln = torch.logsumexp(lp[NO_IDS], dim=0)
        return torch.softmax(torch.stack([ly, ln]), dim=0)[0].item()
    return judge, torch


def bench(judge, torch, imgs, n, warmup, tag):
    for _ in range(warmup):
        judge(imgs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        conf = judge(imgs)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    ts.sort()
    print(f"[RESULT] {tag}: n={n} views={len(imgs)} "
          f"mean={st.mean(ts):.1f}ms std={st.pstdev(ts):.1f} "
          f"p50={ts[len(ts)//2]:.1f} min={ts[0]:.1f} max={ts[-1]:.1f} "
          f"(last_conf={conf:.3f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["gemma", "cosmos"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--attn", default=None, help="attn_implementation: sdpa|flash_attention_2|eager")
    ap.add_argument("--compile", action="store_true", help="torch.compile(mode=reduce-overhead)")
    a = ap.parse_args()
    model_id = a.model or ("nvidia/Cosmos3-Nano" if a.backend == "cosmos" else "google/gemma-4-12b-it")
    judge, torch = build_judge(a.backend, model_id, attn=a.attn, compile_=a.compile)
    cfg = f"{a.backend}[attn={a.attn or 'default'}{',compile' if a.compile else ''}]"
    # extra warmup for compile (graph capture happens on first calls per shape)
    wu = a.warmup + (10 if a.compile else 0)
    for nv in (2, 3):
        imgs = load_real_imgs(nv)
        bench(judge, torch, imgs, a.n, wu, f"{cfg}/{nv}view")
    print("[bench] DONE", flush=True)
