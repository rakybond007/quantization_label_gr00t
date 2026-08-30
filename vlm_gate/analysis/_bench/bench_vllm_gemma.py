"""vLLM latency bench for the gemma judge (in-process LLM, no HTTP).
Same prompt semantics as vlm_gate.py; P(YES) from first-token top-logprobs
(the vlm_gate_cosmos.py.vllm.bak pattern). Times N single calls at 2 and 3 views.
"""
import base64, glob, io, os, sys, time, statistics as st

from PIL import Image
import numpy as np

sys.path.insert(0, os.path.expanduser("~/quantization_agent_workspace/vlm_gate/scripts"))
from vlm_gate import SYSTEM

GUIDANCE = ("Compression is the default: answer YES when the arm is moving through free "
            "space. Answer NO only when the gripper is in delicate contact.")
INSTR = "pick up the object and place it in the basket"

def data_url(im):
    b = io.BytesIO(); im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def load_imgs(n):
    base = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/libero")
    pngs = (sorted(glob.glob(os.path.join(base, "**", "*_img.png"), recursive=True))[:1]
            + sorted(glob.glob(os.path.join(base, "**", "*_wrist.png"), recursive=True))[:1])
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    if not imgs:
        imgs = [Image.fromarray(np.random.randint(0, 255, (256, 256, 3), np.uint8))]
    while len(imgs) < n:
        imgs.append(imgs[len(imgs) % len(imgs)].copy())
    return imgs[:n]

def messages(imgs):
    sys_text = SYSTEM + "\n\nAdditional learned guidance (from prior evaluations):\n" + GUIDANCE
    user = [{"type": "image_url", "image_url": {"url": data_url(im)}} for im in imgs]
    user.append({"type": "text", "text":
        f"Task: {INSTR}\nYou are shown the current camera view(s).\n"
        "Can the next ~1 second of motion be compressed (run at half rate)? "
        "Answer YES (compress) or NO (needs precise full-rate control)."})
    return [{"role": "system", "content": sys_text},
            {"role": "user", "content": user}]

def p_yes(logprob_dict, tok):
    ly = ln = None
    import math
    for tid, lp in logprob_dict.items():
        s = (lp.decoded_token or "").strip().lower()
        if s == "yes":
            ly = lp.logprob if ly is None else max(ly, lp.logprob)
        elif s == "no":
            ln = lp.logprob if ln is None else max(ln, lp.logprob)
    if ly is None and ln is None:
        return -1.0
    ly = ly if ly is not None else -100.0
    ln = ln if ln is not None else -100.0
    m = max(ly, ln)
    ey, en = math.exp(ly - m), math.exp(ln - m)
    return ey / (ey + en)

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    model_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12b-it"
    print(f"[vllm-bench] loading {model_id} ...", flush=True)
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=4096,
              limit_mm_per_prompt={"image": 3}, gpu_memory_utilization=0.85,
              enforce_eager=False)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    tok = None
    print("[vllm-bench] model loaded", flush=True)
    N, WARM = 30, 5
    for nv in (2, 3):
        msgs = messages(load_imgs(nv))
        for _ in range(WARM):
            llm.chat(msgs, sp, use_tqdm=False)
        ts, conf = [], -1.0
        for _ in range(N):
            t0 = time.perf_counter()
            out = llm.chat(msgs, sp, use_tqdm=False)
            ts.append((time.perf_counter() - t0) * 1000.0)
            lp0 = out[0].outputs[0].logprobs
            conf = p_yes(lp0[0], tok) if lp0 else -1.0
        ts.sort()
        print(f"[RESULT] vllm-gemma/{nv}view: n={N} mean={st.mean(ts):.1f}ms "
              f"std={st.pstdev(ts):.1f} p50={ts[len(ts)//2]:.1f} min={ts[0]:.1f} "
              f"max={ts[-1]:.1f} (last_conf={conf:.3f})", flush=True)
    print("[vllm-bench] DONE", flush=True)
