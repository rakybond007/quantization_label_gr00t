"""Self-evolving guidance for the MoE router-bias gate (one cycle).

Unlike the base-GR00T gate (scripts/evolve_gate_prompt.py), the MoE gate biases a
*learned* router, so the natural baseline is the SAME checkpoint with the gate
OFF (router-only), not raw-vs-K2. We label each task by how the gate's extra
compression affected success relative to router-only, feed the per-task table +
aggregate to the frozen evolver against the fixed meta-policy, and write the next
guidance. Quantization rate is read from gate_router.csv (fraction of chunks with
a compressed horizon H in {4,8}).

Usage:
  python scripts/evolve_gate_prompt_moe.py \
    --router output/robocasa/moe_router_confp07_ctrl \
    --gate   output/robocasa/moe_gate_auto_cycle1 \
    [--prev-gate output/robocasa/moe_gate_auto_cycle0] \
    --guidance-file run_scripts/eval/vlm_gate_guidance_moe_auto.txt [--dry-run]
"""
import argparse
import csv
import json
import os
import subprocess

ROBO = "output/robocasa"
GUIDE_META = "scripts/evolving_guide_prompt.txt"
LOG = "analysis/_evolver/evolution_log_moe.jsonl"
VER_DIR = "analysis/_evolver/guidance_versions_moe"


def succ(path):
    if not os.path.isfile(path):
        return None
    n = s = 0
    for ln in open(path):
        if ln.startswith("episode ") and "is_success:" in ln:
            n += 1
            if "True" in ln.split("is_success:", 1)[1].split("action_steps")[0]:
                s += 1
    return s / n if n else None


def succ_steps(path):
    if not os.path.isfile(path):
        return None
    v = []
    for ln in open(path):
        if ln.startswith("episode ") and "is_success:" in ln:
            tail = ln.split("is_success:", 1)[1]
            if "True" in tail.split("action_steps")[0] and "action_steps:" in tail:
                try:
                    v.append(int(tail.split("action_steps:", 1)[1].split()[0]))
                except (ValueError, IndexError):
                    pass
    return sum(v) / len(v) if v else None


def quant_moe(d):
    """Compressed-pick rate = fraction of chunks with horizon H in {4,8}."""
    f = f"{d}/gate_router.csv"
    if not os.path.isfile(f) or os.path.getsize(f) < 30:
        return None
    comp = tot = 0
    for r in csv.DictReader(open(f)):
        try:
            comp += int(r["H"]) in (4, 8); tot += 1
        except (KeyError, ValueError):
            pass
    return comp / tot if tot else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def build_table(router, gate, prev):
    tasks = sorted(t for t in os.listdir(gate) if os.path.isdir(f"{gate}/{t}"))
    rows = []
    for t in tasks:
        r = succ(f"{router}/{t}/prediction.txt")
        g = succ(f"{gate}/{t}/prediction.txt")
        if None in (r, g):
            continue
        # Label safety from the gate's own effect vs router-only:
        #   HARMFUL  = the extra compression cost success here (protect it)
        #   SAFE     = compression tolerated (headroom to push)
        cat = "HARMFUL" if (g - r) <= -0.08 else ("SAFE" if (g - r) >= -0.02 else "moderate")
        qv = quant_moe(f"{gate}/{t}")
        gst = succ_steps(f"{gate}/{t}/prediction.txt")
        rst = succ_steps(f"{router}/{t}/prediction.txt")
        pv = succ(f"{prev}/{t}/prediction.txt") if prev else None
        if cat == "HARMFUL" and qv is not None and qv >= 0.60:
            verdict = "OVER-QUANT (gate over-compressed, lost success)"
        elif cat == "SAFE" and qv is not None and qv < 0.50:
            verdict = "UNDER-QUANT (headroom unused)"
        elif pv is not None and g <= pv - 0.10:
            verdict = f"REGRESSION vs prev ({pv:.2f}->{g:.2f})"
        else:
            verdict = "ok"
        rows.append((t, cat, r, g, qv, gst, rst, pv, verdict))
    return rows


def aggregate(rows):
    fmtp = lambda v: f"{100*v:.0f}%" if v is not None else "-"
    fmts = lambda v: f"{v:.0f}" if v is not None else "-"
    a = ["===== AGGREGATE (read THIS first; decide direction from here) ====="]
    a.append(f"macro success: router-only(no gate)={_mean([x[2] for x in rows]):.3f}  "
             f"gate={_mean([x[3] for x in rows]):.3f}")
    a.append(f"macro succ-only steps: router-only={fmts(_mean([x[6] for x in rows]))}  "
             f"gate={fmts(_mean([x[5] for x in rows]))}   (lower gate = more compression)")
    a.append(f"macro gate compression (chunks with H in 4/8): {fmtp(_mean([x[4] for x in rows]))}")
    for c in ("SAFE", "HARMFUL", "moderate"):
        rs = [x for x in rows if x[1] == c]
        if rs:
            a.append(f"  {c:<8} cluster (n={len(rs)}): mean quant={fmtp(_mean([x[4] for x in rs]))}  "
                     f"gate_succ={_mean([x[3] for x in rs]):.3f}  router_succ={_mean([x[2] for x in rs]):.3f}")
    a.append("  -> SAFE = compression tolerated here (push). HARMFUL = compression hurt success (protect).")
    a.append("  -> Goal: keep macro gate success >= router-only while keeping gate steps low.")
    return "\n".join(a)


def fmt_table(rows):
    out = [f"{'task':<22}{'cat':<9}{'router':>7}{'gate':>6}{'quant':>6}{'g_st':>6}{'r_st':>6}{'prev':>6}  verdict",
           "-" * 92]
    for t, cat, r, g, qv, gst, rst, pv, v in rows:
        out.append(f"{t:<22}{cat:<9}{r:>7.2f}{g:>6.2f}"
                   f"{(f'{100*qv:.0f}%' if qv is not None else '-'):>6}"
                   f"{(f'{gst:.0f}' if gst is not None else '-'):>6}"
                   f"{(f'{rst:.0f}' if rst is not None else '-'):>6}"
                   f"{(f'{pv:.2f}' if pv is not None else '-'):>6}  {v}")
    return "\n".join(out)


def macros(rows):
    return {"succ": round(_mean([x[3] for x in rows]) or 0, 3),
            "steps": round(_mean([x[5] for x in rows]) or 0),
            "quant": round(100 * (_mean([x[4] for x in rows]) or 0))}


def history_text():
    if not os.path.isfile(LOG):
        return "(no prior cycles)"
    lines = []
    for ln in open(LOG):
        try:
            e = json.loads(ln)
            lines.append(f"- {e.get('version','?')}: succ={e.get('macro_succ','?')} "
                         f"steps={e.get('macro_steps','?')} quant={e.get('macro_quant','?')}% "
                         f"| {e.get('change','')[:140]}")
        except Exception:
            pass
    return "\n".join(lines) or "(no prior cycles)"


def call_claude(prompt, model):
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"claude -p failed: {p.stderr[:500]}")
    env = json.loads(p.stdout)
    text = env.get("result", "") if isinstance(env, dict) else str(env)
    s = text.strip()
    if "```" in s:
        s = s.split("```", 2)[1]
        s = s[4:] if s.lower().startswith("json") else s
    s = s[s.find("{"): s.rfind("}") + 1]
    return json.loads(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router", default="output/robocasa/moe_router_confp07_ctrl",
                   help="router-only (gate OFF) baseline dir")
    p.add_argument("--gate", required=True, help="latest MoE gate result dir")
    p.add_argument("--prev-gate", default=None)
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--guidance-file", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows = build_table(args.router, args.gate, args.prev_gate)
    if not rows:
        raise SystemExit("[evolve-moe] no comparable tasks found (check --router/--gate dirs)")
    mac = macros(rows)
    cur = open(args.guidance_file).read()
    meta = open(GUIDE_META).read()
    prompt = (meta
              + "\n\n===== CURRENT GUIDANCE =====\n" + cur
              + "\n\n" + aggregate(rows)
              + "\n\n===== LATEST EVAL (per-task; router-only = gate OFF baseline) =====\n" + fmt_table(rows)
              + "\n\n===== EVOLUTION HISTORY (succ / steps / quant per cycle) =====\n" + history_text()
              + "\n\nPropose the next guidance now as the specified JSON object.")

    res = call_claude(prompt, args.model)
    for k in ("aggregate_read", "diagnosis", "change", "target_tasks", "predicted_effect"):
        print(f"=== {k.upper()} ===\n{res.get(k)}\n")
    print("=== NEW GUIDANCE ===\n" + res.get("new_guidance", ""))
    if args.dry_run:
        print("\n[dry-run] not writing."); return

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(VER_DIR, exist_ok=True)
    ver = f"v{len(os.listdir(VER_DIR)) + 1}_moe"
    open(f"{VER_DIR}/{ver}.txt", "w").write(res["new_guidance"].strip() + "\n")
    open(args.guidance_file, "w").write(res["new_guidance"].strip() + "\n")
    with open(LOG, "a") as f:
        f.write(json.dumps({"version": ver, "gate_dir": args.gate,
                            "macro_succ": mac["succ"], "macro_steps": mac["steps"], "macro_quant": mac["quant"],
                            "aggregate_read": res.get("aggregate_read"), "diagnosis": res.get("diagnosis"),
                            "change": res.get("change"), "target_tasks": res.get("target_tasks"),
                            "predicted_effect": res.get("predicted_effect")}) + "\n")
    print(f"\n[written] {args.guidance_file} + {VER_DIR}/{ver}.txt + logged "
          f"({ver}, succ={mac['succ']} steps={mac['steps']} quant={mac['quant']}%)")


if __name__ == "__main__":
    main()
