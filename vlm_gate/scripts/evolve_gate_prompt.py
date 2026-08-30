"""Self-evolving guidance: one cycle of automatic prompt evolution.

Reads the compression-safety baselines (raw, K2), the latest gate eval (+ the
previous one for regression detection), the current guidance, and the evolution
history; assembles them with a fixed meta-prompt (evolving_guide_prompt.txt);
asks Claude (headless `claude -p`, subscription) to propose the next guidance;
writes it to the live guidance file + a versioned copy + appends to the log.

This automates the human-in-the-loop v0->v9 tuning into a closed loop:
  eval -> aggregate -> evolve (this script) -> eval -> ...

Usage:
  python scripts/evolve_gate_prompt.py \
    --gate output/robocasa/vlm_gate_gemma4_v9 \
    --prev-gate output/robocasa/vlm_gate_gemma4_tau0p5_mv_guide \
    [--dry-run]
"""
import argparse
import csv
import datetime as _dt   # only for stamping the log line (not used for logic)
import json
import os
import subprocess

ROBO = "output/robocasa"
GUIDE_META = "scripts/evolving_guide_prompt.txt"
GUIDE_FILE = "run_scripts/eval/vlm_gate_guidance.txt"
LOG = "analysis/_evolver/evolution_log.jsonl"
VER_DIR = "analysis/_evolver/guidance_versions"
BEST_STATE = "analysis/_evolver/best_state.json"   # v2: running-best for accept/reject gating


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
    """Mean action_steps over SUCCESSFUL episodes only (the protocol metric: failed
    episodes saturate at max_step and corrupt the signal)."""
    if not os.path.isfile(path):
        return None
    vals = []
    for ln in open(path):
        if ln.startswith("episode ") and "is_success:" in ln:
            tail = ln.split("is_success:", 1)[1]
            if "True" in tail.split("action_steps")[0] and "action_steps:" in tail:
                try:
                    vals.append(int(tail.split("action_steps:", 1)[1].split()[0]))
                except (ValueError, IndexError):
                    pass
    return sum(vals) / len(vals) if vals else None


def quant(d):
    f = f"{d}/gate_conf.csv"
    if not os.path.isfile(f) or os.path.getsize(f) < 50:
        return None
    q = tot = 0
    for r in csv.DictReader(open(f)):
        try:
            q += int(r["quantize"]); tot += 1
        except (KeyError, ValueError):
            pass
    return q / tot if tot else None


def build_table(raw, k2, gate, prev, root=ROBO):
    tasks = sorted(t for t in os.listdir(gate) if os.path.isdir(f"{gate}/{t}"))
    rows = []
    for t in tasks:
        r = succ(f"{root}/{raw}/{t}/prediction.txt")
        k = succ(f"{root}/{k2}/{t}/prediction.txt")
        g = succ(f"{gate}/{t}/prediction.txt")
        if None in (r, k, g):
            continue
        cat = "HARMFUL" if (k - r) <= -0.12 else ("SAFE" if (k - r) >= -0.03 else "moderate")
        qv = quant(f"{gate}/{t}")
        gst = succ_steps(f"{gate}/{t}/prediction.txt")          # gate succ-only steps
        rst = succ_steps(f"{root}/{raw}/{t}/prediction.txt")    # raw succ-only steps
        pv = succ(f"{prev}/{t}/prediction.txt") if prev else None
        # verdict
        if cat == "SAFE" and qv is not None and qv < 0.40:
            verdict = "UNDER-QUANT (missed safe speedup)"
        elif cat == "HARMFUL" and qv is not None and qv >= 0.50 and g <= k + 0.05:
            verdict = "OVER-QUANT (no protection)"
        elif pv is not None and g <= pv - 0.10:
            verdict = f"REGRESSION vs prev ({pv:.2f}->{g:.2f})"
        else:
            verdict = "ok"
        rows.append((t, cat, r, k, g, qv, gst, rst, pv, verdict))
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows):
    """Macro success/steps/quant overall and per SAFE/HARMFUL cluster — so the
    evolver sees the DUAL objective and any conservative drift, not just success."""
    def cl(c):
        return [x for x in rows if x[1] == c]
    fmtp = lambda v: f"{100 * v:.0f}%" if v is not None else "-"
    fmts = lambda v: f"{v:.0f}" if v is not None else "-"
    a = []
    a.append("===== AGGREGATE (read THIS first; decide direction from here) =====")
    a.append(f"macro success: raw={_mean([x[2] for x in rows]):.3f}  "
             f"K2={_mean([x[3] for x in rows]):.3f}  gate={_mean([x[4] for x in rows]):.3f}")
    a.append(f"macro succ-only steps: raw={fmts(_mean([x[7] for x in rows]))}  "
             f"gate={fmts(_mean([x[6] for x in rows]))}   (lower gate = more compression)")
    a.append(f"macro gate compression (quant%): {fmtp(_mean([x[5] for x in rows]))}")
    for c in ("SAFE", "HARMFUL", "moderate"):
        rs = cl(c)
        if rs:
            a.append(f"  {c:<8} cluster (n={len(rs)}): mean quant={fmtp(_mean([x[5] for x in rs]))}  "
                     f"gate_succ={_mean([x[4] for x in rs]):.3f}  raw_succ={_mean([x[2] for x in rs]):.3f}  "
                     f"gate_steps={fmts(_mean([x[6] for x in rs]))}")
    a.append("  -> SAFE wants HIGH quant (headroom); HARMFUL wants LOW quant (protect).")
    return "\n".join(a)


def macros(rows):
    return {
        "succ": round(_mean([x[4] for x in rows]), 3) if rows else 0,
        "steps": round(_mean([x[6] for x in rows]) or 0) if rows else 0,
        "quant": round(100 * (_mean([x[5] for x in rows]) or 0)) if rows else 0,
    }


def load_best(path=BEST_STATE):
    if os.path.isfile(path):
        try:
            return json.load(open(path))
        except Exception:
            return None
    return None


def save_best(rec, path=BEST_STATE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rec, open(path, "w"), indent=2)


def decide_accept(cand, best, *, baselines=None, w_succ=0.5, eps_score=0.02,
                  succ_floor_norm=0.5, comp_floor_norm=0.10,
                  floor_succ=None, eps_succ=0.01, eps_steps=5.0):
    """(1+1)-ES acceptance, v3: baseline-anchored composite score + corridor.

    Both objectives are normalized to THIS benchmark's measured achievable
    range (raw = success ceiling / zero compression; always-K2 = compression
    ceiling / success floor):
        succ_norm = (succ - k2_succ) / (raw_succ - k2_succ)      ~1 at raw
        comp_norm = (raw_steps - steps) / (raw_steps - k2_steps) ~1 at K2 speed
    (both capped at 1.15: beating a baseline beyond noise earns no extra credit,
    so a lucky success spike cannot buy unlimited compression loss.)

    Corridor (baseline-relative bounds, not reactive single conditions):
      succ_norm >= succ_floor_norm  (keep at least this share of the K2->raw
                                     success gap recovered)
      comp_norm >= comp_floor_norm  (abandoning compression entirely -> reject,
                                     however high the success)
    Acceptance inside the corridor is by the weighted COMPOSITE
        S = w_succ*succ_norm + (1-w_succ)*comp_norm
    improving the running best's S by > eps_score. No asymmetric branch: a
    success gain may buy back some compression and vice versa, as long as the
    weighted whole moves forward and neither objective leaves its corridor.

    Falls back to the old asymmetric Pareto rule when baselines are missing.
    Returns (accept, reason, info) with score components for the log.
    """
    cs, cst = cand["succ"], cand["steps"]
    if floor_succ is not None and cs < floor_succ - eps_succ:
        return False, f"below success floor ({cs:.3f} < {floor_succ:.3f}-{eps_succ:.3f})", {}
    if not baselines or None in (baselines.get("raw_succ"), baselines.get("k2_succ"),
                                 baselines.get("raw_steps"), baselines.get("k2_steps")):
        # legacy fallback (old rule) — only used when baselines can't be read
        if best is None:
            return True, f"seed running-best (succ={cs} steps={cst})", {}
        bs, bst = best["succ"], best["steps"]
        if cs <= bs - eps_succ and cst >= bst - eps_steps:
            return False, f"Pareto-dominated by best (succ {cs:.3f}<={bs:.3f}, steps {cst}>={bst})", {}
        if ((cs >= bs - eps_succ) and (cst <= bst - eps_steps)) or \
           ((cs >= bs + eps_succ) and (cst <= bst + eps_steps)):
            return True, f"Pareto-improving (succ {bs:.3f}->{cs:.3f}, steps {bst}->{cst})", {}
        return False, f"lateral succ<->steps trade, no dual gain (succ {bs:.3f}->{cs:.3f}, steps {bst}->{cst})", {}

    rs, ks = baselines["raw_succ"], baselines["k2_succ"]
    rst, kst = baselines["raw_steps"], baselines["k2_steps"]
    span_s = max(rs - ks, 1e-6)
    span_t = max(rst - kst, 1e-6)
    norm = lambda s, t: (min((s - ks) / span_s, 1.15), min((rst - t) / span_t, 1.15))

    def score(s, t):
        sn, cn = norm(s, t)
        return w_succ * sn + (1.0 - w_succ) * cn, sn, cn

    S, sn, cn = score(cs, cst)
    info = {"score": round(S, 4), "succ_norm": round(sn, 3), "comp_norm": round(cn, 3)}
    if sn < succ_floor_norm:
        return False, (f"outside success corridor (succ_norm {sn:.2f} < {succ_floor_norm}; "
                       f"succ {cs:.3f} vs raw {rs:.3f}/K2 {ks:.3f})"), info
    if cn < comp_floor_norm:
        return False, (f"outside compression corridor (comp_norm {cn:.2f} < {comp_floor_norm}; "
                       f"steps {cst:.0f} vs raw {rst:.0f}/K2 {kst:.0f} — compression abandoned)"), info
    if best is None:
        return True, f"seed running-best (S={S:.3f} succ_norm={sn:.2f} comp_norm={cn:.2f})", info
    Sb, snb, cnb = score(best["succ"], best["steps"])
    info["best_score"] = round(Sb, 4)
    if S > Sb + eps_score:
        return True, (f"composite improved S {Sb:.3f}->{S:.3f} "
                      f"(succ_norm {snb:.2f}->{sn:.2f}, comp_norm {cnb:.2f}->{cn:.2f})"), info
    return False, (f"composite not improved (S {S:.3f} vs best {Sb:.3f}+eps {eps_score}; "
                   f"succ_norm {sn:.2f}, comp_norm {cn:.2f})"), info


def fmt_table(rows):
    out = [f"{'task':<22}{'cat':<9}{'raw':>5}{'K2':>5}{'gate':>5}{'quant':>6}{'g_st':>6}{'r_st':>6}{'prev':>6}  verdict",
           "-" * 90]
    for t, cat, r, k, g, qv, gst, rst, pv, v in rows:
        out.append(f"{t:<22}{cat:<9}{r:>5.2f}{k:>5.2f}{g:>5.2f}"
                   f"{(f'{100*qv:.0f}%' if qv is not None else '-'):>6}"
                   f"{(f'{gst:.0f}' if gst is not None else '-'):>6}"
                   f"{(f'{rst:.0f}' if rst is not None else '-'):>6}"
                   f"{(f'{pv:.2f}' if pv is not None else '-'):>6}  {v}")
    m = lambda i: _mean([x[i] for x in rows]) or 0
    out.append("-" * 90)
    out.append(f"{'MEAN':<22}{'':<9}{m(2):>5.2f}{m(3):>5.2f}{m(4):>5.2f}")
    return "\n".join(out)


def group_metrics(rows):
    """Per-group succ/quant so the evolver can see WHERE an edit landed, not
    just the macro. Groups = task-name prefixes (suite for libero) when that
    yields a small set, else the SAFE/HARMFUL/moderate clusters."""
    def agg(sel):
        sel = list(sel)
        if not sel:
            return None
        return {"succ": round(_mean([x[4] for x in sel]), 3),
                "quant": round(100 * (_mean([x[5] for x in sel]) or 0))}
    prefixes = {}
    for x in rows:
        prefixes.setdefault(x[0].rsplit("_", 1)[0], []).append(x)
    if 2 <= len(prefixes) <= 8:
        return {k: agg(v) for k, v in sorted(prefixes.items())}
    return {c: agg([x for x in rows if x[1] == c])
            for c in ("SAFE", "HARMFUL", "moderate")
            if any(x[1] == c for x in rows)}


def history_text(path=None):
    path = path or LOG
    if not os.path.isfile(path):
        return "(no prior cycles)"
    lines = []
    for ln in open(path):
        try:
            e = json.loads(ln)
            acc = e.get("last_cand_accepted")
            tag = "ACCEPTED" if acc else ("rejected" if acc is not None else "?")
            lines.append(
                f"- {e.get('version','?')} [{tag}]: succ={e.get('macro_succ','?')} "
                f"steps={e.get('macro_steps','?')} quant={e.get('macro_quant','?')}% "
                f"| {e.get('change','')[:140]}")
            pg = e.get("per_group")
            if pg:
                lines.append("    per-group: " + "  ".join(
                    f"{k}={v['succ']:.2f}/{v['quant']}%" for k, v in pg.items() if v))
        except Exception:
            pass
    return ("\n".join(lines)
            + "\n(read the per-group lines causally: they show WHERE each edit "
              "actually landed — which group's quant moved and which group's "
              "success paid for it. Compose the next edit from the parts that "
              "worked; do not resubmit a rejected direction at the same strength.)"
            ) or "(no prior cycles)"


def call_claude(prompt, model, retries=4):
    """Run headless `claude -p` and parse the JSON object from its reply.

    Retries on transient failures (non-zero exit, timeout, empty/garbled
    output) with exponential backoff so a single API/network blip does not
    kill a multi-cycle self-evolve loop (observed: cycle-3 evolve died on an
    empty-stderr claude failure while claude was otherwise healthy).
    """
    import time as _t
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    last = ""
    for attempt in range(retries):
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
            if p.returncode != 0:
                last = f"rc={p.returncode} stderr={p.stderr[:300]!r} stdout={p.stdout[:200]!r}"
                raise RuntimeError(last)
            env = json.loads(p.stdout)
            text = env.get("result", "") if isinstance(env, dict) else str(env)
            s = text.strip()
            if "```" in s:
                s = s.split("```", 2)[1]
                s = s[4:] if s.lower().startswith("json") else s
            s = s[s.find("{"): s.rfind("}") + 1]
            return json.loads(s)
        except Exception as e:  # noqa
            last = str(e) or type(e).__name__
            if attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"[evolve] claude attempt {attempt+1}/{retries} failed ({last[:160]}); retrying in {wait}s", flush=True)
                _t.sleep(wait)
    raise RuntimeError(f"claude -p failed after {retries} attempts: {last[:400]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="baseline_full_v2_with_action_steps")
    p.add_argument("--k2", default="baseline_compress_K2")
    p.add_argument("--gate", required=True, help="latest gate result dir (under cwd)")
    p.add_argument("--prev-gate", default=None, help="previous gate dir (regression detection)")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--guidance-file", default=GUIDE_FILE, help="guidance file to read current + write next")
    p.add_argument("--dry-run", action="store_true", help="print proposed guidance, do not write")
    p.add_argument("--dump-prompt", default=None,
                   help="write the assembled prompt to this path and exit (no Claude call); for assembly inspection")
    # v2 accept/reject gating knobs
    p.add_argument("--no-gating", action="store_true",
                   help="disable accept/reject; always adopt the proposal (v1 behaviour)")
    p.add_argument("--floor-succ", type=float, default=None,
                   help="hard success floor: reject candidates below this (minus eps). Off by default.")
    p.add_argument("--eps-succ", type=float, default=0.01, help="success noise margin for gating")
    p.add_argument("--eps-steps", type=float, default=5.0, help="succ-only-steps noise margin for gating")
    # v3 composite gating (baseline-anchored). See decide_accept docstring.
    p.add_argument("--w-succ", type=float, default=0.5,
                   help="composite weight on succ_norm (1-w on comp_norm)")
    p.add_argument("--eps-score", type=float, default=0.02,
                   help="min composite-score gain over running best to accept")
    p.add_argument("--succ-floor-norm", type=float, default=0.5,
                   help="corridor: min share of the K2->raw success gap to keep")
    p.add_argument("--comp-floor-norm", type=float, default=0.10,
                   help="corridor: min share of the raw->K2 step reduction to keep")
    p.add_argument("--target-succ", type=float, default=0.645,
                   help="manual-best success to surpass (v9=0.645); shown to evolver as a baseline, not a ceiling")
    p.add_argument("--target-steps", type=float, default=249,
                   help="manual-best succ-only steps to beat (v9=249)")
    # Per-model (gemma|cosmos) isolation: separate version dir / log / gating state.
    p.add_argument("--ver-dir", default=VER_DIR, help="dir for versioned guidance files")
    p.add_argument("--log", dest="logpath", default=LOG, help="evolution log jsonl")
    p.add_argument("--best-state", default=BEST_STATE, help="running-best gating-state json")
    p.add_argument("--root-dir", default=ROBO, help="root dir holding raw/k2 baseline task subdirs (e.g. output/libero)")
    p.add_argument("--benchmark-context", default=None,
                   help="per-benchmark context file (tasks/cameras/action scheme + raw/K2 baselines) injected "
                        "into the evolver prompt. If unset, auto-detects <root-dir>/benchmark_context.txt.")
    args = p.parse_args()

    rows = build_table(args.raw, args.k2, args.gate, args.prev_gate, root=args.root_dir)
    mac = macros(rows)                       # metrics of the JUST-EVALUATED guidance
    macro = mac["succ"]
    cur_guidance = open(args.guidance_file).read()
    meta = open(GUIDE_META).read()

    # Per-benchmark context (tasks/cameras/action scheme + raw/K2 achievable range),
    # injected so the generic meta-prompt adapts to THIS benchmark. Explicit arg
    # wins; otherwise auto-detect <root-dir>/benchmark_context.txt.
    bctx_path = args.benchmark_context
    if not bctx_path:
        cand = os.path.join(args.root_dir, "benchmark_context.txt")
        bctx_path = cand if os.path.isfile(cand) else None
    benchmark_block = ""
    if bctx_path and os.path.isfile(bctx_path):
        benchmark_block = "\n\n===== BENCHMARK CONTEXT =====\n" + open(bctx_path).read().strip()

    # --- v2: accept/reject gating against running best (1+1-ES, Pareto) -------
    # `cur_guidance` is the candidate that produced THIS eval (`args.gate`).
    gating = not args.no_gating
    best = load_best(args.best_state) if gating else None
    gate_info = {}
    if gating:
        # Baselines measured on THIS benchmark anchor the composite normalization.
        baselines = {
            "raw_succ": _mean([x[2] for x in rows]),
            "k2_succ": _mean([x[3] for x in rows]),
            "raw_steps": _mean([x[7] for x in rows]),
            "k2_steps": _mean([succ_steps(f"{args.root_dir}/{args.k2}/{x[0]}/prediction.txt")
                               for x in rows]),
        }
        accepted, decision, gate_info = decide_accept(
            mac, best, baselines=baselines,
            w_succ=args.w_succ, eps_score=args.eps_score,
            succ_floor_norm=args.succ_floor_norm, comp_floor_norm=args.comp_floor_norm,
            floor_succ=args.floor_succ,
            eps_succ=args.eps_succ, eps_steps=args.eps_steps,
        )
        cand_rec = {"succ": mac["succ"], "steps": mac["steps"], "quant": mac["quant"],
                    "gate_dir": args.gate, "guidance": cur_guidance}
        if accepted:
            if not (args.dry_run or args.dump_prompt):
                save_best(cand_rec, args.best_state)   # new running best; mutate forward from here
            best = cand_rec
            base_guidance = cur_guidance
        else:
            base_guidance = best["guidance"] if best else cur_guidance   # revert; mutate from best
        print(f"\n=== GATING === accepted={accepted} :: {decision}")
        if best:
            print(f"    running-best: succ={best['succ']} steps={best['steps']} quant={best['quant']}%")
    else:
        accepted, decision, base_guidance = True, "gating disabled (v1 behaviour)", cur_guidance

    target_block = ""
    if args.target_succ is not None:
        target_block = (
            "\n\n===== TARGET TO SURPASS (manual best — a BASELINE to beat, NOT a ceiling) =====\n"
            f"success={args.target_succ:.3f}  succ-only steps={args.target_steps:.0f}\n"
            "Aim for the upper-left frontier: success >= this AND steps <= this. A "
            "success<->steps trade that lands Pareto-dominated by this point is NOT progress."
        )
    gating_block = ""
    if gating:
        gating_block = (
            "\n\n===== ACCEPT/REJECT (this loop now keeps a running best) =====\n"
            f"Last candidate was {'ACCEPTED' if accepted else 'REJECTED'}: {decision}.\n"
            + (f"Running best so far: success={best['succ']} steps={best['steps']} "
               f"quant={best['quant']}%.\n" if best else "")
            + ("The CURRENT GUIDANCE below has been REVERTED to the running best — the last "
               "edit regressed, so propose a DIFFERENT direction (do not repeat it).\n"
               if not accepted else
               "The CURRENT GUIDANCE below IS the new running best — build on it.\n")
        )

    prompt = (
        meta
        + benchmark_block
        + "\n\n===== CURRENT GUIDANCE =====\n" + base_guidance
        + target_block + gating_block
        + "\n\n" + aggregate(rows)
        + "\n\n===== LATEST EVAL (per-task) =====\n" + fmt_table(rows)
        + "\n\n===== EVOLUTION HISTORY (succ / steps / quant per cycle, with per-group breakdown) =====\n"
        + history_text(args.logpath)
        + "\n\nPropose the next guidance now as the specified JSON object."
    )

    if args.dump_prompt:
        open(args.dump_prompt, "w").write(prompt)
        print(f"[dump-prompt] accepted={accepted} :: {decision}")
        print(f"[dump-prompt] wrote {len(prompt)} chars to {args.dump_prompt}")
        return

    res = call_claude(prompt, args.model)
    print("=== AGGREGATE READ ===\n" + res.get("aggregate_read", ""))
    print("\n=== DIAGNOSIS ===\n" + res.get("diagnosis", ""))
    print("\n=== CHANGE ===\n" + res.get("change", ""))
    print("\n=== TARGETS ===", res.get("target_tasks"))
    print("\n=== PREDICTED EFFECT ===\n" + res.get("predicted_effect", ""))
    print("\n=== NEW GUIDANCE ===\n" + res.get("new_guidance", ""))

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return

    os.makedirs(os.path.dirname(args.logpath), exist_ok=True)
    os.makedirs(args.ver_dir, exist_ok=True)
    n_prev = len([f for f in os.listdir(args.ver_dir)]) if os.path.isdir(args.ver_dir) else 0
    ver = f"v{n_prev + 1}_auto"
    open(f"{args.ver_dir}/{ver}.txt", "w").write(res["new_guidance"].strip() + "\n")
    open(args.guidance_file, "w").write(res["new_guidance"].strip() + "\n")
    with open(args.logpath, "a") as f:
        f.write(json.dumps({
            "version": ver, "gate_dir": args.gate,
            "macro_succ": macro, "macro_steps": mac["steps"], "macro_quant": mac["quant"],
            "per_group": group_metrics(rows),
            "last_cand_accepted": bool(accepted), "gating_decision": decision,
            "gate_score": gate_info or None,
            "best_succ": (best or {}).get("succ"), "best_steps": (best or {}).get("steps"),
            "aggregate_read": res.get("aggregate_read"),
            "diagnosis": res.get("diagnosis"), "change": res.get("change"),
            "target_tasks": res.get("target_tasks"), "predicted_effect": res.get("predicted_effect"),
        }) + "\n")
    print(f"\n[written] {args.guidance_file}  +  {VER_DIR}/{ver}.txt  +  logged "
          f"({ver}, succ={macro} steps={mac['steps']} quant={mac['quant']}%, "
          f"last_cand={'ACCEPT' if accepted else 'REJECT'})")


if __name__ == "__main__":
    main()
