# Handoff: Self-Evolving Gate — findings & research direction

Relayed from the multigpu_workspace session (paper + ATQ side). Context for whoever
continues the self-evolving work in this workspace.

## TL;DR
Self-evolving the gate guidance **does** beat the naive seed, but it is **noisy /
non-convergent** and has **not yet beaten the hand-tuned `v9` prompt**. The gap to
manual-best is small on success but real on the Pareto frontier. The research goal is
to make the evolver **converge stably and reach/surpass v9** — by improving the
**evolver meta-policy**, not by more brute-force cycles.

## Numbers (base-GR00T, no internal router, RoboCasa-300, ~1200 ep)
| setting | Succ % | Succ-steps |
|---|---|---|
| raw (no gate) | 65.7 | 327 |
| always-K2 (always compress) | 59.8 | 214 |
| self-evolve naive (cycle 1) | 62.5 | 244 |
| self-evolve best (cycle 6) | 64.1 | 268 |
| manual `tau0p5` (no guidance) | 61.7 | 250 |
| manual `tau0p5_mv_guide` (v7-ish) | 63.6 | 247 |
| **manual `v9` (best)** | **64.5** | **249** |

## Reading of the data
1. **naive → evolve works**: 62.5 → 64.1 = **+1.6pt** over the naive seed. The core
   claim ("self-evolving beats naive") holds.
2. **v9 is NOT a hard success ceiling**: self-evolve best 64.1 vs v9 64.5 = **0.4pt**.
   Nearly matched on success.
3. **But v9 dominates the frontier**: v9 = 64.5%/**249 steps** vs best evolve cycle =
   64.1%/**268 steps** → v9 is both higher success AND ~8% fewer steps. The evolve
   cycles only **trade** success↔steps (cyc6 high-succ/high-steps; cyc7 low-succ/low-steps
   ~233) and never hit v9's high-succ+low-steps sweet spot.
4. **Manual climbed monotonically** (61.7→63.6→64.5); **auto oscillates** (61.8–64.1,
   non-monotonic, drifts). So the bottleneck is **convergence stability**, not capability.

## MoE / ATQ side (for context — different model, has internal router)
- The good MoE gate result (signed bias s=0.5 ≈ 66.1/263 vs router-only 65.8/292,
  −10% steps, success held) came from the **bias-bug FIX** (router-bias was being
  written to the wrong head class), **not** from prompt evolution. See
  `probe_router_bias.py` and commit `7dc3bfa`.
- MoE self-evolve cycles (~64.1–65.5) sit **below** the v9-seeded start (~66.1) → same
  story: evolve doesn't reliably beat the manual best.

## Research direction (next evolver-prompt experiments)
The meta-policy was already redesigned once (single-task fixation + success-only →
aggregate + dual-objective; the dry-run reversed the old ratchet). The evolver now
*reasons* correctly but cycles still don't beat manual-best. Next targets:
1. **Convergence stability**: add explicit accept/reject gating — reject any edit that
   regresses vs the running best; keep best-so-far (no drift / no ratchet backslide).
2. **Pareto-only acceptance**: don't accept success↔steps trades. Only accept edits
   that hold success while compressing more (or raise success at equal steps). Forbid
   Pareto-dominated moves.
3. **Surpass v9**: treat manual-best as a baseline to beat, not a ceiling — explore
   cues beyond v9's frontier point.

## Where the data / artifacts live
- Eval outputs: `output/robocasa/vlm_gate_auto_cycle*` and the symlinked baselines
  (`baseline_full_v2_with_action_steps` = raw, `baseline_compress_K2` = always-K2).
- Evolver logs / guidance versions: `analysis/_evolver/evolution_log*.jsonl`,
  `analysis/_evolver/guidance_versions{,_moe}/` (v9 = `guidance_versions/v9_manual.txt`).
- Meta-policy: `scripts/evolving_guide_prompt.txt`.
