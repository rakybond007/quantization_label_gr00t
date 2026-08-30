---
name: evolver-composite-gating
description: Accept-gate v3 — baseline-anchored composite score + corridor; fixed anti-compression drift the old asymmetric Pareto rule caused
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
---

The self-evolve accept gate (`decide_accept` in `vlm_gate/scripts/evolve_gate_prompt.py`)
was rewritten (2026-07-13) after the cosmos-TTL run drifted anti-compression: the old
rule required a ≥5-step gain to accept a compress-more candidate (rejected v2:
quant 12%, steps −4 — missed by 1 step) but tolerated +5-step regression on a
success-up candidate with NO quant term (accepted v5: quant 3% ≈ no compression).
The evolver's *intent* was correct every cycle — the accept CODE caused the drift.

**v3 rule (user-directed design: no single-condition hard cuts; baseline-relative
bounds + holistic scoring):** normalize both objectives to the benchmark's measured
raw↔K2 range — succ_norm=(succ−K2succ)/(raw−K2succ), comp_norm=(raw_steps−steps)/
(raw_steps−K2_steps), both capped at 1.15 — then (a) corridor: succ_norm≥0.5 and
comp_norm≥0.10 (abandoning compression rejects regardless of success), (b) accept iff
composite S=w·succ_norm+(1−w)·comp_norm beats running best by >eps_score (defaults
w=0.5, eps=0.02). Score components logged as `gate_score` in the evolution jsonl.
CLI: --w-succ --eps-score --succ-floor-norm --comp-floor-norm; legacy rule only as
fallback when baselines are unreadable.

Validated by replaying all three recorded LIBERO trajectories: cosmos-ttl → v2
accepted/v5 rejected (bug fixed); gemma-ttl → v1 stays best (identical outcome, so
that run needs no rerun); cosmos non-TTL → v3 (21%) still accepted.

Meta-prompt (`evolving_guide_prompt.txt`) additions: compression is FIRST-CLASS
(quant→0 = failure even if success rises), and rule 8 ESCALATE (if a predicted quant
move didn't materialize, don't retry same-strength rewording; two non-responses =
abandon that cue family). Old cosmos-ttl artifacts archived under
`output/libero/_archive_cosmos_ttl_gateV2/` + `analysis/_evolver/_archive_gateV2/`.
See [[terminology-naive-guidance]], [[libero-preemption-gate-bug]].
