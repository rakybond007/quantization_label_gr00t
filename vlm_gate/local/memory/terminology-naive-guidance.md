---
name: terminology-naive-guidance
description: "Terminology prefs — \"naive guidance\" not \"seed\"; \"quantization/quantize\" not \"compression/compress\""
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
---

In the self-evolving compression-gate work, refer to the initial hand-written
guidance prompt (the starting point of the self-evolve loop, used in Cycle 1) as
**"naive guidance"**, NOT "seed" / "naive seed". Applies to prose, captions, and
result tables (e.g. row label "Cycle 1 (naive guidance)").

**Why:** the user found "seed" ambiguous — it read as if it might mean the base/raw
policy. "naive guidance" is unambiguous: it is the naive natural-language prompt the
VLM gate starts from, distinct from `base/raw` (uncompressed GR00T policy) and
`always-K2` (unconditional compression).

**How to apply:** when describing that self-evolve did not improve over its starting
prompt, say "did not beat the naive guidance" (not "did not beat the seed"). See
[[libero-preemption-gate-bug]] context (LIBERO self-evolve results).

**Also:** call the action **"quantization" / "quantize"**, NOT "compression /
compress", throughout this work (prose, table rows, captions) — e.g. "No
quantization (raw)", "Always quantize (K=2)", "quantization gate", "quantizes 47%".
The scheme is K-block action quantization, so "quantization" is the correct term.
