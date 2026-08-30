# Compression-Aware Routing (compaware)

How to bias the MoE router toward shorter/more compressed expert decoders so
the model executes fewer environment steps per chunk. Two related mechanisms
are documented here; pick the one that matches your training fork.

---

## TL;DR

| Mechanism | Flag | Where applied | Quality-aware? | Verdict from sim |
|-----------|------|--------------|----------------|------------------|
| **compaware (legacy)** | `--moe-compression-weight λ_c` | total loss bonus on mean router prob | ❌ blind | over-pushes at λ_c=0.1; mild λ_c=0.02~0.03 only |
| **length-cost on KL target ([D])** | `--moe-length-cost-weight γ` | EMA-normalized KL target softmax (metaq fork) | ✓ sample-conditional | safer; γ=0.05~0.10 |

For a real-robot deployment, **prefer the length-cost on KL target** ([D])
combined with [B] EMA normalization. It bakes in a sample-conditional
quality-vs-length trade-off: easy moments compress, precise moments stay
raw, without router collapse.

---

## 1. compaware (legacy `moe_compression_weight`)

### Math
```
total_loss += -λ_c · Σ_i (mean_prob[i] · w_i)
```
where `w_i = norm(1 / horizon_i)` (so shorter horizons get larger w, i.e.
larger bonus). The bonus is on the BATCH-MEAN router probability — every
sample is pushed in the same direction, regardless of difficulty.

### Code site
`gr00t/model/action_head/flow_matching_action_head.py`:
- Config field `moe_compression_weight: float = 0.0`
- Applied inside `_moe_forward` near the soft-mixture / balance block:
```python
if self.moe_compression_weight > 0:
    inv = 1.0 / torch.tensor(self.moe_expert_horizons, dtype=...)
    inv = inv / inv.sum()
    bonus = -self.moe_compression_weight * (mean_p * inv).sum()
    total = total + warmup * bonus
```

### CLI flag
```
--moe-compression-weight 0.02
```
Persisted to checkpoint config via the `moe_compression_weight` field; the
loaded model knows nothing about it at inference (it only affects training).

### Recommended values (sim experiments)

| Benchmark + decoder set | Safe range | Notes |
|-------------------------|-----------|-------|
| MoE3 (u16/c8/c4) single_pick | 0 (off) | λ_c=0.1 → c4 picks 93%, success drops 33pp on libero |
| MoE4 (u16/c8/c4/u8) single_pick | 0 (off) | λ_c=0.1 still pushes c4 heavily (~70%); libero −12pp, robocasa modest |
| Per-quad 4q×3o (MoE3 decoders) | 0.02~0.03 | quad-level independence prevents chunk-wide collapse; opt2 (λ_c=0.03) gave robocasa 63.1%/234 step (−2.6pp accuracy, −28.5% step) |
| Per-quad+u8 | 0.02~0.03 | same as above, tested as `opt2` ablation |

### When NOT to use compaware
- Single_pick with full coverage decoders (c4 has lowest natural loss → wins
  by both compaware bonus AND quality signal → catastrophic collapse).
- Any setup where success rate is already at the ceiling — extra compression
  push almost always costs quality.

---

## 2. Length-cost on KL target ([D], metaq fork)

### Math
With [B] EMA normalization on (`--moe-normalize-loss-for-kl`), the per-expert
loss is divided by its running EMA (≈ order 1). The KL teacher target is:
```
target_signal[b, i] = (loss[b, i] / EMA_i) + γ · (h_i / H_max)
target_dist[b]      = softmax( - target_signal[b] / τ_target )
KL_supervise        = KL( router_probs || target_dist )
```
The length cost is added INSIDE the softmax of the per-sample target. When
the per-sample loss strongly prefers one expert (`loss[i]/EMA_i` ≪ others),
the +γ·h term is a small perturbation — quality dominates. When losses are
similar (easy sample), the +γ·h term dominates — router shifts to shorter.

### Code site
`gr00t/model/action_head/flow_matching_action_head_metaq.py` (and v2):
```python
if self.moe_normalize_loss_for_kl:
    # ... update EMA ...
    target_signal = loss / EMA                            # ~order 1
else:
    target_signal = loss
if self.moe_length_cost_weight > 0:
    horizons = torch.tensor(self.moe_expert_horizons, ...)
    length_cost = horizons / horizons.max()
    target_signal = target_signal + self.moe_length_cost_weight * length_cost
target_dist = softmax(-target_signal / max(self.moe_target_temp, 1e-3))
```

### CLI flags
```
--moe-normalize-loss-for-kl               # [B] turn on EMA normalization
--moe-loss-ema-momentum 0.99              # default
--moe-length-cost-weight 0.10             # [D] γ in EMA-normalized loss units
--moe-target-temp 0.3                     # softmax temperature
```

### Recommended γ values (EMA-normalized units)

| γ | Effect | Use when |
|---|--------|----------|
| 0 | no compression push | accuracy-only baseline / first run |
| **0.05** | very mild push, almost no quality cost | safer real-robot start |
| **0.10** | moderate push (~10% length penalty vs quality scale) | general default |
| 0.20 | aggressive — over-compression risk | only if you have headroom |
| ≥ 0.30 | likely degrades quality measurably | not recommended |

### Soft-mixture loss is unchanged
Length cost is **only** on the supervise KL target, not on the soft-mixture
loss. The decoders themselves still get clean, unpenalized training signal.
The router is the only module whose target shifts.

---

## 3. Real-robot deployment tips

### Tuning approach
1. **Start with γ=0 ([D] off)** to establish the model's natural routing
   behavior on your task distribution. Log expert pick rates.
2. If you observe the router rarely picks compressed experts (chunk-level
   step count near the raw-expert maximum), bump γ to **0.05**.
3. Re-train for ~10K-20K steps from the no-compaware checkpoint (warm
   start). Compare success rate and step count.
4. If success holds (≤ 1pp drop) and step count drops noticeably, try
   γ=0.10. Otherwise leave at 0.05.

### Sample-conditional behavior in deployment
With [D] active:
- **Precise manipulation** (grasping, insertion, fine alignment) → loss for
  raw expert is markedly lower → router picks raw.
- **Free-space motion / large transit** (reaching, lifting) → all experts
  have similar loss → router picks shorter expert → fewer env-steps.

So real-robot benefit is task-dependent. Workspaces with significant
transit phases benefit most.

### Inference-time monitoring
Log per-chunk router pick and chunk length (`action_pred.shape[0]` for the
metaq fork). Track:
- average chunk length per episode → effective compression rate
- failure modes: which expert was picked just before a failure step?
- expert balance over a session — if one expert dominates >80%, the trade-off
  has collapsed (probably tune γ or temperature).

### Safety knobs for real robot
- Set `--moe-min-prob 0.1` (default 0.05) at training to prevent dead
  experts — guarantees every expert gets some gradient. Robust router.
- Keep `--moe-target-temp 0.3` (default). Sharper temperatures (e.g. 0.15)
  make routing too commit-y on noisy real-world observations.
- Avoid compaware with λ_c > 0.05 on single_pick — collapse risk too high.
- If a task has hard precision-critical moments, train with a slightly
  higher `moe_supervise_weight` (e.g. 0.15) so the router heavily follows
  the quality signal.

### Per-task pre-set
If you have N task families with different precision requirements, a single
γ won't be optimal for all. Two options:
1. **Conservative γ (e.g. 0.05)** for everything — small uniform benefit,
   no precision risk.
2. **Per-task γ override at inference** — re-train heads is unnecessary;
   instead clamp router_probs (e.g. force pick of u16 in precision tasks)
   via the eval client. Useful when task identity is known at deploy time.

---

## 4. Summary

For real-robot work:
- Train with **[B] EMA normalization always on** (essentially free, removes
  expert loss-scale bias).
- Use **[D] length cost γ=0.05~0.10** as the compaware mechanism (safer
  than legacy `moe_compression_weight`).
- Validate per-task that success holds before pushing γ higher.
- Monitor expert pick distribution in the field; tune down if you see
  collapse.

For sim baselines / leaderboards:
- Legacy `moe_compression_weight` at 0.02~0.03 is only safe on per-quad
  routing (independent quad decisions prevent chunk-wide collapse). Stay
  away on single_pick — proven to over-compress.

Code refs:
- legacy compaware: `flow_matching_action_head.py` config field
  `moe_compression_weight`, `_moe_forward` near total-loss line
- [D] length cost: `flow_matching_action_head_metaq{,_v2}.py` config field
  `moe_length_cost_weight`, applied after [B] EMA normalization in the KL
  target block
- Experiment matrix examples:
  `run_scripts/train/robocasa/finetune_*_metaq_n8_with_length_cost_0p10.sh`
  `run_scripts/train/robocasa/finetune_*_pqm_4q_n8_opt2_balanced.sh`
