# 4-Expert MoE Action Head

Variable-horizon Mixture-of-Experts on top of the GR00T N1.5 flow-matching action head. A small router picks one of 4 experts per chunk; each expert produces an action chunk of its own horizon (16, 8, 8, or 4). Goal: preserve the success rate of the H=16 baseline while letting the router pick a shorter chunk on easy segments to cut wall-clock rollout time.

## Experts

`H` denotes the configured `action_horizon` (16 for robocasa). All targets are derived from the GT trajectory `actions[:, :H]`.

| idx | name | output horizon | source horizon (body) | target |
|-----|------|----------------|-----------------------|--------|
| 0 | `main` | H (=16) | H | `actions[:, :H]` (raw) |
| 1 | `m8`   | H/2 (=8)  | H | `_compress_actions(actions[:, :H], factor=2)` (sum-pair) |
| 2 | `m4`   | H/4 (=4)  | H/2 | `_compress_actions(actions[:, :H/2], factor=2)` (sum-pair) |
| 3 | `n8`   | H/2 (=8)  | H/2 | `actions[:, :H/2]` (raw) |

`_compress_actions` sums `factor` adjacent steps for continuous dims (delta accumulation). For dims listed in `discrete_action_dims` (e.g. gripper, control_mode) it takes the LAST value of each group — sum is meaningless for binary signals.

Experts 1 and 2 are "merged" (sum-pair compressed); experts 0 and 3 are "raw". m8 and n8 both produce 8 steps but represent different temporal extents (16 sim steps vs 8). The router can in principle pick m8 (long horizon, coarser) or n8 (short horizon, fine) for the same situation.

## Body modes

`moe_body_mode` controls how the shared DiT body is invoked across experts:

### `per_expert_h` (Option B — used in production)
Each expert runs a separate body forward at its source horizon. Decoders are existing `CategorySpecificMLP` modules:
- `action_decoder` (main, H steps)
- `m8_action_decoder`, `m4_action_decoder`, `n8_action_decoder` (4 / 8 step variants)

Inference cost = single body forward at the picked expert's horizon (cheap for m4/n8/m8). Training cost = 4 body forwards per batch (one per expert).

### `shared_h16` (Option A — implemented, lower success in evals)
One body forward at h=H. Output `body_out` is fed to per-expert `ExpertHead` modules:

```
ExpertHead(target_h, body_dim, action_dim, n_layers)
  query_tokens : (target_h, body_dim)   -- learned
  cross-attn   : queries cross-attend to body_out
  proj         : body_dim -> action_dim
```

Compression (e.g. 16→8) is fully learned via the cross-attn rather than fixed pooling. m4/n8 read only the first `H/2` positions of `body_out` (their "source slice"). Inference uses an x_t reconstruction trick across denoising steps because `body_out` length stays at H while `x_t` lives in the expert's `target_h` space (`_moe_shared_h16_denoise`).

In practice `per_expert_h` won by ~7%p success on robocasa (0.669 vs 0.597) — the cross-attn experts in shared_h16 didn't get enough capacity to recover the expert-specific representations.

## Router

`head_router`: 2-layer MLP `[backbone_dim + state_dim] -> moe_router_hidden -> num_experts(=4)`.

Inputs: mean-pooled VL embeds ⊕ mean-pooled state features (one vector per sample).

Training-time probabilities:
```
log_router   = log_softmax(logits / moe_router_temp)
router_probs = exp(log_router)                       # single derivation chain (DDP-friendly)
router_probs = clamp(router_probs, min=moe_min_prob)
router_probs = router_probs / router_probs.sum(-1, keepdim=True)
```

`moe_min_prob` (default 0.05) prevents dead experts: every expert always gets at least `min_prob × loss_i` of gradient even if the router commits.

`moe_uniform_warmup_steps` (default 0; set >0 to enable) forces probs to `1/K` for the first N steps so all experts learn from real GT before the router differentiates.

## Loss

```
total = soft_mixture
      + warmup * moe_balance_weight   * balance
      + warmup * moe_supervise_weight * KL_supervise
      + 1e-12  * anti_unused_L2

soft_mixture  = (router_probs * per_sample_losses).sum(-1).mean()
balance       = ((mean_router_probs - 1/K) ** 2).sum()
target_dist   = softmax(-per_sample_losses.detach() / moe_target_temp)
KL_supervise  = KL(router_probs || target_dist)
warmup        = min(1.0, step / moe_router_warmup_steps)
```

- **`per_sample_losses` (B, 4)**: per-expert flow-matching MSE, computed at each expert's `target_h`. In `shared_h16`, expert `i`'s noise is derived from the shared `noise_full` to keep the flow-matching theory consistent (e.g. m8's noise = sum-pair of `noise_full`).
- **soft mixture**: standard soft-MoE — gradient flows through both `router_probs` (router learns to favor low-loss experts) and `losses` (experts keep learning).
- **load balance**: pushes the batch-mean router distribution toward uniform.
- **supervised KL**: per-sample router target = softmax of negative losses (lower loss → higher target prob).
- **anti-unused L2**: tiny `sum(p**2) * 1e-12` over every expert/decoder param. Required because DDP's reducer expects every parameter to receive gradient on every iteration; with `min_prob` and dropout-style routing, some expert params can otherwise have zero grad on a given batch and trigger reducer errors. Magnitude is too small to affect learning.

## Inference

`head="moe"` selects the routing path. Router runs once on the same pooled features, `argmax` (or `multinomial` if `moe_inference_stochastic=True`) picks one expert, only that expert is decoded:

- `per_expert_h`: one body forward at `horizons[picked]`, then the picked decoder's denoise loop.
- `shared_h16`: body forward at H, then the picked `ExpertHead` runs the denoise loop with x_t in `target_h` space.

The response includes `moe_picked` (expert id) and `moe_probs` for analysis.

The eval client (`scripts/robocasa_service_moe.py`) consumes the entire returned chunk before re-routing — chunk length itself determines replan cadence. Router picks are tallied and written to `prediction.txt` as `router_picks: {16: A, 8: B, 4: C}`.

## Configuration cheat sheet

Required flags to enable training (all 4 experts must be active):
```
--use-merged-8-head --use-merged-4-head --use-native-8-head
--use-moe-routing --moe-body-mode=per_expert_h --moe-num-experts=4
```

Tuned defaults that worked on robocasa:
```
--moe-router-temp=0.5         # softmax temperature on router logits (training)
--moe-target-temp=0.3         # softmax temperature on -loss (KL target)
--moe-balance-weight=0.05     # uniformity regularizer
--moe-supervise-weight=0.1    # KL(router || softmax(-loss / τ))
--moe-router-warmup-steps=5000
--moe-min-prob=0.05           # anti-collapse floor (default)
--moe-expert-n-layers=2       # only used in shared_h16 mode
--discrete-action-dims 6 11   # robocasa: gripper, control_mode (not summed)
```

Inference (server side):
```
--head moe --denoising-steps 4
```

## Related files

- `gr00t/model/action_head/flow_matching_action_head.py` — `ExpertHead`, `_moe_forward`, `_moe_shared_h16_denoise`, MoE config fields
- `scripts/gr00t_finetune.py` — CLI flags + late-attach setup for the new heads + router
- `scripts/robocasa_service_moe.py` — variable-horizon eval client (full chunk consumption)
- `run_scripts/train/robocasa/finetune_gr00t_n1_5_moe4_per_expert_body.sh` — production training recipe
- `run_scripts/train/robocasa/finetune_gr00t_n1_5_moe4_shared_body.sh` — shared_h16 variant
- `run_scripts/eval/eval_robocasa_moe4_per_expert.sh`, `eval_robocasa_moe4_shared_body.sh` — eval recipes
