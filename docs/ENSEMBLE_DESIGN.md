# Ensemble Design for GR00T-N1.5 Multi-Horizon Training

This document collects the ensemble-related training and inference designs built
on top of the base GR00T-N1.5-3B action head. It covers:

1. Baseline (no multi-horizon)
2. Multi-horizon auxiliary heads (f2, f4) + discrete-dim fix
3. Native 8-step decoder (`merged_8`, a.k.a. `m8`)
4. Inference-time ensembles (WLS, ensemble, ensemble_fix)
5. Training-time ensemble-aware losses (cross-head consistency)
6. Summary table of variants and how to train / eval each

Everything discussed here lives in
`gr00t/model/action_head/flow_matching_action_head.py` on the training side and
`scripts/inference_service.py` / `scripts/robocasa_service*.py` on the eval side.

---

## 1. Setting and Notation

- Action dim for `single_panda_gripper`: 12 (indices 0..11)
- **Discrete dims**: 6 (`gripper_close`) and 11 (`control_mode`) — binary
- **Continuous dims**: rest (EE pos/rot, base motion)
- Every action is a `delta` from the current state, so sums are meaningful for
  continuous dims (sum-of-deltas = total displacement).

We write the network's prediction at env-step granularity as
`a[0], a[1], ..., a[15]` for a 16-env-step chunk. For discrete dims, "sum" is
meaningless; we use "last value of the group" instead.

---

## 2. Decoder Heads Trained Jointly

All heads share the same DiT body, VL encoder, state encoder, action encoder,
and position embedding. Each head has a dedicated `CategorySpecificMLP`
decoder. Differences are only in (a) the decoder's target shape/semantics and
(b) optional gradient scaling into the shared body.

| Head | Horizon | Each step covers | Target = compress(actions_ext, factor) | Body grad |
|------|---------|------------------|----------------------------------------|-----------|
| `main` (a.k.a. main16) | 16 | 1 env-step | `actions[:, :16]` (identity) | 1.0 |
| `f2`                   | 16 | 2 env-steps | `_compress(actions_ext[:, :32], f=2)` | **0.1** (aux) |
| `f4`                   | 16 | 4 env-steps | `_compress(actions_ext[:, :64], f=4)` | **0.1** (aux) |
| `m8` (`merged_8`)      | **8** | 2 env-steps | `_compress(actions[:, :16], f=2)` | 1.0 (co-main) |

The first 8 rows of `f2`'s target are **structurally identical** to `m8`'s full
target — both are `sum-of-2` deltas over the first 16 env-steps. This gives us
a free consistency signal (Section 5).

### `_compress_actions` with discrete-dim fix

```
grouped        = actions.reshape(B, H, factor, D)
compressed     = grouped.sum(dim=2)              # continuous: sum
if discrete_action_dims:
    compressed[..., disc_idx] = grouped[:, :, -1, :][..., disc_idx]   # discrete: last-of-group
```

Without the fix, binary gripper/control signals become noisy targets (e.g.
gripper=2 from summing two 1's). **All "discfix" variants below include this
fix.**

---

## 3. Training Losses

Let `L_head` be the flow-matching MSE loss for that head. All losses are
trained jointly on a single forward pass of the shared body.

### 3.1 Base multi-horizon loss (pre-merged_8, pre-consistency)

```
L_total = 1.0 * L_main
        + warmup(step) * (1.0 * L_f2 + 1.0 * L_f4)
```

- `warmup(step) = min(1, step / aux_loss_warmup_steps)` — ramps aux weights 0→1
  over 5000 steps by default.
- Aux losses use `GradScale(0.1)` on the path into the shared DiT body. The
  aux decoders themselves receive full gradient.
- Main head is protected from aux interference by (a) `grad_scale_to_body=0.1`
  on aux losses and (b) warmup.

### 3.2 Adding `merged_8` (co-main)

```
L_total = 1.0 * L_main
        + 1.0 * L_m8                                         (new)
        + warmup(step) * (1.0 * L_f2 + 1.0 * L_f4)
```

- `m8` is a *native* 8-step decoder — its output shape is `(B, 8, D)`.
- Treated as a second main: full gradient to body, no warmup. Rationale: it is
  as high-priority as main for downstream eval (native 8-step execution).

### 3.3 Adding the ensemble-consistency loss

The three heads `main` (pair-summed), `f2[:8]`, and `m8` all predict the **same
quantity**: `[a[0]+a[1], a[2]+a[3], ..., a[14]+a[15]]`. Because their targets
are mathematically identical, a cross-head consistency signal is both cheap
(zero extra forward passes) and well-defined.

**Shared timestep**: without a shared `t`, each head sees a different
noise level per batch and their predicted velocities are not directly
comparable. We sample one `t ~ Beta(alpha, beta)` per batch and pass it to all
three heads.

**Anchor choice**: `f2[:8]` is the anchor; `main`-pooled and `m8` are pulled
toward it (f2 is chosen because it already covers a wider context and is less
likely to diverge).

```
v_main_8  = v_main[:, 0::2, :] + v_main[:, 1::2, :]        # (B, 8, D)
v_f2_8    = v_f2[:, :8, :]                                  # (B, 8, D)
L_consist = MSE(v_main_8, v_f2_8) + MSE(v_m8, v_f2_8)
```

All three gradients flow (no stop-grad) — this is mutual pull, not distillation.

```
L_total += c_warmup(step) * 0.1 * L_consist
```

- `c_warmup(step) = min(1, step / 2000)` — ramps 0→1 over the first 2k steps.
- Weight 0.1 keeps the regularizer small compared to main task losses.
- `f4` is intentionally excluded from consistency: its scale (4 env-steps) is
  different from the 2-env-step "m8 unit," and handling it would need WLS
  (section 4.3), defeating the "simple and cheap" goal.

### Why this works cheaply

- No additional forward passes. The velocities `v_main`, `v_f2`, `v_m8` are
  already produced by each head's own flow-matching loss.
- Gradient path is just a couple of MSEs through existing predicted tensors.
- Shared `t` is a minor refactor — flow matching is unbiased regardless of
  whether `t` is independent or shared per head.

---

## 4. Inference-Time Ensembles

At inference each head runs its own flow-matching denoising loop (4 steps, each
denoises once). Heads are then combined to form a single action chunk.

### 4.1 `main` (baseline)

Just run `action_decoder` → 16-step chunk. No ensemble.

### 4.2 `m8` (native 8-step)

Run `m8_action_decoder` → 8-step chunk. Each step commands a sum-of-2 delta,
so the client executes 8 env-steps and the robot traverses the same distance
as a 16-step main chunk.

Pro: shortest rollouts (-33% env steps vs. main on successful episodes).
Con: trained delta magnitudes are larger, which can saturate the controller.

### 4.3 `ensemble` (Weighted Least Squares over main + f2 + f4)

Given predicted deltas `main[0..15]`, `f2[0..15]`, `f4[0..15]`, solve:

```
min_a  Σ_i  (a[i] - main[i])^2                           # H=16 eqs, weight 1
     + Σ_i  (1/2) * (sum(a[2i..2i+1]) - f2[i])^2          # 8 eqs, weight 1/2
     + Σ_i  (1/4) * (sum(a[4i..4i+3]) - f4[i])^2          # 4 eqs, weight 1/4
```

Weights are `1/f` (variance of sum of `f` i.i.d. samples is `f*sigma^2`). In
matrix form: `a* = (A^T W A)^-1 A^T W b`. Currently implemented with an
explicit `torch.linalg.inv` call inside an `autocast(enabled=False)` block
because `inv` does not support bf16.

Pro: statistically principled combination; best success rate on our eval
(68.0% vs. 66.75% baseline).
Con: numerics (inv) are sensitive; bf16 needs casting; 16-step output so
rollouts are the same length as main.

### 4.4 `ensemble_fix`

Same as `ensemble` but after solving, the discrete dims are overwritten with
`main`'s prediction:

```
a[..., disc_idx] = main[..., disc_idx]
```

Motivation: the LS system averages binary signals (gripper, control_mode),
which produces fractional values that don't correspond to any valid command.
Using `main`'s binary decision directly is semantically clean.

### 4.5 `f2` / `f4` standalone (diagnostic)

Useful mostly for debugging — each alone lags `main` substantially because
training prioritizes `main` (the aux heads have grad-scale 0.1 into the body
and are warmed up).

---

## 5. Variants Trained So Far

| # | Name | Extras vs. baseline | CKPT dir (60k) | Eval result (1200 ep) |
|---|------|---------------------|----------------|-----------------------|
| 0 | `baseline`              | none                                    | `groot_n1_5_bs64_baseline`                | **66.75%**¹ |
| 1 | `baseline + merge`      | client-side 16→8 post-hoc merge         | (same ckpt as #0, different eval client)  | **59.42%** (−33% rollout) |
| 2 | `multi_horizon`         | f2 + f4 aux heads                       | `groot_n1_5_bs64_multi_horizon`           | 67.16% (main), 68.00% (ensemble), 66.98% (ensemble_fix) |
| 3 | `multi_horizon_discfix` | #2 + discrete-dim fix in `_compress`    | `groot_n1_5_bs64_multi_horizon_discfix`   | TBD (Job 290439, resumed) |
| 4 | `mh + m8`               | #3 + native 8-step head                 | `groot_n1_5_bs64_mh_m8_discfix`           | TBD (Job 291228) |
| 5 | `mh + m8 + econsist`    | #4 + ensemble-consistency loss          | `groot_n1_5_bs64_mh_m8_econsist_discfix`  | TBD |

¹ Evaluated on 1137/1200 ep; projection to 1200 is ~66%.

---

## 6. Quick Reference: Training & Eval Commands

### Train variant #5 (all features on)

```bash
sbatch run_scripts/train/robocasa/finetune_gr00t_n1_5_mh_m8_econsist.sh
```

Key flags the script passes:

```
--use-multi-horizon-loss
--multi-horizon-factors 2 4
--multi-horizon-loss-weights 1.0 1.0
--aux-grad-scale-to-body 0.1
--aux-loss-warmup-steps 5000
--discrete-action-dims 6 11
--use-merged-8-head
--merged-8-weight 1.0
--use-ensemble-consistency-loss
--ensemble-consistency-weight 0.1
--ensemble-consistency-warmup-steps 2000
```

### Eval a checkpoint with a given head

Server (uses `inference_service.py`):

```bash
python scripts/inference_service.py --server \
    --model_path <CKPT_PATH> \
    --data_config single_panda_gripper \
    --embodiment_tag new_embodiment \
    --denoising_steps 4 \
    --head <HEAD> \
    --discrete-action-dims 6 11
```

`<HEAD>` ∈ `{main, f2, f4, m8, ensemble, ensemble_fix}`.

Client:

| Head | Client script | `--action_horizon` |
|------|---------------|--------------------|
| `main`, `f2`, `f4`, `ensemble`, `ensemble_fix` | `scripts/robocasa_service.py` | 16 |
| `m8`                                           | `scripts/robocasa_service.py` | 8  |
| `main` + post-hoc merge (variant #1)           | `scripts/robocasa_service_merged.py` | 16 (server) / merged to 8 (exec) |

### Preset SLURM scripts

| Variant | sbatch script |
|---------|---------------|
| #0 baseline eval             | `run_scripts/eval/eval_baseline_full.sh`         |
| #1 baseline + merge eval     | `run_scripts/eval/eval_baseline_merged.sh`       |
| #2/#3 multi-horizon eval     | `run_scripts/eval/eval_multi_horizon_full.sh` (set `HEAD={main,f2,f4,ensemble,ensemble_fix}` via `--export=ALL,HEAD=...`) |

---

## 7. Design Choices & Tradeoffs

| Concern | Decision | Rationale |
|---------|----------|-----------|
| f2/f4 horizon | 16 (not 8/4) | Reuses 16-step DiT query structure; avoids custom pos-embeds. |
| m8 horizon    | **8** (native)  | The whole point — produce a true 8-step output the client can execute as-is. DiT handles variable horizon; only decoder is new. |
| Body grad for aux | scale 0.1 | Prevents aux losses from pulling main off its own optimum early in training. |
| Main vs co-main | m8 is co-main (full grad) | Its 8-step output is as important as main for downstream use. |
| Aux warmup steps  | 5000 | Lets main stabilize before aux kicks in. |
| Consistency anchor | f2[:8] | f2 has the widest context (32 env-steps), least likely to drift. |
| Consistency weight | 0.1 (post warmup) | Regularizer strength, not task strength. Empirically safe. |
| Consistency warmup | 2000 | Avoids noisy early-step velocities poisoning main. |
| f4 in consistency | Excluded | Different scale (4 env-steps) would need WLS — defeats "simple" goal. |
| bf16 and `torch.linalg.inv` | Cast to fp32 in `autocast(enabled=False)` block | `inv` doesn't support low-precision. |
| Discrete dim handling in compress | last-of-group (not sum) | Binary signals — sum is meaningless. |
| Discrete dim handling in LS (ensemble_fix) | Overwrite with main | Averaging binary signals is meaningless. |

---

## 8. Implementation Map

| Concept | File : function / class |
|---------|------------------------|
| Config fields                  | `flow_matching_action_head.py : FlowmatchingActionHeadConfig` |
| Decoder heads                  | `flow_matching_action_head.py : FlowmatchingActionHead.__init__` |
| Flow-matching loss (per head)  | `flow_matching_action_head.py : FlowmatchingActionHead._flow_matching_loss` |
| Compression w/ discrete fix    | `flow_matching_action_head.py : FlowmatchingActionHead._compress_actions` |
| Joint training loss assembly   | `flow_matching_action_head.py : FlowmatchingActionHead.forward` |
| Denoise loop (variable horizon)| `flow_matching_action_head.py : FlowmatchingActionHead._denoise_with_decoder` |
| Single-head inference          | `flow_matching_action_head.py : FlowmatchingActionHead.get_action (head='main'/'f2'/'f4'/'m8')` |
| WLS ensemble                   | `flow_matching_action_head.py : _build_ensemble_ls_matrix + _ensemble_least_squares` |
| `ensemble` / `ensemble_fix`    | `flow_matching_action_head.py : get_action (head='ensemble' / 'ensemble_fix')` |
| CLI flags                      | `scripts/gr00t_finetune.py : ArgsConfig` |
| Server-side head selection     | `scripts/inference_service.py` (`--head` arg) |
| Client (16-step)               | `scripts/robocasa_service.py` |
| Client (16→8 post-hoc merge)   | `scripts/robocasa_service_merged.py` |
