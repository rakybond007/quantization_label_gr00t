---
name: k-ladder-saturation
description: "Causal K-ladder study — K2 is the closed-loop saturation of temporal quantization; clipping/dynamics unlocks don't recover theoretical speedup; varK protects success only"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-07-31T09:56:33.668Z
---

Full-scale causal study (2026-07-18~20, both benches, 50ep/task each mode) answering
"do SUCCEEDING episodes get the theoretical step reduction at K>2?" — **No, and no
benchmark-code unlock fixes it.**

Paired succ-only step ratio vs K2 (theory K3=1.5x, K4=2.0x), stair-step unlocks:
naive → clipK (command clip removed, `--clip-scale`=K) → dynK (torque caps + OSC kp
also scaled, `--dyn-scale`): LIBERO K3 1.04→0.96→0.97x, K4 1.04→1.00→1.05x;
robocasa K3 1.15→1.11→1.05x, K4 1.23→1.21→1.18x. Success rates identical across
layers. **Eliminated in order: command clipping ✗, robot dynamics ✗ → binding
constraint is the policy-controller closed loop itself** (50ms second-order transient
+ replan re-anchoring absorbs residual error into fresh trajectories). K2 ≈ the
saturation point; beyond-K2 quantization buys no time on any axis.

varK (magnitude-aware variable-K merge, `--vark-bound 0.95`, gripper-transition
break): protects success at forced high compression (LIBERO K3 0.438→0.821, even
above K2's 0.801; robocasa 0.497→0.626) but is SLOWER than K2 (0.8-1.0x) — value is
success preservation, not speed. Client flags live in both eval clients
(`_vark_compress` / `vark_compress_chunk`, `_patch_clip_bounds`, `_patch_dynamics`);
baselines: `baseline_{K2,K3,K4,varK3,varK4,clipK3,clipK4,dynK3,dynK4}` (libero) and
`baseline_compress_*` (robocasa). Conclusions recorded in both
`benchmark_context.txt` files for the evolver.

**2026-07-31 correction (direct probe):** the original clipK/dynK full runs were
silently UNPATCHED — robosuite `env.reset()` re-creates controllers each episode,
wiping patches applied once at env construction (fixed: re-apply after every reset,
eval_taskwise client). A slim srun probe (libero_goal task0, 5ep, per-step
commanded-vs-achieved EE displacement via DIAG_LOG=1 sidecar) then showed:
K3 tracking ratio 0.11 (vs 0.21 @K2); with working clip+kp ×3 it rises to 0.34,
×10 to 0.59 — but success collapses 2/5→0/5 and reversal/tortuosity explode
(0.09→0.16 / 3.0→6.1). So the controller low-pass was PROTECTING K3; once big
jumps actually execute, the policy goes out-of-distribution and thrashes.
**K2-saturation conclusion unchanged but now proven by direct measurement:
binding constraint = policy closed-loop stability, not clipping, not gains.**
Probe artifacts: vlm_gate/output/_diag_recovery/.

**Research implication:** the speed axis closes at K2; the contribution axis is WHERE
K2 applies (VLM gate + TTL) with varK as a success-preserving fallback ablation.
LIBERO raw baseline caveat: `baseline_raw` lacks per-episode steps on 32/40 tasks —
use K2 as the paired anchor. See [[evolver-composite-gating]],
[[terminology-naive-guidance]].
