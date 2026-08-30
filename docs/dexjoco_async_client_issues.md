# DexJoCo async eval client — known issues (deferred)

Status: **deferred**. Use `scripts/dexjoco_eval_gr00t_sync.py` (sync wrapper)
for evals until these are fixed upstream.

## Summary
`external_dependencies/dexjoco/dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py`
mis-handles MoE-emitted short chunks (e.g. m8 = 8 steps), causing the robot to
remain effectively frozen during a rollout. Diagnosed via controlled ablation
on tmux GPU; full results in `analysis/_smoke/dexjoco_moe_ablate*/`.

## What we observed (hammer_nail, 60k MoE ckpt, head=moe → router picks m8)
| eval client variant                        | succ  | frame diff |
|--------------------------------------------|:-----:|:----------:|
| async default                              | 0/1   | 0.034 (frozen) |
| async + `--replan-ratio 0`                 | 0/2   | 0.015 (worse) |
| async + no per-action stale-slicing+drop   | 0/2   | 0.538 (motion) |
| async + no_blend only                      | 0/2   | 0.048 (frozen) |
| async + no_stale_drop only                 | 0/2   | 0.538 (motion) |
| **sync wrapper**                           | **2/2** | **1.168 (full)** |

Baseline ckpt at 16-step chunks works under async because chunks survive the
slice/drop with margin (16 - inference_latency ≈ 11 actions remain). 8-step
chunks lose most actions and the robot stays.

## Root cause (in `receive_actions` of upstream client)
1. **Stale-chunk drop** at `if action_chunk_timestamp_range[1] <= now_timestamp: continue`
   silently discards chunks whose end timestamp has passed. With 8-step chunks
   and ~5 dt inference latency this often discards the whole chunk.
2. **Per-action staleness slicing**:
   `action = action_chunk.action[(now - chunk.ts) : (now + chunk_len - chunk.ts)]`
   trims off the leading 5 actions of each chunk that arrives 5 dt after obs.
3. Combined with the eager replan trigger (`buffer < 0.8 * 30 = 24` always true
   for any 8-step chunk), nearly every chunk yields only a handful of usable
   actions, and `env.stay()` dominates between replans.

## Proposed upstream fix (for later)
- Adapt the staleness/slicing logic to chunk length: when `chunk_len < some
  threshold`, do not slice and do not drop; instead start at the head of the
  chunk and overwrite buffer/timestamp accordingly.
- Or take the entire chunk verbatim once per `infer()` and disable
  blending/slicing entirely — effectively the sync behavior with optional
  asynchronous pipelining.

## Current workaround
`scripts/dexjoco_eval_gr00t_sync.py` — calls `client.infer(get_obs()) ->
execute all chunk actions -> infer()` with no buffer, no slicing, no drop.
Verified PASS for MoE head=moe (2/2 hammer_nail on smoke) and used in the
production sbatch (`run_scripts/eval/eval_dexjoco_single_arm_*`).

## Confirmed by ablation (smoke n=2, hammer_nail, MoE head=moe)

| variant                                    | succ  | frame diff |
|--------------------------------------------|:-----:|:----------:|
| async default                              | 0/1   | 0.034 |
| async + `--replan-ratio 0`                 | 0/2   | 0.015 |
| async + no_blend                           | 0/2   | 0.048 |
| async + no_stale_drop                      | 0/2   | 0.538 |
| **async + no_slice + no_drop + no_blend**  | **2/2** | (sync-level motion, ablate3 rerun) |
| sync wrapper                               | 2/2   | 1.168 |

The third row above is the proposed patch: replace `receive_actions` with the
"append chunk verbatim starting at now_timestamp" variant (see ablate3 wrapper
in this commit's history).
