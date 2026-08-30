---
name: libero-preemption-gate-bug
description: LIBERO gated eval lost per-episode gate data on background-partition requeue; fixed with durable sidecar
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
---

The LIBERO gated-quant eval client (`Isaac-GR00T/gr00t/eval/libero/eval_taskwise_gr00t_quantize.py`)
accumulated per-episode records (steps, gate quant decisions) **in memory** and wrote
`prediction.txt` only once at task end. Its resume path skipped episodes whose replay
video already existed but did NOT re-record them. On the `background` SLURM partition,
jobs get **preempted and requeued**; a requeued task found all videos present, skipped
every episode, and wrote an **empty** `prediction.txt` (0 episode lines, no
`gate_quantize_rate`). `N_results.txt` still showed a success *rate* (counted from video
files) but `succ-only steps: 0.00 (over 0 ep)` — so the evolver silently saw a partial
task set (e.g. cycle-1: gemma 36/40, cosmos 28/40 tasks) and computed macro metrics over
a biased subset. Whole task_idx slices (the requeued array jobs: gemma array_6, cosmos
array_4/8/9) vanished from the evolver's view.

**Why it matters:** the self-evolve accept/reject gating compares cycles by macro
success/steps/quant; if each cycle covers a different 36–28/40 subset, the comparison
(and the v9-style running-best baseline) is invalid. Corrupted data is **unrecoverable**
(gate_conf.csv had only its header; results.txt lacks per-ep steps).

**Fix:** durable per-episode JSONL sidecar `<gate_out>/<suite>_<idx>/ep_records.jsonl`,
appended+flushed the instant each episode finishes (BEFORE the video write); on (re)start
the task loop reloads it to reconstruct every aggregate exactly once and skips episodes
by sidecar membership, not video existence. Logic verified by a preemption-simulation
unit test (kill at ep30 → requeue → full 50-ep prediction.txt with correct steps +
gate_quantize_rate). GPU integration test in the tmux 1:1 pane still pending (alloc
expired).

**robocasa client is NOT affected** (`vlm_gate/scripts/robocasa_service_compress.py`):
it appends each episode's success+steps directly to `prediction.txt` (line ~244) and
resumes by reading it — designed preemption-safe. Minor caveat: its end-of-run
`gate_quantize_rate` may undercount on resumed episodes (gate_yes/total are in-memory
counters not reloaded), but success/steps are correct, so completed robocasa results
stand. See [[ops-download-and-bg-task-lessons]] (verify-by-artifact, monitor long tasks)
and [[vlm-gate-env-separation]].
