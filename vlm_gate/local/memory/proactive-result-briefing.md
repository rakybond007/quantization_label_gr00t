---
name: proactive-result-briefing
description: "User wants immediate briefings whenever an experiment result lands, without being asked"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-07-31T11:51:55.731Z
---

2026-07-31: "실험 돌리는데 결과가 나온것들이 있으면 브리핑을 그때 그때 해줘."

**Why:** the user checks in irregularly and doesn't want to poll; results were
sitting finished (A' cosmos, diag probes) until they asked.

**How to apply:** when submitting slurm jobs whose results matter, also start a
background watcher shell (squeue poll loop that exits when the job set drains) so
the completion notification triggers an unprompted briefing turn. Brief with
numbers + interpretation, not just "done". See [[scout-message-discipline]] for
tone; keep each briefing outcome-first.
