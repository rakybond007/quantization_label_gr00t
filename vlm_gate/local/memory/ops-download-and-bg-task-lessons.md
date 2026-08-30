---
name: ops-download-and-bg-task-lessons
description: "Operational lessons — never mask exit codes with pipes, disable hf-xet for large HF downloads, verify by artifact, monitor long bg tasks"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
---

Three mistakes made downloading Cosmos3-Nano (32GB) that must not recur:

1. **Never `cmd | tail` a command whose success you care about.** The pipeline's
   exit code is `tail`'s (0), masking the real failure. The hf download was
   `Killed` but the background task reported exit 0, so it was falsely reported as
   "complete." Use `set -o pipefail`, or redirect to a log with no pipe so the
   task exit == the command exit, AND **verify by artifact** (file count/size),
   not the exit code.
2. **Large HuggingFace downloads: set `HF_HUB_DISABLE_XET=1`.** hf-xet got
   SIGKILLed (even with 66G RAM free) and stalled at metadata-only. Plain HTTP
   then ran clean at ~170MB/s. Also `rm blobs/*.incomplete` before retrying
   (xet→http partial-blob formats differ). `--max-workers 4` to bound memory.
3. **Don't fire-and-wait on long background tasks.** The user noticed a long gap
   with no progress because the download had died silently. Checkpoint progress
   (e.g. `du -sm` the cache twice to confirm it's growing) instead of only
   waiting for the completion notification.

**Why:** the user re-requested an expired GPU alloc before realizing the download
had actually failed — wasted wall-clock + eroded trust.

**How to apply:** for any download/build in this workspace, log to a file (no
pipe-masking), disable xet for HF pulls, and confirm by inspecting the produced
artifact. See [[cosmos-judge-model]], [[vlm-gate-env-separation]].
