---
name: libero-install-repoint
description: Disk-full incident wiped ~/neurips_2026_workspace (incl. LIBERO); libero paths now repointed to ~/multigpu_workspace/LIBERO via ~/.libero/config.yaml
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
---

During the 2026-07-13 shared-volume disk-full incident, `~/neurips_2026_workspace/`
(including its LIBERO checkout) was deleted entirely. Symptom: every LIBERO eval
client crashed at env init with FileNotFoundError on
`.../neurips_2026_workspace/LIBERO/libero/libero/init_files/...pruned_init`, while
array jobs still showed COMPLETED in ~40-60s (clients die instantly; tasks whose
gate outputs already existed masked the failure → partial 36/40 coverage that
silently tainted two self-evolve attempts, quarantined under
`analysis/_evolver/_archive_enospc{,2}/`).

**Fix (do NOT re-clone into neurips path):** the libero conda env's editable
install already pointed at the surviving complete copy `~/multigpu_workspace/LIBERO`;
only `~/.libero/config.yaml` still held the dead neurips paths (assets, bddl_files,
benchmark_root, init_states). Repointed all of them to
`/sjw_alinlab2/home/hojin2/multigpu_workspace/LIBERO/...` (backup:
`~/.libero/config.yaml.bak_enospc`) and verified by loading task-5 init states of
all 4 suites via `get_libero_path("init_states")`. The `datasets:` warning is
harmless for eval.

**Why it matters:** `~/.libero/config.yaml` is user-global — any future LIBERO
work resolves paths through it, and a dead path there fails ALL libero runs while
sbatch arrays still exit "COMPLETED". When LIBERO evals suddenly produce whole
missing task_idx slices with sub-minute job times, check this config first.
See [[libero-preemption-gate-bug]] (the coverage-count symptom looks identical),
[[evolver-composite-gating]].
