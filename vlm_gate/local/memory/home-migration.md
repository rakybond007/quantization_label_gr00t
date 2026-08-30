---
name: home-migration
description: Home moved /sjw_alinlab2/home/hojin2 → /sjw_alinlab/home/hojin2 (2026-07-14); first-pass copy running; cutover checklist at ~/_migration/CUTOVER_CHECKLIST.md
metadata:
  type: project
---

The user's home directory moved: OLD `/sjw_alinlab2/home/hojin2` (92%-full volume)
→ NEW `/sjw_alinlab/home/hojin2` (huge new volume). `$HOME`/passwd already point at
the NEW home; the OLD volume stays mounted, and the three in-flight self-evolve
loops keep running against OLD paths (their propagated HOME is old) — do not touch
old-home files they read (esp. old `~/.libero/config.yaml`).

Migration state (2026-07-14): background first-pass copies launched with setsid —
quantization_agent_workspace(26G) / multigpu_workspace / ~/.cache/huggingface /
~/.local + ~/.libero (new copy's config.yaml already re-pointed to the new home) —
plus a conda-pack pipeline (fresh Miniconda base at NEW ~/miniconda3, then
quant_gate / quant_gate_eval / vlm_judge / libero packed → untarred → conda-unpack;
editable installs excluded, must re-link at cutover). `.claude` was merged into the
new home with project keys remapped (`-sjw-alinlab2-home-*` → `-sjw-alinlab-home-*`),
so memory/context follow the new workspace path.

Key facts: our scripts contain ZERO old-home literals (all $HOME-relative → adapt
automatically). The uv venvs (cosmos_judge_venv, vllm_judge_venv) need shebang +
pyvenv.cfg sed at cutover. FINAL steps live in
`/sjw_alinlab/home/hojin2/_migration/CUTOVER_CHECKLIST.md` — delta rsync after the
loops finish, uv venv fix, libero editable re-link, .bashrc conda init path, smoke
tests. Logs: `~/_migration/logs/`. See [[libero-install-repoint]],
[[vlm-gate-env-separation]].
