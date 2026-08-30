# RoboCasa365 Setup — Run Log

## Investigation Steps

1. `find ... -iname "*365*"` in workspace → no hits. No existing local "365" artifacts.
2. Listed `multigpu_workspace/robocasa/` → standard robocasa clone (LICENSE, README, robocasa/, setup.py, docs/). README cites v0.2 (Oct 2024) but the cloned `robocasa/environments/kitchen/` matches the v1.0 (RoboCasa365) layout.
3. Listed `robocasa/environments/kitchen/single_stage/` → 8 task family files (coffee, doors, drawer, microwave, navigate, pnp, sink, stove).
4. Listed `robocasa/environments/kitchen/multi_stage/` → 20 activity directories (baking, boiling, brewing, chopping_food, clearing_table, defrosting_food, frying, making_toast, meat_preparation, mixing_and_blending, reheating_food, restocking_supplies, sanitize_surface, serving_food, setting_the_table, snack_preparation, steaming_food, tidying_cabinets_and_drawers, washing_dishes, washing_fruits_and_vegetables).
5. Class count via Python regex over kitchen tree → 123 task class definitions (includes intermediate base classes; instance-level split is 65 atomic + 300 composite per the paper).
6. Enumerated tasks in `kimtaey/robocasa_mg_gr00t_300/meta/episodes.jsonl` → 24 unique task names (all atomic, original RoboCasa v0.2 family).
7. Web search "robocasa 365 tasks benchmark NVIDIA GR00T" → top hit is the **arXiv 2603.04356** paper *"RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots"* (ICLR 2026). Confirmed: 65 atomic + 300 composite = 365.
8. Web search "site:huggingface.co robocasa365" → found `nvidia/robocasa365-datasets`. WebFetch on it returned HTTP 401 (gated / requires auth).
9. WebFetch on `https://huggingface.co/kimtaey` → only `_30 / _100 / _300` robocasa datasets exist; no `_full / _365` variant.
10. WebFetch on robocasa releases → confirmed v1.0 = "RoboCasa365 release" (Feb 2026), v0.2 was the original (Oct 2024 / Feb 2025).
11. Read existing `Isaac-GR00T/run_scripts/train/robocasa/finetune_gr00t_n1_5_baseline.sh` and `..._mh_m8_econsist.sh` → used as templates (preserved Slurm header, conda activation pattern, batch=32, 2 GPUs, 60k steps, `single_panda_gripper` config, all flags).
12. Read existing `Isaac-GR00T/run_scripts/eval/eval_gr00t_robocasa.sh` → confirmed eval client is `Isaac-GR00T/scripts/robocasa_service.py --client` driven; reused unchanged in the doc's eval section.

## Files Created

- `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/run_scripts/train/robocasa365/finetune_robocasa365_baseline.sh`
- `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/run_scripts/train/robocasa365/finetune_robocasa365_mh_m8_econsist.sh`
- `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/docs/robocasa365_setup.md`
- `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/docs/robocasa365_setup_log.md` (this file)

## Actions NOT Taken (per instructions)

- No sbatch submitted.
- No HF download initiated (`nvidia/robocasa365-datasets` is multi-TB and gated).
- Existing 24-task run scripts left untouched.

## Confidence

- "RoboCasa365 = 365 tasks (65 atomic + 300 composite)": **high** (matches paper, NVIDIA blog, robocasa.ai).
- Cloned `robocasa/` already supports v1.0 task surface: **high** (multi_stage tree exists with all 20 activity classes from the paper).
- `nvidia/robocasa365-datasets` is the right dataset to stage: **medium-high** (only HF dataset matching the name; gated access prevents direct verification of contents/format).
- Format compatibility with `gr00t_finetune.py` (LeRobot v2+): **uncertain** — likely needs a conversion pass; flagged in the setup doc.
