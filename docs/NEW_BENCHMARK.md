# Adding a New Benchmark to the GR00T-N1.5 Stack

Status: 2026-08-31. Written from a survey of `robocasa365` and `RoboCasa GR-1`.
Nothing in this doc has been trained or submitted; every claim marked "verified"
was checked against code or data on disk, and everything unverified is labelled.

Repos referred to below:

| Alias | Path |
|---|---|
| `ISAAC` (private fork, `quant_gate` env) | `/sjw_alinlab/home/hojin2/quantization_agent_workspace/Isaac-GR00T` |
| `ISAAC_SHARED` | `/sjw_alinlab/home/hojin2/multigpu_workspace/Isaac-GR00T` |
| `WS` | `/sjw_alinlab/home/hojin2/quantization_agent_workspace` |

---

## The six touch points (checklist)

### 1. LeRobot v2.0-compatible layout

GR00T's loader formats `video_path` with `episode_chunk` / `episode_index`
(`gr00t/data/dataset.py:627-631`) and reads `meta/{info,episodes,tasks,stats}.json[l]`.
A v3.0 *packed* release must be converted first.

- Converter: `ISAAC_SHARED/scripts/convert_lerobot_v30_to_v20.py --src --dst`
  (re-encodes one mp4 per episode).
- **Note:** the sbatch wrapper referenced in earlier notes as
  `WS/vlm_gate/run_scripts/data/sbatch_dexjoco_convert_v30_to_v20.sh` does **not**
  exist on disk today; `WS/vlm_gate/run_scripts/` has only `cache/ eval/ label/ train/`.
- `codebase_version` is **not** checked anywhere in `gr00t/` (verified by grep), so a
  `v2.1` dataset whose `data_path`/`video_path` already use `episode_chunk` loads
  as-is. robocasa365 is exactly that case — no conversion needed.

### 2. `meta/modality.json` with per-slice keys matching the DataConfig

The monolithic `modality.json` some releases ship will not load. Requirements,
all verified against `gr00t/data/schema.py` and `gr00t/data/dataset.py`:

- `state` / `action` entries: `start`, `end`, and optionally `original_key`,
  `rotation_type`, `absolute`, `dtype`, `range`.
- `rotation_type` is **load-bearing**: `gr00t/data/transform/state_action.py:421`
  reads it as the *source* representation for a `rotation_6d` target. Omitting it on
  a quaternion slice that the DataConfig lists in `state_target_rotations` breaks the
  transform silently or loudly.
- `dtype: int64` + `range: [0,1]` are what make `"binary"` normalization work for
  gripper/control-mode dims.
- `video` entries: the key is the *DataConfig* name (`left_view`), `original_key` is
  the on-disk feature name — and `original_key` is substituted into `video_path`
  as `video_key`, so it must equal the directory name under `videos/chunk-*/`.
- `annotation` entries also honour `original_key` (`dataset.py:775`), so a dataset
  shipping `annotation.human.task_description` can be exposed to a DataConfig that
  wants `annotation.human.action.task_description` without touching the parquet.

Worked examples: `WS/vlm_gate/n17/templates/dexjoco_single_arm_modality.json`,
`WS/vlm_gate/n17/templates/robocasa365_single_panda_gripper_modality.json` (new, see below).

### 3. A DataConfig in `gr00t/experiment/data_config.py`

Templates: `DexJoCoSingleArmMultiHorizonDataConfig` (~line 1769),
`LiberoDataConfig` / `LiberoMultiHorizonDataConfig` (~lines 1115 / 1220),
`SinglePandaGripperDataConfig` (~line 759, this is the RoboCasa one),
`FourierGr1ArmsWaistDataConfig` (~line 808, this is the GR-1 one).
Register the new class in `DATA_CONFIG_MAP` at the bottom of the file.

### 4. Embodiment tag

`gr00t/data/embodiment_tags.py`. `EMBODIMENT_TAG_MAPPING` carries the action-expert
projector index: `new_embodiment: 31`, `libero: 31`, `gr1: 24`, `oxe_droid: 17`,
`agibot_genie1: 26`. Reusing `new_embodiment` avoids adding an index.

### 5. A training job script

`ISAAC_SHARED/run_scripts/train/{robocasa,libero,gr1_tabletop,robocasa365}/*.sh`.
`gr00t_finetune.py --dataset-path` takes a list; more than one path switches it to
`LeRobotMixtureDataset` automatically.

### 6. Eval harness / server

All evals are a **two-process ZMQ split**: policy server in the GR00T env,
simulator client in a simulator-specific env.

- RoboCasa (PandaOmron): `ISAAC_SHARED/scripts/robocasa_service.py --server|--client`,
  env construction in `gr00t/eval/wrappers/robocasa_wrapper.py::load_robocasa_gym_env`,
  obs→gr00t key map in `RoboCasaWrapper._robocasa_keys_to_gr00t_keys`.
- GR-1 tabletop: `ISAAC_SHARED/scripts/inference_service.py --server` +
  `scripts/simulation_service.py --client`, env ids from
  `robocasa-gr1-tabletop-tasks/robocasa/utils/gym_utils/gymnasium_groot.py`.
- LIBERO: `ISAAC_SHARED/gr00t/eval/libero/eval_taskwise_gr00t*.py`.

---

## Benchmark A — robocasa365

**Data on disk (verified):**
`/rlwrld-foundry-artifacts/.storage1/sjw_dataset/dataset/robocasa365_v101/v1.0/target/composite`
— 16 *composite* task dirs, each `<Task>/<YYYYMMDD>/lerobot/`. 23 GB, 8,077 episodes,
6,002,265 frames, 20 fps, LeRobot `v2.1`. `PandaOmron`, 3× 256×256 cameras
(`robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`),
`observation.state` 16-D, `action` 12-D.

Per-task counts (episodes / frames): DeliverStraw 504/433,307 · GetToastedBread
506/654,840 · KettleBoiling 501/228,349 · LoadDishwasher 501/369,430 ·
PackIdenticalLunches 501/719,964 · PreSoakPan 501/395,501 · PrepareCoffee 514/279,534 ·
RinseSinkBasin 509/211,036 · ScrubCuttingBoard 504/228,864 · SearingMeat 501/439,466 ·
SetUpCuttingStation 504/336,025 · StackBowlsCabinet 515/175,620 · SteamInMicrowave
511/487,012 · StirVegetables 501/409,202 · StoreLeftoversInBowl 503/390,829 ·
WashLettuce 501/243,286.

**Touch-point status**

| # | Status |
|---|---|
| 1 layout | **DONE.** v2.1 but chunk-formatted; loads without conversion (smoke-tested). |
| 2 modality.json | **FIXED.** Shipped file is per-slice and its state/action offsets are already exactly `SinglePandaGripperDataConfig`'s — but its video keys are `robot0_*`, its annotation key is `human.task_description`, and it drops every `rotation_type`/`dtype`/`range`. Corrected file written (see below). |
| 3 DataConfig | **DONE, reuse `single_panda_gripper`.** state slices base_position 0:3, base_rotation 3:7, eef_pos_rel 7:10, eef_rot_rel 10:14, gripper_qpos 14:16; action base_motion 0:4, control_mode 4:5, eef_pos 5:8, eef_rot 8:11, gripper_close 11:12 — an exact match. Note the 16-D state lacks the absolute-eef / joint / qvel slices of the incumbent 53-D `kimtaey/robocasa_mg_gr00t_300`; `single_panda_gripper` never asks for those. |
| 4 embodiment tag | **DONE, reuse `new_embodiment`** (what the incumbent robocasa runs use). The dataset's own `meta/embodiment.json` says `robocasa_panda_omron`, which is not a GR00T tag — ignore it. |
| 5 train script | Templates exist at `ISAAC_SHARED/run_scripts/train/robocasa365/finetune_robocasa365_{baseline,mh_m8_econsist}.sh` but they still point at the 24-task `kimtaey/robocasa_mg_gr00t_300` subset. Repoint `--dataset-path` at the 16 composite dirs (each with the corrected `modality.json` in place). |
| 6 eval | **BLOCKED — simulator upgrade required.** See below. |

**Simulator gap (verified, and it corrects an earlier note):**
`ISAAC_SHARED/docs/robocasa365_setup.md` claims the cloned
`multigpu_workspace/robocasa` "already contains the v1.0/365 task surface". **It does
not.** That checkout is `robocasa 0.2.0` (`robocasa/__init__.py:331`) on tag `v0.2`;
of the 16 env classes only 5 register (`KettleBoiling`, `PreSoakPan`, `PrepareCoffee`,
`SearingMeat`, `SteamInMicrowave`); 11 are missing. Verified in `quant_gate_eval` via
`robosuite.environments.base.REGISTERED_ENVS`.

The dataset states its own provenance in
`.../lerobot/extras/dataset_meta.json`: `env_version 0.5.1`, `robosuite_version 1.5.2`,
`mujoco_version 3.3.1`, plus env kwargs `obj_instance_split="target"`, `clutter_mode=1`,
10 fixed `layout_and_style_ids` pairs, `HYBRID_MOBILE_BASE` controller.

The needed package is the upstream branch **`robocasa/robocasa @ robocasa365_release`**
(= tag `v1.0`, `__version__ = "1.0.0"`). I shallow-cloned it to
`WS/external/robocasa365` (73 MB, commit `0c81ff9`) and confirmed **all 16 env classes
are present**, under `robocasa/environments/kitchen/composite/<activity>/` (a new
directory name; 0.2.0 used `multi_stage/`).

Its hard pins (`robocasa/__init__.py:1000-1027`, `setup.py`):
`mujoco==3.3.1`, `numpy==2.2.5`, `numba==0.61.2`, `scipy==1.15.3`, `robosuite>=1.5.2`,
`lerobot==0.3.3`. The installed stack is `mujoco 3.2.6` + `numpy 1.23.5`, and
`robocasa 0.2.0` *asserts* `mujoco == 3.2.6` — so the two cannot coexist in one env.

**Can it go in a separate env without disturbing the existing ones? Yes, with care.**
`robocasa` and `robosuite` are installed **editable** into both `quant_gate_eval` and
`robocasa_gr00t`, pointing at the shared trees
`multigpu_workspace/{robocasa,robosuite}`. Therefore:
- Do **not** `pip install -e` anything into those trees (that writes `*.egg-info` into
  shared source).
- Create a fresh env (e.g. `robocasa365_eval`), `pip install mujoco==3.3.1 numpy==2.2.5
  numba==0.61.2 scipy==1.15.3 robosuite==1.5.2` (all on PyPI; PyPI is reachable), and
  put `WS/external/robocasa365` on `PYTHONPATH` rather than installing it. Nothing
  shared is touched.
- Assets: robocasa365 adds a new source, `nvidia/PhysicalAI-Kitchen-Assets`
  (`fixtures_lightwheel.zip`, `objects_lightwheel.zip`) on top of the
  `robocasa/robocasa-assets` textures/objaverse/aigen packs, downloaded into
  `<clone>/robocasa/models/assets/` by `robocasa/scripts/download_kitchen_assets.py`.
  The 0.2.0 asset tree is 8.0 GB, so budget **≥ 8 GB and probably more**; the exact
  size could not be measured — the HuggingFace API is not reachable from this host
  (the GitHub API and PyPI are).
- The remaining risk is the **client side of the ZMQ split**: `robocasa_service.py
  --client` imports `gr00t.model.policy`, and `ISAAC`'s `pyproject.toml` pins
  `numpy>=1.23.5,<2.0.0`. Installing gr00t `--no-deps` into the numpy-2.2.5 env and
  verifying the import is the concrete next step; whether gr00t's code is numpy-2 clean
  has **not** been established.
- `load_robocasa_gym_env` does not expose `clutter_mode`; add it (and keep
  `obj_instance_split="target"`) to reproduce the demo distribution.

**Single biggest blocker:** the simulator. Everything else (data, modality, DataConfig,
embodiment tag) is ready; eval needs robocasa 1.0.0 in a numpy-2.2.5 / mujoco-3.3.1 env
plus its lightwheel asset download.

---

## Benchmark B — RoboCasa GR-1

**Candidates measured (all under `/rlwrld-foundry-artifacts/.storage1/sjw_dataset/dataset/`):**

| Dataset | Size | Episodes | Frames | fps | Version |
|---|---:|---:|---:|---:|---|
| `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*` (24 tasks) | 40 GB | 24,000 | 6,020,058 | 20 | v2.0 |
| `robocasa_gr1_tabletop/sim_300demos` | 12 GB | 7,200 | 1,804,329 | 20 | v2.0 |
| `robocasa_gr1_tabletop/sim_100demos` | 4.5 GB | 2,400 | 600,508 | 20 | v2.0 |
| `gr1_robocurate_dataset_neurips/*` (8 variants) | 5.5 GB total | 692–6,705 | 64k–624k | 16 | **no `codebase_version`** |
| `GR00T-GR1/PhysicalAI-Robotics-GR00T-{GR1,Eval,Eval10}` | 15 GB | — | — | — | raw `.mp4`, not LeRobot |

**Action/state 44-D layout: yes, documented, and it is in the data.** Every GR-1
dataset above ships a per-slice `meta/modality.json` with the same split, and it is
also written up in `ISAAC_SHARED/docs/gr1_tabletop_setup_log.md` §4:

| slice | start | end | dims |
|---|---:|---:|---:|
| left_arm | 0 | 7 | 7 |
| left_hand | 7 | 13 | 6 |
| left_leg | 13 | 19 | 6 |
| neck | 19 | 22 | 3 |
| right_arm | 22 | 29 | 7 |
| right_hand | 29 | 35 | 6 |
| right_leg | 35 | 41 | 6 |
| waist | 41 | 44 | 3 |

Identical for `observation.state` and `action`. Legs are all-zero in `stats.json`
(disabled). `FourierGr1ArmsWaistDataConfig` selects arms+hands+waist = 29 dims; all
continuous, so no `--discrete-action-dims`.

**Touch-point status**

| # | Status |
|---|---|
| 1 layout | **DONE** for the three v2.0 sim datasets. The robocurate variants have **no `codebase_version`** and their `info.json` lacks `timestamp`/`episode_index`/`index`/`next.*`; `robot_type` is `"dream"` (DreamGen synthetic rollouts, 16 fps). Loadability not established. |
| 2 modality.json | **ALREADY PRESENT AND CORRECT.** Nothing to write. |
| 3 DataConfig | **ALREADY PRESENT.** `fourier_gr1_arms_only`, `fourier_gr1_arms_waist`, `fourier_gr1_arms_waist_multi_horizon`, `fourier_gr1_full_upper_body` in `DATA_CONFIG_MAP`. `annotation.human.coarse_action` (the language key of `fourier_gr1_arms_waist`) is the key the datasets ship. |
| 4 embodiment tag | **ALREADY PRESENT.** `EmbodimentTag.GR1 = "gr1"`, projector index 24. |
| 5 train script | **ALREADY PRESENT.** `ISAAC_SHARED/run_scripts/train/gr1_tabletop/finetune_gr1_{baseline,mh_m8_econsist}.sh` (2 GPUs, batch 32, 60k steps, `--embodiment-tag gr1`). Never submitted. |
| 6 eval | **PRESENT BUT CURRENTLY BROKEN** — one version assert. See below. |

**Eval state (verified):** `multigpu_workspace/robocasa-gr1-tabletop-tasks` is cloned,
its 8.5 GB tabletop assets are downloaded, and all 24 `gr1_unified/*_Env` gymnasium ids
register. Procedure is written up in `ISAAC_SHARED/docs/gr1_tabletop_eval_setup.md`
(server = `scripts/inference_service.py`, client = `scripts/simulation_service.py`).

But the client env `gr1_alinvla` no longer imports:

```
robocasa-gr1-tabletop-tasks/robocasa/__init__.py:124
AssertionError: robosuite version must be 1.5.{0,1}
```

`gr1_alinvla` resolves `robosuite` through an editable pointer to the **shared**
`multigpu_workspace/robosuite`, which is at **1.5.2** (needed by `robocasa 0.2.0`).
Upgrading that tree for the PandaOmron benchmark broke the GR-1 tabletop client.
When `robosuite.__version__` is monkey-patched to `"1.5.1"` the import succeeds and all
24 env ids register — i.e. this is only an assert, but do **not** patch it in place.

Fix without touching anything shared: new env `gr1_eval` with
`pip install robosuite==1.5.1` (on PyPI, confirmed) and
`PYTHONPATH=$HOME/multigpu_workspace/robocasa-gr1-tabletop-tasks` so the tabletop
`robocasa` package is imported, not installed. Whether robocasa's tabletop code is
truly 1.5.2-incompatible, or the assert is merely stale, was **not** determined.

**Recommendation: build on `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*`.**
Reasons: (a) its 24 task directories correspond 1:1 to the 24 `gr1_unified/*_Env`
gymnasium ids the eval client can actually run, so train and eval task sets match —
`robocasa_gr1_tabletop` has 187 tasks in one flat dataset with no matching env list,
and the robocurate sets are synthetic with no environment at all; (b) it is the dataset
the existing `run_scripts/train/gr1_tabletop/*.sh` already point at; (c) 6.0 M frames,
1000 episodes/task, v2.0, correct modality.json. Use `sim_100demos` only if 40 GB /
6 M frames is too large for a first run — but accept that it is not directly evaluable.
`GR00T-GR1/*` is a raw-video corpus (`gr1/1.mp4`, …), not trainable as-is.

**Single biggest blocker:** the `robosuite` 1.5.1-vs-1.5.2 collision between the GR-1
tabletop client and the shared robosuite checkout. It is a one-env fix, not a
simulator port.

---

## What was written during this survey

- `WS/vlm_gate/n17/templates/robocasa365_single_panda_gripper_modality.json` — the
  corrected robocasa365 `modality.json`. Derived by taking the *shipped* file's
  state/action offsets (verified against the parquet: `observation.state` 16-D,
  `action` 12-D), renaming the three video keys to the `SinglePandaGripperDataConfig`
  names with `original_key` pointing at the real `videos/chunk-*/` directory names
  (mapping taken from `RoboCasaWrapper._robocasa_keys_to_gr00t_keys`:
  agentview_left→left_view, agentview_right→right_view, eye_in_hand→wrist_view),
  exposing `annotation.human.task_description` as
  `human.action.task_description`, and restoring the `rotation_type` / `absolute` /
  `dtype` / `range` fields from the incumbent
  `kimtaey/robocasa_mg_gr00t_300/meta/modality.json`. **Nothing was guessed.**
- `WS/vlm_gate/n17/overlays/robocasa365_DeliverStraw/` — a symlink farm
  (`data`, `videos` and all of `meta/` symlinked to the read-only dataset, with only
  `meta/modality.json` a real file) used to smoke-test the above without writing into
  the dataset. Reusable pattern for the other 15 tasks.
- `WS/external/robocasa365` — shallow clone of `robocasa @ robocasa365_release`.

**Smoke test result (CPU, `quant_gate` env, `HF_HUB_OFFLINE=1`):** loading the overlay
with `LeRobotSingleDataset` and `single_panda_gripper`'s modality keys yields
`len = 433,307` and a correct sample — `video.{left,right,wrist}_view (1,256,256,3)`,
`state.*` (3/4/2/3/4), `action.*` 16-step, language = the task string, and
`rotation_type = QUATERNION` on `base_rotation` / `end_effector_rotation_relative`.
The **shipped** `modality.json` fails the same load with
`video key left_view not found in metadata` — confirming the fix is necessary.
The GR-1 `gr1_unified.PnPBottleToCabinetClose_*` dataset loads unmodified with the
`fourier_gr1_arms_waist` keys (`len = 352,726`).

No GPU jobs were submitted, no dataset directory or shared conda env was modified.
