# GR-1 Tabletop Setup Log

Working notes captured while setting up the GR00T-N1.5 fine-tune + sim eval
pipeline on the 24 RoboCasa GR-1 tabletop tasks.

## 1. Reference repo and tasks

`robocasa-gr1-tabletop-tasks/README.md` (the upstream NVIDIA fork) lists 24
`gr1_unified/...GR1ArmsAndWaistFourierHands_Env` envs and prescribes:

- **Server**: `python3 scripts/inference_service.py --server --model_path
  <CKPT> --data_config fourier_gr1_arms_waist`
- **Client**: `python3 scripts/simulation_service.py --client --env_name
  <ENV> --max_episode_steps 720 --n_envs 5 --n_episodes 10`
- Posttraining example uses 8 H100s, batch 60, lr 3e-5, max-steps 30k,
  save-steps 5k, `--data-config fourier_gr1_arms_waist --embodiment_tag gr1
  --tune-visual`.

Existing GR1 references in our `Isaac-GR00T`:
- `gr00t/eval/simulation.py:257` and `gr00t/eval/robocasa_simulation.py:289`
  default to `robocasa_gr1_arms_only_fourier_hands/...` envs (different task
  family from tabletop; not used here).

## 2. Conda env audit

`/sjw_alinlab2/home/hojin2/miniconda3/envs/`:

```
alin_multi_meta  alin_vla       alin_vla_q35   cosmos_robocasa
gr00t            gr1_alinvla    libero         libero310
robocasa         robocasa_alinvla  robocasa_alinvla_convnext
robocasa_gr00t   robocasa_vam
```

`gr1_alinvla` already contains editable installs of `gr00t==1.1.0`,
`robocasa==0.2.0`, `robosuite==1.5.2` (per
`lib/python3.10/site-packages/__editable__*.pth`). It is suitable as the
**simulation client** env. It is **not** suitable as the policy server: its
`transformers` is too old for our Eagle/Qwen processor (an import test from
`gr00t.experiment.data_config` raised
`ImportError: cannot import name 'VideoInput' from 'transformers.image_utils'`).

The `gr00t` env (used for our existing libero / robocasa training) loads our
local `gr00t` package successfully and is the right env for both training and
the inference server.

So the layout for eval is: server=`gr00t`, client=`gr1_alinvla` — same
two-env pattern we use for libero and robocasa.

## 3. Dataset on disk

`huggingface-cli download` is **not needed** — the dataset already lives
under
`/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/`.

- 24 sub-directories, one per task, each named
  `gr1_unified.<TaskName>_GR1ArmsAndWaistFourierHands_1000`.
- Each contains `data/chunk-000/episode_{000000..000999}.parquet` (1000
  episodes, 1 chunk), `videos/...`, and `meta/{info,modality,episodes,tasks,
  stats,relative_stats}.json[l]`.
- Total size: **79 GB** (each task ~1.9 GB).

`PhysicalAI-Robotics-GR00T-Teleop-Sim` (the dataset name from the upstream
README) and `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` (the on-disk name)
appear to be the same content for the GR-1 ArmsAndWaistFourierHands subset
— matching robot_type, fps, action shape, and per-task 1000 episodes. **Did
not re-download.**

If a re-download is ever needed:

```bash
huggingface-cli download --repo-type dataset \
    nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim \
    --local-dir /sjw_alinlab2/home/hojin2/multigpu_workspace/data/gr1_teleop_sim/
```

## 4. Dataset structure

Per `meta/info.json`:
- `robot_type`: `GR1ArmsAndWaistFourierHands`
- `fps`: 20.0
- `total_episodes`: 1000 (per task), `total_frames` ≈ 266k (varies),
  `total_chunks`: 1
- `observation.images.ego_view`: 256x256x3 video, h264
- `observation.state`: shape (44,)
- `action`: shape (44,)
- `annotation.human.coarse_action`: int64 (used as language key)

Per `meta/modality.json` (44-dim split):

| Group | start | end | dims |
| --- | ---: | ---: | ---: |
| left_arm  | 0  | 7  | 7 |
| left_hand | 7  | 13 | 6 |
| left_leg  | 13 | 19 | 6 |
| neck      | 19 | 22 | 3 |
| right_arm | 22 | 29 | 7 |
| right_hand| 29 | 35 | 6 |
| right_leg | 35 | 41 | 6 |
| waist     | 41 | 44 | 3 |

The `FourierGr1ArmsWaistDataConfig` in `gr00t/experiment/data_config.py`
selects exactly arms + hands + waist (29 dims), dropping legs (always 0 in
stats anyway) and neck. This matches the upstream eval data config.

`stats.json` action min/max for one task confirms:
- arm joints in radian-ish ranges, e.g. left_arm[0] ∈ [-1.42, +0.40]
- hand joints in [-1.5, +1.5] and [-3, +3] / [+3, +3] (note: index 12 is
  constant at +3.0 — likely a fixed pinky DOF; min_max normalization handles
  it, but it is a no-op axis)
- left_leg / right_leg dims all (0.0, 0.0) — confirming legs are disabled
- waist dims small (~0.3 magnitude)

All 29 selected action dims are continuous (no binary gripper) — so for the
`mh_m8_econsist` variant we **omit `--discrete-action-dims`** (the robocasa
script set 6,11 for `gripper_close` and `control_mode`; libero used 6 for
`gripper_close` only; GR-1 has neither).

## 5. Existing data configs

`gr00t/experiment/data_config.py` already registers:
- `fourier_gr1_arms_only` (FourierGr1ArmsOnlyDataConfig) — sin/cos state, min_max action, ego_view.
- `fourier_gr1_arms_waist` (FourierGr1ArmsWaistDataConfig) — same idea + waist; `annotation.human.coarse_action` as language. **Used as-is for the baseline run.**
- `fourier_gr1_full_upper_body` (FourierGr1FullUpperBodyDataConfig) — also includes neck.

For multi-horizon training I added a new sibling:
- `fourier_gr1_arms_waist_multi_horizon` (FourierGr1ArmsWaistMultiHorizonDataConfig)
  - Inherits `FourierGr1ArmsWaistDataConfig` (same modality keys + sin/cos state).
  - Sets `action_indices = list(range(64))` (extended fetch).
  - Overrides `transform()` to pass `action_horizon=16` and
    `extended_action_horizon=64` to `GR00TTransform` — matching the pattern of
    `LiberoMultiHorizonDataConfig` and `SinglePandaGripperMultiHorizonDataConfig`.
  - Registered in `DATA_CONFIG_MAP` as
    `"fourier_gr1_arms_waist_multi_horizon"`.

Verified the new config loads: `python -c "from gr00t.experiment.data_config
import DATA_CONFIG_MAP; cfg =
DATA_CONFIG_MAP['fourier_gr1_arms_waist_multi_horizon']; ..."` returned
`action_indices len: 64`, `action_horizon_output: 16`, and the correct
modality keys.

## 6. Sbatch scripts created

In `Isaac-GR00T/run_scripts/train/gr1_tabletop/`:

- `finetune_gr1_baseline.sh` — `--data-config fourier_gr1_arms_waist`,
  no multi-horizon flags, mirrors `robocasa/finetune_gr00t_n1_5_baseline.sh`.
- `finetune_gr1_mh_m8_econsist.sh` — `--data-config
  fourier_gr1_arms_waist_multi_horizon`, full mh+m8+econsist stack mirroring
  `libero/finetune_gr00t_n1_5_libero_mh_m8_econsist.sh`. **No
  `--discrete-action-dims`** flag (justification above).

Both scripts:
- Use `--num-gpus 2 --batch-size 32` (global batch 64).
- `--max-steps 60000 --save-steps 10000` (matches our standard cadence).
- `--embodiment-tag gr1`, `--base-model-path nvidia/GR00T-N1.5-3B`.
- Pass all 24 task dataset paths in a `DATASETS=( ... )` array, which
  `gr00t_finetune.py` accepts as `--dataset-path` (List[str]) — it uses
  `LeRobotMixtureDataset` automatically when more than one path is given.
- `WANDB_PROJECT=GR00T-gr1-tabletop`.
- Output: `ckpt/gr1_tabletop/groot/groot_n1_5_bs64_{baseline,mh_m8_econsist}/`.

Both are `chmod +x`'d. **Not submitted** per instructions.

## 7. Eval doc

`Isaac-GR00T/docs/gr1_tabletop_eval_setup.md` describes the two-env
client-server pattern using the upstream `inference_service.py` /
`simulation_service.py` entrypoints, including a 24-task sweep snippet.

## 8. Blockers / caveats

- **`gr1_alinvla` env has stale `gr00t==1.1.0`**: it cannot import our local
  `gr00t.experiment.data_config` because `transformers` is older than the
  Eagle processor expects. This is fine for the **client** (it only imports
  `gr00t.eval.simulation` from the editable old gr00t package, which talks
  to the server over ZMQ), but it means we **cannot use this env for the
  policy server**. Use `gr00t` for the server instead. Documented in the eval
  setup doc.
- The `nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim` repo on HF has
  not been verified to be byte-identical to the on-disk
  `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` mirror. The robot_type, fps,
  action layout, modality keys, and per-task episode counts all match the
  spec in the upstream README, so I treat them as equivalent. If the user
  wants a clean download, the `huggingface-cli` command is documented above.
- The merged-8 head is present in our trained checkpoint but the upstream
  `simulation_service.py --client` does not expose `--head m8` selection. To
  evaluate the m8 chunk you would need a small client wrapper (mirroring our
  libero / robocasa eval clients) that calls the policy with `head='m8'`.
  Not in scope for this setup task.
- Did not run a smoke training step (would download `nvidia/GR00T-N1.5-3B`
  weights and chew GPU). The data config + dataset paths + flags were
  validated by importing `DATA_CONFIG_MAP` only.
