# SimplerEnv Setup Log

Notes from the agent session that wired up SimplerEnv (Bridge / WidowX +
Fractal / Google Robot) for GR00T-N1.5 fine-tuning + evaluation.

## What was completed

### 1. SimplerEnv repo

- Cloned `https://github.com/youliangtan/SimplerEnv` to
  `/sjw_alinlab2/home/hojin2/multigpu_workspace/SimplerEnv`.
- Submodule `ManiSkill2_real2sim` initialized
  (commit `c2a9e87c186300b694da6f2497dd68d2c347a4b7`).

### 2. Conda env

- Created fresh `simpler_env` conda env (Python 3.10).
- Installed `numpy==1.24.4` (pinned per upstream README).
- BLOCKED: the two `pip install -e .` calls (`ManiSkill2_real2sim` and
  the SimplerEnv top-level package) install Python from cloned external
  repos and were denied by the agent sandbox. Documented the manual steps
  for the user in `SimplerEnv/SETUP_NOTES.md`.

### 3. Data configs in Isaac-GR00T

- Copied `examples/SimplerEnv/{custom_data_config.py, bridge_modality.json,
  fractal_modality.json, README.md}` from
  `Isaac-GR00T-AlinVLA/examples/SimplerEnv/` to
  `Isaac-GR00T/examples/SimplerEnv/`.
- Added four new entries to `gr00t/experiment/data_config.py` so they
  resolve through the `--data-config` CLI Literal (the upstream
  `module:ClassName` syntax is NOT supported by `gr00t_finetune.py` /
  `inference_service.py` -- both use `Literal[tuple(DATA_CONFIG_MAP.keys())]`):
  - `simplerenv_bridge` (single-horizon, action_indices = range(16))
  - `simplerenv_bridge_multi_horizon` (action_indices = range(64),
    extended_action_horizon=64, action_horizon=16)
  - `simplerenv_fractal`
  - `simplerenv_fractal_multi_horizon`
- Verified import + `modality_config()` + `transform()` work via
  `python -c "from gr00t.experiment.data_config import DATA_CONFIG_MAP; ..."`
  in the `gr00t` env.

The legacy `bridge` key (uses IPEC-COMMUNITY/bridge eef_position-style
state/action keys) was kept untouched for backwards compatibility.

### 4. Datasets

- Started background HF downloads to
  `/sjw_alinlab2/home/hojin2/multigpu_workspace/data/simplerenv/`:
  - `IPEC-COMMUNITY/bridge_orig_lerobot` -> `bridge_orig_lerobot/`
  - `IPEC-COMMUNITY/fractal20220817_data_lerobot` -> `fractal20220817_data_lerobot/`
- These are LARGE (Bridge ~140 GB, Fractal ~150 GB on HF). Both download
  jobs were still running at end of session (each ~9 MB after ~5 min).
  Available disk on `/sjw_alinlab2`: 1.5 TB / 15 TB total (90% full --
  monitor).
- Background task IDs: bridge=`bgpg0w2u2`, fractal=`b2dn7eqx3`. Output
  logs in `/tmp/claude-*/tasks/`. They'll keep streaming files into the
  workspace dataset dirs in alphabetical order (`data/` then `meta/`).
- Also discovered the RLDS / TFDS versions at
  `/open-x-embodiment/bridge_orig` (124 GB) and
  `/open-x-embodiment/fractal20220817_data` (98 GB) -- those are NOT
  LeRobot format and would need conversion to use with our data configs.
  Sticking with the HF LeRobot download path is simpler.

ACTION REQUIRED FOR USER: After the downloads finish, copy the modality
configs in:

```bash
cp /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/examples/SimplerEnv/bridge_modality.json \
   /sjw_alinlab2/home/hojin2/multigpu_workspace/data/simplerenv/bridge_orig_lerobot/meta/modality.json

cp /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/examples/SimplerEnv/fractal_modality.json \
   /sjw_alinlab2/home/hojin2/multigpu_workspace/data/simplerenv/fractal20220817_data_lerobot/meta/modality.json
```

(Both downloads were still in `data/`; the `meta/` dir comes later in the
HF download order -- you may need to wait or re-run the
`huggingface-cli download` once to confirm completion.)

### 5. Sbatch scripts

Created in `Isaac-GR00T/run_scripts/train/simplerenv/`:

| Script | Data config | Notes |
| --- | --- | --- |
| `finetune_bridge_baseline.sh` | `simplerenv_bridge` | No multi-horizon; pattern matches `robocasa/finetune_gr00t_n1_5_baseline.sh` |
| `finetune_bridge_mh_m8_econsist.sh` | `simplerenv_bridge_multi_horizon` | Adds `--use-multi-horizon-loss --multi-horizon-factors 2 4 --discrete-action-dims 6 --use-merged-8-head --use-ensemble-consistency-loss` (mirrors `robocasa/finetune_gr00t_n1_5_mh_m8_econsist.sh`) |
| `finetune_fractal_baseline.sh` | `simplerenv_fractal` | |
| `finetune_fractal_mh_m8_econsist.sh` | `simplerenv_fractal_multi_horizon` | |

All four use:
- 2 GPU, batch_size 32 (effective 64)
- max-steps 60000, save-steps 10000
- `--embodiment-tag new_embodiment`
- `--base-model-path nvidia/GR00T-N1.5-3B`
- `--report-to wandb` with `WANDB_PROJECT=GR00T-simplerenv`

`--discrete-action-dims 6` reflects the gripper index in the concatenated
action vector (Bridge / Fractal both have layout
`[x, y, z, roll, pitch, yaw, gripper]`).

### 6. Eval setup notes

Wrote `docs/simpler_env_eval_setup.md` with:
- Per-checkpoint inference-server commands (Bridge baseline, Bridge mh_m8,
  Fractal baseline, Fractal mh_m8) including `--head` and
  `--discrete-action-dims` choices.
- `eval_simpler.py` client commands per task (full task name list copied
  from upstream README).
- Variant-aggregation eval script names.
- Common args (`--eval_count`, `--episode_length`, `--action_horizon`,
  `--headless`, `--output_video_dir`).
- Sanity-test flow with the upstream
  `youliangtan/gr00t-n1.5-bridge-posttrain` HF checkpoint.

## Files created / modified

| Path | Type |
| --- | --- |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/SimplerEnv/` | Cloned repo |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/SimplerEnv/SETUP_NOTES.md` | New -- manual install steps |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/examples/SimplerEnv/custom_data_config.py` | Copied from AlinVLA fork |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/examples/SimplerEnv/bridge_modality.json` | Copied |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/examples/SimplerEnv/fractal_modality.json` | Copied |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/examples/SimplerEnv/README.md` | Copied |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/gr00t/experiment/data_config.py` | Modified -- added 4 SimplerEnv configs + map entries |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/run_scripts/train/simplerenv/finetune_bridge_baseline.sh` | New |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/run_scripts/train/simplerenv/finetune_bridge_mh_m8_econsist.sh` | New |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/run_scripts/train/simplerenv/finetune_fractal_baseline.sh` | New |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/run_scripts/train/simplerenv/finetune_fractal_mh_m8_econsist.sh` | New |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/docs/simpler_env_eval_setup.md` | New |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T/docs/simpler_env_setup_log.md` | New (this file) |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/data/simplerenv/bridge_orig_lerobot/` | Created -- HF download in progress |
| `/sjw_alinlab2/home/hojin2/multigpu_workspace/data/simplerenv/fractal20220817_data_lerobot/` | Created -- HF download in progress |

## Blockers / TODO

1. **`pip install -e .` for ManiSkill2_real2sim and SimplerEnv** -- sandbox
   denied untrusted-code execution. User must run the two install commands
   in `SimplerEnv/SETUP_NOTES.md` manually inside the `simpler_env` conda
   env.
2. **Dataset downloads** are still running in background (~9 MB each at
   end of session, target ~140-150 GB each). Wait for completion before
   submitting sbatch jobs. Monitor:
   `du -sh /sjw_alinlab2/home/hojin2/multigpu_workspace/data/simplerenv/*`.
3. **Modality.json copy** can only happen after `meta/` dir is downloaded
   (HF order: `data/` -> `meta/`). Snippet provided above.
4. **No sbatch jobs were submitted** (per instructions). Submit when
   ready:
   ```
   cd /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T
   sbatch run_scripts/train/simplerenv/finetune_bridge_baseline.sh
   sbatch run_scripts/train/simplerenv/finetune_bridge_mh_m8_econsist.sh
   sbatch run_scripts/train/simplerenv/finetune_fractal_baseline.sh
   sbatch run_scripts/train/simplerenv/finetune_fractal_mh_m8_econsist.sh
   ```
5. **Disk pressure**: `/sjw_alinlab2` is 90% full (1.5 TB free). Bridge +
   Fractal LeRobot together will eat ~280-300 GB. Verify before / during
   download.
