# GR-1 Tabletop Simulation Eval Setup

This doc covers how to run closed-loop simulation evaluation on the 24 RoboCasa
GR-1 tabletop tasks for a GR00T-N1.5 checkpoint trained with one of the
`run_scripts/train/gr1_tabletop/finetune_gr1_*.sh` scripts.

The setup follows the pattern of the existing robocasa / libero evals
(client-server over ZMQ), but uses the upstream
`scripts/inference_service.py` (server) and `scripts/simulation_service.py`
(client) entrypoints from Isaac-GR00T, which already know about the
`gr1_unified/...FourierHands_Env` environments.

## Conda envs

Two envs are required because the simulator and the policy server have
incompatible dependency versions:

| Env | Purpose | Notes |
| --- | --- | --- |
| `gr00t` | Inference server (policy) | The same env we use for our multi-horizon / m8 / econsist training. Has the up-to-date `Isaac-GR00T` checkout from this repo installed editable. |
| `gr1_alinvla` | Simulation client | Already has editable installs of `gr00t==1.1.0`, `robocasa==0.2.0`, `robosuite==1.5.2`, plus the `robocasa-gr1-tabletop-tasks` package and downloaded tabletop assets. **Do not pip-install our local Isaac-GR00T into this env** — its older `transformers` is not compatible with the Eagle/Qwen processors we use. |

If `gr1_alinvla` is missing the tabletop assets you can re-run them per the
upstream README:

```bash
conda activate gr1_alinvla
cd $HOME/multigpu_workspace/robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y
```

## Step 1: start the inference server (env: `gr00t`)

```bash
conda activate gr00t
cd $HOME/multigpu_workspace/Isaac-GR00T

CKPT=ckpt/gr1_tabletop/groot/groot_n1_5_bs64_mh_m8_econsist/checkpoint-60000  # or any saved step

python scripts/inference_service.py --server \
    --model_path "$CKPT" \
    --embodiment_tag gr1 \
    --data_config fourier_gr1_arms_waist \
    --port 5555
```

Notes:
- The training-time data config can be either `fourier_gr1_arms_waist` (baseline)
  or `fourier_gr1_arms_waist_multi_horizon` (our multi-horizon variant). At
  inference we always use the **non**-multi-horizon variant
  (`fourier_gr1_arms_waist`) because we only need the first 16-step output
  chunk; the action_keys / state_keys / video_keys / language_keys are
  identical between the two.
- `--embodiment_tag gr1` matches the GR1 entry in
  `gr00t/data/embodiment_tags.py` and is what the model was trained with.
- For the `mh_m8_econsist` variant, if you want to evaluate the merged-8
  chunked head, pass `--head m8` to whichever client wrapper supports it; the
  upstream `simulation_service.py --client` only consumes the default 16-step
  chunk via `get_action`.

## Step 2: run the simulation client (env: `gr1_alinvla`)

In a second shell:

```bash
conda activate gr1_alinvla
cd $HOME/multigpu_workspace/Isaac-GR00T

# Pick one of the 24 task envs (full list in
# robocasa-gr1-tabletop-tasks/README.md). Example:
ENV=gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env

python scripts/simulation_service.py --client \
    --env_name "$ENV" \
    --host localhost \
    --port 5555 \
    --video_dir ./videos/gr1_tabletop/$(basename $ENV) \
    --max_episode_steps 720 \
    --n_envs 5 \
    --n_episodes 10 \
    --n_action_steps 16
```

This connects to the server, executes a 16-step open-loop chunk per
environment step, and writes per-episode rollout videos plus a success metric
to stdout.

## Running all 24 tasks (sweep)

A simple bash sweep — start the server once, then run the client per task:

```bash
ALL_ENVS=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

mkdir -p output/alin_vla/gr1_tabletop
for ENV in "${ALL_ENVS[@]}"; do
  TASK=$(basename "$ENV")
  echo "=== $TASK ==="
  python scripts/simulation_service.py --client \
      --env_name "$ENV" \
      --host localhost --port 5555 \
      --video_dir ./videos/gr1_tabletop/$TASK \
      --max_episode_steps 720 --n_envs 5 --n_episodes 10 \
    | tee output/alin_vla/gr1_tabletop/${TASK}.log
done
```

Aggregate success rates from `output/alin_vla/gr1_tabletop/*.log`. The
upstream `scripts/simulation_service.py` prints `n_success / n_episodes` at
the end of each run.

## Quick smoke test

To sanity-check the wiring without the full sweep:

1. Server: same command as Step 1 but pointing at `nvidia/GR00T-N1.5-3B`
   (the released checkpoint) so you don't depend on local training first.
2. Client: a single env with `--n_envs 1 --n_episodes 1
   --max_episode_steps 200`. If it starts producing actions and saving a
   video, the client/server pair is wired correctly.
