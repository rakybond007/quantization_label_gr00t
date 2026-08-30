# SimplerEnv Evaluation Setup (Bridge / WidowX + Fractal / Google Robot)

End-to-end notes for evaluating GR00T-N1.5 checkpoints (baseline and our
`mh_m8_econsist` variant) on the SimplerEnv benchmark, using the
youliangtan SimplerEnv fork at
`/sjw_alinlab2/home/hojin2/multigpu_workspace/SimplerEnv`.

## Pre-reqs

- `gr00t` conda env (Python 3.10) — for the inference server.
- `simpler_env` conda env (Python 3.10) — for the maniskill2 client.
  See `SimplerEnv/SETUP_NOTES.md` for the install commands the user must
  run manually (sandbox blocked the `pip install -e .` of the cloned repo).
- A trained GR00T checkpoint (see `Isaac-GR00T/run_scripts/train/simplerenv/`).
- Datasets (only needed if also training): downloaded into
  `/sjw_alinlab2/home/hojin2/multigpu_workspace/data/simplerenv/`.

## Step 1: launch the inference server (in `gr00t` env)

The server loads the checkpoint and handles policy inference over a
ZMQ socket. Same process is used for both Bridge and Fractal — only the
`--data-config` and checkpoint path differ.

### Bridge / WidowX, baseline checkpoint

```bash
conda activate gr00t
cd /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T
python scripts/inference_service.py \
    --server \
    --model-path ckpt/simplerenv/bridge/groot_n1_5_bs64_baseline/checkpoint-60000 \
    --data-config simplerenv_bridge \
    --embodiment-tag new_embodiment \
    --denoising-steps 8 \
    --port 5555
```

### Bridge / WidowX, mh_m8_econsist checkpoint

```bash
conda activate gr00t
cd /sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T
python scripts/inference_service.py \
    --server \
    --model-path ckpt/simplerenv/bridge/groot_n1_5_bs64_mh_m8_econsist/checkpoint-60000 \
    --data-config simplerenv_bridge_multi_horizon \
    --embodiment-tag new_embodiment \
    --denoising-steps 8 \
    --head main \
    --discrete-action-dims 6 \
    --port 5555
```

Choices for `--head` (only meaningful if checkpoint trained with
multi-horizon loss):
- `main` — standard 16-step head
- `merged_8` — 8-step m8 head
- `f2`, `f4` — auxiliary compressed heads
- `ensemble` — least-squares combine main+f2+f4
- `ensemble_fix` — same as `ensemble` but discrete dims taken from `main`
  (recommended for tasks with binary gripper).

Bridge action layout: `[x, y, z, roll, pitch, yaw, gripper]` so the gripper
index in the concatenated action vector is `6`. Use
`--discrete-action-dims 6` for the `ensemble_fix` head.

### Fractal / Google Robot

Same pattern — replace `bridge` with `fractal`:

```bash
conda activate gr00t
python scripts/inference_service.py \
    --server \
    --model-path ckpt/simplerenv/fractal/groot_n1_5_bs64_baseline/checkpoint-60000 \
    --data-config simplerenv_fractal \
    --embodiment-tag new_embodiment \
    --denoising-steps 8 \
    --port 5555
```

For the mh_m8_econsist variant: `--data-config simplerenv_fractal_multi_horizon`.

## Step 2: run the eval client (in `simpler_env` env)

Single task:

```bash
conda activate simpler_env
cd /sjw_alinlab2/home/hojin2/multigpu_workspace/SimplerEnv
python eval_simpler.py --env widowx_spoon_on_towel --groot_port 5555
python eval_simpler.py --env google_robot_pick_object --groot_port 5555
```

Available task names (per upstream README):

WidowX (Bridge):
- `widowx_spoon_on_towel`
- `widowx_carrot_on_plate`
- `widowx_put_eggplant_in_basket`
- `widowx_stack_cube`
- `widowx_put_eggplant_in_sink`
- `widowx_close_drawer`
- `widowx_open_drawer`

Google Robot (Fractal):
- `google_robot_pick_coke_can`
- `google_robot_pick_object`
- `google_robot_move_near`
- `google_robot_open_drawer`
- `google_robot_close_drawer`
- `google_robot_place_in_closed_drawer`

Variant aggregation evals (longer; loops over scene/object variants):

```bash
bash run_evaluations_variant_agg_drawer.sh
bash run_evaluations_variant_agg_move_near.sh
bash run_evaluations_variant_agg_pick_coke_can.sh
```

## Useful eval_simpler.py args (besides `--env` and `--groot_port`)

- `--eval_count <N>` — episodes per setting (default 50; upstream report uses 300).
- `--episode_length <N>` — max steps per episode (default 120).
- `--action_horizon <N>` — how many predicted action steps to execute per
  policy call (default 1; tune up to use the diffusion policy's chunk).
- `--output_video_dir <path>` — dump rollout MP4s here.
- `--headless` — disable on-screen rendering (required on a SLURM node).

## Sanity check (no trained ckpt)

You can smoke-test the server <-> client wiring with the upstream Bridge
checkpoint:

```bash
# server
conda activate gr00t
python scripts/inference_service.py \
    --server \
    --model-path youliangtan/gr00t-n1.5-bridge-posttrain \
    --data-config simplerenv_bridge \
    --embodiment-tag new_embodiment \
    --denoising-steps 8 \
    --port 5555

# client
conda activate simpler_env
python eval_simpler.py --env widowx_spoon_on_towel --groot_port 5555 --eval_count 1 --headless
```

## Gotchas

- The `simpler_env` env requires `numpy<2.0` and SAPIEN (Vulkan). If Vulkan
  fails on the eval node, see the upstream troubleshooting in
  `SimplerEnv/README.md`.
- The server must be running before launching `eval_simpler.py`.
- `--port` on the server and `--groot_port` on the client must match.
- Bridge gripper dim index is `6`; Fractal gripper dim index is also `6`
  (both have the same `[x, y, z, R, P, Y, gripper]` layout).
