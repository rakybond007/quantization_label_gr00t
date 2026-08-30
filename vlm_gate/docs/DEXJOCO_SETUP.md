# DexJoCo setup for the quantizability gate

Everything needed to reconstruct the DexJoCo half of the quantization-gate
pipeline on a fresh server. DexJoCo is the second benchmark after RoboCasa
Kitchen; the contract (16-step chunks, K2 compression, per-chunk gate,
`prediction.txt`) is identical, only the simulator and the action semantics
differ.

**Scope: single-arm tasks only.**

---

## 0. TL;DR

```bash
# one GPU is enough for a naive-K rollout (server + MuJoCo EGL share it)
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
sbatch $WS/vlm_gate/run_scripts/eval/eval_dexjoco_gated.sh          # 6 tasks, K=2, no judge
JUDGE_BACKEND=cosmos TAU=0.5 sbatch .../eval_dexjoco_gated.sh       # gated
K=1 sbatch .../eval_dexjoco_gated.sh                                # uncompressed reference
```

Files added by this work (all under `$WS/vlm_gate/`):

| file | role |
|---|---|
| `scripts/dexjoco_service_compress.py` | eval client: rollout + K-compression + gate; writes `prediction.txt` |
| `scripts/dexjoco_descriptors.py` | deterministic risk descriptors + `facts_text()` |
| `run_scripts/eval/eval_dexjoco_gated.sh` | sbatch launcher (array over the 6 single-arm tasks) |
| `docs/DEXJOCO_SETUP.md` | this file |

Nothing under `$WS/Isaac-GR00T*` or `$MG` was modified.

---

## 1. The three moving parts

DexJoCo keeps the OpenPI process split, so a rollout is **three processes in
three different Python environments**:

```
  [ GR00T policy server ]  envs/quant_gate   GPU0
        serve_policy_dexjoco.py  --  OpenPI websocket, port PORT
              ^  obs {base, wrist, state(23), prompt}
              |  actions (16, 22)
  [ DexJoCo eval client ]  envs/dexjoco      GPU0 (MuJoCo EGL)
        dexjoco_service_compress.py
              |  POST /judge {images_b64, instruction, guidance}
              v  {"decision", "confidence"}
  [ judge ]  envs/vlm_judge | cosmos_judge_venv | envs/quant_gate_eval   GPU1
```

Never `conda activate`; call the interpreters by absolute path, exactly as
`eval_robocasa_gated.sh` does. The three stacks are mutually incompatible
(MuJoCo/numpy 1.26 vs transformers/torch).

---

## 2. Environment

### 2.1 The `dexjoco` conda env (simulator + eval client)

```bash
git clone https://github.com/DexJoCo/dexjoco.git        # -> $DEXJOCO
cd $DEXJOCO
conda env create -f environment-dexjoco.yaml            # creates env "dexjoco"
pip install -e .                                        # dexjoco + dexjoco_openpi_client
pip install -e openpi/packages/openpi-client            # openpi_client
```

On this server it already exists and is **editable-installed from
`$MG/external_dependencies/dexjoco`** (note: *not* `$MG/dexjoco`, which is a
second copy — the installed one is what `import dexjoco` resolves to):

```
python      /sjw_alinlab/home/hojin2/miniconda3/envs/dexjoco/bin/python   (3.11.15)
numpy       1.26.4      mujoco 3.4.0      gymnasium 1.0.0
scipy       1.17.1      imageio (ffmpeg)  Pillow 12.2.0
dexjoco               -> $MG/external_dependencies/dexjoco/dexjoco/dexjoco
dexjoco_openpi_client -> $MG/external_dependencies/dexjoco/dexjoco/dexjoco_openpi_client
openpi_client         -> $MG/external_dependencies/dexjoco/openpi/packages/openpi-client/...
```

Verified working headless: `MUJOCO_GL=egl`, env constructs, resets, renders two
640x640 RGB cameras, and steps. Do **not** upgrade numpy in this env — MuJoCo
and the `openpi_client` msgpack path are pinned against 1.26.

### 2.2 The GR00T env (policy server)

`envs/quant_gate` (python 3.10.20) already imports the workspace fork
`$WS/Isaac-GR00T/gr00t`, and that fork already contains the DexJoCo data
configs and the serving adapter:

```
$WS/Isaac-GR00T/gr00t/experiment/data_config.py :  "dexjoco_single_arm_multi_horizon"
                                                   "dexjoco_dual_arm_multi_horizon"
$WS/Isaac-GR00T/scripts/serve_policy_dexjoco.py
```

The server needs the nvidia lib path prefix (see `SERVER_LD` in the launcher)
and `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

### 2.3 Judge env

Unchanged from RoboCasa: `envs/vlm_judge` (gemma), `cosmos_judge_venv`
(Cosmos3-Nano), `envs/quant_gate_eval` (`module_gate_server.py` student gate).
The gate HTTP contract is the same, so all three drop straight in.

---

## 3. Checkpoint and data config

```
CKPT  $MG/Isaac-GR00T/ckpt/dexjoco/groot/
        groot_n1_5_bs64_single_arm_multitask_baseline/checkpoint-60000   (7.1 GB)
data_config      dexjoco_single_arm_multi_horizon
embodiment_tag   new_embodiment
backbone         eagle   (model_type gr00t_n1_5, base Gr00tPolicy)
action_horizon   16      (data config fetches 64 steps for the MoE multi-horizon
                          losses; the baseline head emits the first 16)
```

Sibling checkpoints in the same directory, for reference:
`..._single_arm_multitask_moe4_v1_balance`, `..._moe4_v1_no_balance` (MoE — must
be served with `Gr00tPolicyFairMoe`; `serve_policy_dexjoco.py` auto-detects this
from `config.json:model_type`), and two `dual_arm` baselines (out of scope).

Modality contract of the adapter (single arm):

```
obs   base   (640,640,3) uint8 -> video.front
      wrist  (640,640,3) uint8 -> video.wrist
      state  (23,)  = [pos(3), quat(4), hand(16)]
             -> state.arm_pos / state.arm_rot(quat) / state.hand
      prompt str     -> annotation.human.action.task_description
act   (16, 22) = concat[action.arm_pos(3), action.arm_rot(3, rotvec), action.hand(16)]
```

---

## 4. Task set — which ones are single-arm

DexJoCo ships **11 tasks**; `robot_type` in `configs/*/<task>.yaml` is the
authority. **Six are single-arm:**

| task id | prompt |
|---|---|
| `water_plant` | Grasp the watering can and apply water to the plant. |
| `hammer_nail` | Use the hammer to drive the nail into the wooden board. |
| `pick_bucket` | Place the boxed food into the bucket and then lift the bucket. |
| `pinch_tongs` | Grasp the tongs and perform three consecutive open-close motions. |
| `fold_glasses` | Fold the glasses and place them into the case. |
| `click_mouse` | Move the mouse to the purple mouse pad and click the left mouse button. |

The other five are bimanual and **out of scope**: `bimanual_assembly`,
`bimanual_hanoi`, `bimanual_microwave_cook`, `bimanual_photograph`,
`bimanual_unlock_ipad`.

These ids are exactly what the eval scripts want: they are both the
`CONFIG_MAPPING` key (`dexjoco.tasks.CONFIG_MAPPING`) and the yaml basename.

### 4.1 PITFALL — pick the right config family

There are four config families under `dexjoco/configs/`. For single-arm GR00T
eval you **must** use `rand_obj/` (or `rand_full/`):

```
configs/rand_obj/water_plant.yaml     camera_mapping: {base: front,  wrist: wrist}   <-- USE THIS
configs/rand_full/water_plant.yaml    camera_mapping: {base: random_camera, wrist: wrist}
configs/multi_task/water_plant.yaml   camera_mapping: {base: front, wrist_left: wrist, wrist_right: wrist}  <-- WRONG
```

`multi_task/` uses the *bimanual* camera key names even for single-arm tasks, so
the observation dict has no `wrist` key and the GR00T adapter raises a KeyError.
`dexjoco_service_compress.py` hard-fails with an explicit message if the
`camera_mapping` keys are not exactly `{base, wrist}`.

`click_mouse` uses `base: ego_right` (not `front`) and needs a 30-step pre-roll
to a fixed pose to match the dataset's start state; the client replicates it.

---

## 5. *** Actions are ABSOLUTE targets, not deltas ***

This is the single most important difference from RoboCasa and it inverts the
compression rule.

A single-arm DexJoCo action is
`[arm_pos(3) | arm_rot(3, rotation vector) | hand(16 joint targets)]`, and every
group is an **absolute servo target** in the robot base frame.

Three independent pieces of evidence:

1. **Dataset metadata of the checkpoint** —
   `ckpt/.../checkpoint-60000/experiment_cfg/metadata.json`:
   ```json
   "action": {"arm_pos": {"absolute": true, ...},
              "arm_rot": {"absolute": true, "rotation_type": null, "shape": [3], ...},
              "hand":    {"absolute": true, "shape": [16], ...}}
   ```
2. **`DexJoCoOpenPIEnv.stay()`** holds the current pose by re-sending the
   *current state* as the action (`xyz = arm[:3]; rotvec = quat->rotvec; concat`).
   A delta controller holds still with a **zero** action; an absolute one has to
   be told where it already is. This is decisive.
3. **`DexJoCoOpenPIEnv._process_action`** converts the rotvec back to a
   quaternion and hands the raw pose to the operational-space controller
   (`sim/controllers/opspace.py`), which servos toward it.

### Consequence

Adjacent actions must **NEVER be summed**. Summing two absolute poses commands
roughly twice the world coordinate — the arm flies off. K-compression is done by
**block-last skipping**: keep the last target of each K-block, drop the
intermediate way-points, execute the `T mod K` tail raw.

```python
# dexjoco_service_compress.compress_chunk
for i in range(T // K):
    out.append(v[(i + 1) * K - 1])     # block-last: the surviving absolute target
for j in range(T // K * K, T):
    out.append(v[j])                   # ragged tail, raw
```

This is the correct absolute-space analogue of RoboCasa's delta summation: both
make the arm cover K steps of motion in one control tick, and both therefore
risk the controller failing to track. There is also **no binary gripper dim** to
latch — the 16-DoF hand is continuous and is skipped like everything else.

The K2 feasibility signal is correspondingly different: instead of "does the
summed delta exceed the controller's [-1,1] clip", we measure "how large is the
one-tick pose jump the controller is asked to service"
(`dexjoco_descriptors.skip_excess`, `prediction.txt: merge_jump_over`).

---

## 6. Descriptor layer

`scripts/dexjoco_descriptors.py` mirrors `robocasa_descriptors.py` but is
re-derived for the 22-dim absolute action space (RoboCasa's `a[:, 5:8]` EE-delta
and `a[:, 11]` gripper indices do **not** apply).

Signals (`descriptors(a, f, n, k)`), all computed from *differences of
consecutive absolute targets*:

| signal | meaning |
|---|---|
| `speed_mean` / `speed_max` | EE translation per control step (m) |
| `rot_speed_mean` / `rot_speed_max` | geodesic wrist rotation per step (rad) |
| `hand_speed_mean` / `hand_speed_max` | L2 hand-joint motion per step (rad) |
| `hand_change` / `hand_trend` | a finger transition happens; sign says closing vs opening |
| `reversal` | >90 deg turn between two significant consecutive displacements |
| `closed_slow` | hand configuration frozen while the arm creeps -> precise placement |
| `decel` | decelerating to a near stop |
| `skip_excess` (+ `_pos`/`_rot`/`_hand`) | fraction of K-merged commands whose one-tick jump exceeds what the opspace controller can track |
| `merged_pos_max` / `merged_rot_max` / `merged_hand_max` | worst merged jump |

`facts_text(x)` renders them as resolved declarative English, same style and
contract as RoboCasa's. `computed_risk(x)` / `action_rule_block(chunk)` give the
hard deterministic veto (`hand_transition`, `precise_hold`, `reversal`,
`skip_excess`), wired to the client's `--action-rules 1`.

### Threshold calibration

`JUMP_LIMIT_POS / _ROT / _HAND`, `SLOW_POS`, `FAST_POS`, `HAND_ACTIVE` are the
tunables. They were set from real GR00T chunk statistics collected on
`water_plant` with `--dump-actions` (see §8). Re-run that dump on a new
checkpoint/task mix before trusting the rule layer:

```bash
python scripts/dexjoco_service_compress.py ... --compress-k 1 \
       --dump-actions /tmp/chunks.npz
```

---

## 7. Running one rollout

Naive-K (no judge), 2 episodes, on one GPU:

```bash
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules
srun --gpus=1 --job-name=dexjoco_smoke --wckey=project-short-name:sub_fast \
     --exclude=worker-node100,worker-node1,worker-node104,worker-node3 \
     bash $WS/vlm_gate/_tmp/smoke_dexjoco.sh
```

or by hand:

```bash
# 1) server (envs/quant_gate)
LD_LIBRARY_PATH="$SERVER_LD" $CONDA/envs/quant_gate/bin/python -u \
  $WS/Isaac-GR00T/scripts/serve_policy_dexjoco.py --port 18877 \
  --model_path $CKPT --data_config dexjoco_single_arm_multi_horizon \
  --embodiment_tag new_embodiment --denoising_steps 4 --head main

# 2) client (envs/dexjoco)
MUJOCO_GL=egl PYTHONPATH=$WS/vlm_gate/scripts \
  $CONDA/envs/dexjoco/bin/python -u $WS/vlm_gate/scripts/dexjoco_service_compress.py \
  --env_name water_plant --host 127.0.0.1 --port 18877 \
  --video_dir out/wp --n_episodes 2 --max_episode_steps 700 --compress-k 2
```

**Judge by artifacts, not by exit code.** The artifacts are:

```
out/wp/prediction.txt                    episode 0 is_success: [ True] action_steps: 231
out/wp/episode_00_success/front.mp4      per-camera rollout video
out/wp/episode_00_success/wrist.mp4
out/wp/gate_conf.csv                     only when --judge-url is set
out/wp/success_rate_<n>_<N>.txt
```

`prediction.txt` is byte-compatible with the RoboCasa analysis scripts: same
`episode N is_success: [ True] action_steps: M` lines, same trailing
`is_success: <rate>` / `compress_k:` / `exec_chunk_len_mean:` /
`gate_quantize_rate:` / `gate_call_rate:` / `gate_conf_*` keys.

Verified against real RoboCasa output (`output/_vark_smoke/robocasa/task/`), not
assumed: RoboCasa writes the **numpy repr of a 1-element bool array**, so the
success token is `[ True]` *with a leading space* and `[False]` *without* (both
5 chars inside the brackets). The client reproduces this with `[{flag:>5}]`.

### PITFALL — that leading space breaks the obvious resume parser

RoboCasa's own resume parser is `tail.strip().split()[0].strip("[],")`. On
`[ True]` that yields `"["` -> `""`, so **every successful episode is dropped**,
while `[False]` parses fine; and the trailing summary line `is_success: 0.5000`
*is* accepted as a phantom episode. I reproduced this bug verbatim in the first
version of this client. The fixed parser matches a bracket/space-tolerant regex

```python
re.compile(r"is_success:\s*\[?\s*(True|False)\s*\]?")
```

which handles `[ True]`, `[False]` and a plain `True`, and correctly ignores the
summary and every other `*_rate:` key. If you write a new analysis script,
use this pattern, not `split()[0]`.

---

## 8. Observed smoke numbers

`water_plant`, `rand_obj` config, seed 42, `checkpoint-60000`, `--denoising-steps 4`,
2 episodes, `--max_episode_steps 700`, one A100-class GPU shared by the GR00T
server and the MuJoCo EGL renderer.

**K=1 (uncompressed reference)** — `prediction.txt`:

```
episode 0 is_success: [ True] action_steps: 317
episode 1 is_success: [ True] action_steps: 287
is_success: 1.0000
compress_k: 1
exec_chunk_len_mean: 16.00
```

**K=2 (naive block-last skipping, no judge)**:

```
episode 0 is_success: [False] action_steps: 700     <- hit the 700-step smoke cap
episode 1 is_success: [ True] action_steps: 224
is_success: 0.5000
compress_k: 2
exec_chunk_len_mean: 8.00
merge_jump_over: 56/926 merge_jump_excess: 0.8223
```

Read: naive K2 halves the executed chunk length (16 -> 8) and on the episode it
still solves it cuts 287 -> 224 executed steps (0.78x), but it loses episode 0.
That is exactly the success/compression gap the gate is meant to close. Two
episodes is *not* a success-rate measurement — it only proves the pipeline is
real. 6% of the merged commands (56/926) demand a one-tick jump beyond
`JUMP_LIMIT_POS`.

### Descriptor calibration (from `--dump-actions`, 38 real chunks)

| quantity | p50 | p90 | p99 | max |
|---|---|---|---|---|
| single-step EE translation (m) | 0.0080 | 0.0159 | 0.0300 | 0.0416 |
| single-step wrist rotation (rad) | 0.0283 | 0.0627 | 0.1322 | — |
| single-step hand L2 (rad) | 0.0279 | 0.0767 | 0.2307 | 0.3162 |
| K2-merged EE translation (m) | 0.0103 | 0.0238 | 0.0572 | 0.0738 |
| K2-merged hand L2 (rad) | 0.0358 | 0.1481 | 0.3314 | 0.4120 |

The first-pass thresholds (`HAND_ACTIVE=0.03`, reversal on any `cos<0`) fired on
87% / 95% of chunks — useless as risk signals. Re-tuned to `HAND_ACTIVE=0.10`,
`REVERSAL_MIN=0.012` with `REVERSAL_COS=-0.5`, `JUMP_LIMIT_ROT=0.20`, giving:

```
hand_change  0.289    reversal 0.079    closed_slow 0.026    decel 0.026
skip_excess  mean 0.086, frac>0.20 = 0.184
--action-rules 1 veto: 42% of chunks (hand_transition 11, skip_excess 3,
                                      precise_hold 1, reversal 1, out of 38)
```

**Redo this calibration for every new task/checkpoint** — the numbers above come
from `water_plant` only.

### Gate wiring check (stub judge)

`_tmp/smoke_dexjoco_gated.sh` runs the same rollout against
`_tmp/stub_judge.py`, a 30-line HTTP server that alternates
`confidence` 0.9 / 0.1. It exercises the whole gate path without spending a GPU
on a real VLM. 1 episode, `--gate-subchunk 8 --gate-ttl-max 2 --action-rules 1
--judge-facts 1`:

```
episode 0 is_success: [False] action_steps: 149
compress_k: 2
exec_chunk_len_mean: 7.09
merge_jump_over: 1/20 merge_jump_excess: 0.0234
action_rule_blocks: 8 {'hand_transition': 3, 'skip_excess': 4, 'reversal': 1}
gate_quantize_rate: 0.2273 (5/22)
gate_threshold: 0.5
gate_call_rate: 0.4091 (9/22)
gate_ttl: max=2 lo=0.15 hi=0.3 hand_trigger=1
gate_conf_mean: 0.5727 min: 0.1000 max: 0.9000 p50: 0.9000
```

with `gate_conf.csv`:

```
episode,step,conf,quantize,called,instruction
0,0,0.9000,0,1,'Grasp the watering can and apply water to the plant.'
0,8,0.9000,0,0,'...'
0,16,0.9000,0,0,'...'
0,24,0.1000,0,1,'...'
```

Confirms: HTTP judge call, 8-step sub-chunking (`step` column strides by 8), TTL
reuse (`called=0` rows), the deterministic rule veto ANDed on top (conf 0.9 but
`quantize=0`), and every summary key the RoboCasa analysis scripts read.
**The success/failure of this run is meaningless** — the judge is a coin flip.


---

## 9. Pitfalls hit while building this

1. **`/tmp` is node-local.** A helper script written to the login node's `/tmp`
   is invisible inside `srun`. Keep smoke scripts in `$WS/vlm_gate/_tmp/`.
2. **`multi_task/` configs break single-arm GR00T eval** (§4.1).
3. **Client must NOT resize frames.** The `dexjoco_single_arm_multi_horizon`
   data config registers the dataset's 640x640 video resolution and does
   `VideoCrop`+`VideoResize` itself. Upstream `_process_obs` resizes to 224x224
   with `resize_with_pad`, which trips GR00T's `VideoToTensor` resolution check.
   Both `Isaac-GR00T/scripts/dexjoco_eval_gr00t.py` and our client monkey-patch
   `_process_obs` to send native resolution. Keep the patch.
4. **Websocket arrays come back read-only**, and `scipy`'s `R.from_rotvec`
   refuses them. Copy with `np.array(a, dtype=np.float64)` before `env.step`.
5. **MoE checkpoints must not be served with the base policy class.** Loading a
   `gr00t_n1_5_fair_moe` ckpt with `Gr00tPolicy` silently downgrades the routing
   and emits near-zero actions — a robot that never moves.
   `serve_policy_dexjoco.py` auto-detects, but do not bypass it.
6. **The OpenPI websocket server prints no explicit "ready" banner** after
   binding. The launcher greps `Creating server` and then sleeps; do not shorten
   that sleep or the first `client.infer` races the bind.
7. **`--gpus=1` is enough** for `JUDGE_BACKEND=none`; the MuJoCo EGL renderer and
   the GR00T server coexist on one GPU. With a judge, use 2.
8. **Do not upgrade numpy/transformers in `envs/dexjoco`** — MuJoCo 3.4 and the
   msgpack numpy codec in `openpi_client` are pinned to numpy 1.26.
9. **Episode length is task-terminated**, not fixed. `env.is_done` fires on the
   task's own success/failure condition; `--max_episode_steps` is only a cap.
   Because compression shortens the executed sequence, `action_steps` in
   `prediction.txt` is the count of *executed control steps*, which is exactly
   the quantity the K-ladder analysis wants.
