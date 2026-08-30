# Running the RoboCasa Kitchen evaluation

The benchmark is **24 kitchen tasks × 50 episodes = 1,200 closed-loop episodes**.
It reports success rate and average steps-to-completion; the gate's whole purpose
is to cut steps without losing success, so both numbers matter.

Reproducibility noise is **±1.5 pp** on success, measured by retraining an
identical student and re-running. Do not read differences smaller than that as
real, and do not quote partial runs — a run that looked like 0.732 at 336
episodes finished at 0.643.

---

## 1. What you need

| piece | where | size |
|---|---|---|
| base policy | [`prehj/groot-n15-robocasa-kitchen-baseline`](https://huggingface.co/prehj/groot-n15-robocasa-kitchen-baseline) | 7.1 GB |
| A′ gate | [`prehj/groot-n15-quantizability-gate-A-robocasa`](https://huggingface.co/prehj/groot-n15-quantizability-gate-A-robocasa) | 1.3 MB |
| *or* architecture C | [`prehj/groot-n15-quantizability-gate-C-robocasa`](https://huggingface.co/prehj/groot-n15-quantizability-gate-C-robocasa) | 7.6 GB |
| code | this branch | |
| simulator | RoboCasa + robosuite (below) | |

**A′ needs the base policy; C does not** — C is one self-contained policy with
the gate token inside its DiT. Mixing them up is the most common setup error.

Hardware: 2 GPUs is the comfortable layout (policy on GPU0, VLM judge on GPU1).
With the distilled A′ gate one GPU is enough — the 0.32 M-param CNN rides along
on GPU0 next to GR00T.

---

## 2. Environments

Three separate environments, and the separation is deliberate — do not merge
them. Package versions that actually work:

### `quant_gate` — serves the GR00T policy
```
python 3.10.20   torch 2.5.1+cu124   torchvision 0.20.1   numpy 1.26.4
transformers 4.51.3   diffusers 0.30.2   accelerate 1.2.1   flash-attn 2.7.1.post4
av 12.3.0   opencv-python 4.8.0.74   pandas 2.2.3   pyarrow 14.0.1
```
Install this repo in it: `pip install -e .`

### `quant_gate_eval` — runs the RoboCasa client and trains/serves A′
Same core as above, plus the simulator:
```
mujoco 3.2.6   robosuite 1.5.2 (editable)   robocasa 0.2.0 (editable)
transformers 5.15.0   pandas 2.1.4
```

> **numpy must stay at 1.26.4 here.** robocasa, tensorflow, cv2 and pyarrow in
> this env are all compiled against numpy 1.x; upgrading to numpy 2 breaks
> `import pandas` and `import cv2` outright. Anything that wants numpy 2 —
> `sentence-transformers`, for instance — goes in its own environment. Installing
> it here once cost a day of debugging.

### judge environment (only if you use a VLM judge rather than the A′ student)
- Gemma-4: `vlm_judge` env — transformers ≥ 4.52, numpy 2.x is fine here
- Cosmos3-Nano: a separate venv (`cosmos_judge_venv`) on transformers with
  cu124 wheels — it is served through `transformers`, not vLLM

Running the published A′ gate needs **none** of these — it is a plain CNN.

### Simulator install notes
```bash
git clone https://github.com/robocasa/robocasa && pip install -e robocasa
git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1 && pip install -e robosuite
python -m robocasa.scripts.download_kitchen_assets       # assets are large
```
RoboCasa needs a working GL context. On a headless node set
`MUJOCO_GL=egl` (or `osmesa`). Missing kitchen assets show up as instant episode
failures rather than a clear error.

---

## 3. Architecture at runtime

Three processes talk over localhost HTTP:

```
robocasa_service_compress.py          <-- the RoboCasa client, owns the env loop
   |  POST /act        -> inference_service.py   (GR00T policy, GPU0)
   |  POST /judge      -> the gate               (A' student, VLM, or C's own token)
   v
   per-chunk decision: compress (K2) or run the chunk as-is
```

The judge contract is the same for every backend, so the student and the VLM are
interchangeable:

```json
POST /judge  {"images_b64": [...3 views...], "instruction": "...", "guidance": "..."}
        ->   {"decision": "YES", "confidence": 0.71}
```

That is why you can swap a 1.3 MB CNN in for a 12 B VLM without touching the
harness.

---

## 4. Run it

```bash
sbatch run_scripts/eval/eval_robocasa_gated.sh          # SLURM, 8-way array, 3 tasks each
```

Or directly, one task:

```bash
# GPU0: policy server
python scripts/inference_service.py --server --port 10000 \
    --model_path <base-policy> --data_config single_panda_gripper \
    --embodiment_tag new_embodiment --denoising_steps 4 --head main

# gate server (A' student — fine on the same GPU)
python scripts/module_gate_server.py --ckpt A_robocasa_gemma4.pt \
    --task-emb robocasa_task_embeddings.npz --port 20000

# client
python scripts/robocasa_service_compress.py --port 10000 --judge-url http://127.0.0.1:20000 \
    --env_name PnPCounterToSink --n_episodes 50 --max_episode_steps 1500 \
    --seed 42 --generative_textures --compress-k 2 --judge-threshold 0.5 \
    --video_dir out/PnPCounterToSink
```

### Knobs that matter

| variable | default | what it does |
|---|---|---|
| `TAU` | 0.5 | compress when gate confidence ≥ τ. **Search this** — see below |
| `JUDGE_BACKEND` | `gemma` | `module` (A′), `moduleB`, `cosmos`, `gemma` |
| `MODULE_CKPT` | — | required when `JUDGE_BACKEND=module` |
| `N_EPISODES` | 50 | keep at 50 for comparable numbers |
| `MAX_STEPS` | 1500 | episode cap |
| `CLIP_SCALE` | 1 | controller action clip. K2 merges exceed ±1 in 23.5 % of blocks; ×3 removes clipping entirely |
| `JUDGE_ACTIONS` | 0 | also send the planned action numbers to the judge |
| `K3TAU` | 0 | K3 ladder. Leave off — K3 is below K2 even unclipped |

### The 24 tasks

`TurnSinkSpout TurnOnStove TurnOnSinkFaucet TurnOnMicrowave TurnOffStove
TurnOffSinkFaucet TurnOffMicrowave PnPStoveToCounter PnPSinkToCounter
PnPMicrowaveToCounter PnPCounterToStove PnPCounterToSink PnPCounterToMicrowave
PnPCounterToCab PnPCabToCounter OpenSingleDoor OpenDrawer OpenDoubleDoor
CoffeeSetupMug CoffeeServeMug CoffeePressButton CloseSingleDoor CloseDrawer
CloseDoubleDoor`

---

## 5. Choosing τ

**τ is not a constant and does not transfer between gates.** Judges calibrate
differently: a cosmos3-distilled student tops out at confidence 0.726, a
gemma4-distilled one reaches 0.999. The same τ=0.5 is a completely different
operating point on those two distributions. Copying a threshold across gates once
silently disabled a whole experiment — the branch never fired and the results
were invalid.

So: sweep τ per gate, on tuning tasks held out from the tasks you report. A
practical ladder is τ ∈ {0.38, 0.42, 0.5, 0.6} — for the cosmos A′ student that
moves the compression rate across 0.19 → 0.63. Then read the
success/steps trade-off:

| τ | success | avg steps |
|---|---|---|
| — (uncompressed) | 0.657 | 327 |
| 0.5 | 0.659 | 299 |
| 0.42 | 0.635 | 264 |
| 0.38 | 0.608 | 258 |
| — (naive K2, no gate) | 0.598 | 221 |

If your labels are rank-normalized, τ reads directly as the block rate, which
makes the sweep interpretable.

---

## 6. Reading the results

Each task directory gets a `prediction.txt` plus per-episode sidecar files.
Aggregate across all 24 before concluding anything — per-task variance is large,
and PnP tasks behave very differently from knob/button tasks.

Two failure modes that have actually bitten us:

- **Quoting a partial run.** Always report how many tasks and episodes finished.
- **Preemption.** On a requeueing partition the run can restart and silently
  truncate `prediction.txt`. The durable per-episode sidecar exists for exactly
  this reason — trust it over the aggregate file.

For comparisons, use paired per-task bootstrap confidence intervals rather than
comparing two point estimates; with ±1.5 pp noise, a 1-point difference is not a
result.
