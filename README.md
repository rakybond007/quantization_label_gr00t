# Quantizability labelling for GR00T

A robot policy plans 16 steps at a time. Some of those stretches are reaching
through empty space and could be executed at half rate with nothing lost;
others are the moment a gripper closes on a handle, where dropping every
second command breaks the task. This repository builds the thing that tells
them apart: a **gate** that answers, per chunk, whether it can be compressed.

Compressing every chunk uniformly is the baseline to beat. On RoboCasa's 24
tasks, running uncompressed succeeds 65.6% of the time in 330 steps, and
blanket K=2 compression succeeds 59.9% in 216. A gate that picks its chunks
lands at 62.7–64.2% in 252–276. **Where you compress matters more than how
much.**

Everything needed to run this is here — the GR00T model code sits alongside
the gate work, so there is no second repository to fetch and keep in step.
The upstream NVIDIA README is [README_upstream_gr00t.md](README_upstream_gr00t.md).

## How the gate decides

```
1. Embodiment adapter    What compression even means depends on the action space
                         RoboCasa, LIBERO  = end-effector deltas -> sum adjacent steps
                         dexjoco, allex    = absolute joint targets -> drop intermediate ones
2. Deterministic layer   What the action numbers alone already answer: gripper
                         transitions, path bends, precision holds, whether a merged
                         command would leave the controller's range. Never asked of a VLM.
3. VLM layer             Only what the numbers cannot answer in principle: what is
                         this scene, what is the arm up against. One call per chunk.
4. Aggregate & distil    noisy-OR, rank-normalise, train a small student on the result
```

Layer 3 exists because layer 2 has a blind spot, not to double-check it. Ask a
VLM something the arithmetic already settled and it recites the fact back —
an early version collapsed to two distinct answers across the whole dataset.
So the computed values are **stated to the model as facts and never asked
about**.

The student is 1.3 MB against the policy's 7.1 GB, and takes no action input
by design, so it runs concurrently with action-head denoising rather than
after it.

## Layout

| Path | What it is |
|---|---|
| `gr00t/`, `scripts/` | GR00T model code: training (`scripts/gr00t_finetune.py`), serving (`scripts/serve_policy.py`) |
| `vlm_gate/scripts/*_descriptors*.py` | Layer 2, one module per benchmark |
| `vlm_gate/analysis/_evolver/` | Layer 3: the guidance texts and question sets |
| `vlm_gate/scripts/cosmos_1call_v6.py` | Layer 3 driver — one judge call per chunk, reading answer-slot probabilities |
| `vlm_gate/scripts/aggregate_*.py` | Layer 4: labels to parquet |
| `vlm_gate/scripts/recompute_soft_*.py` | Makes the computed flags continuous, with no teacher calls |
| `vlm_gate/scripts/train_gate_module.py` | Student training (CNN or frozen DINOv3) |
| `vlm_gate/scripts/*_service_compress.py` | Closed-loop evaluation with the gate in the loop |
| `vlm_gate/run_scripts/` | The slurm jobs that actually ran all of the above |
| `bin/qgate`, `vlm_gate/qgate/` | Reading and verifying results |

## Prompts

The prompt is what separates a useful labelling pass from a useless one.
`GUIDANCE` selects the generation (`vlm_gate/scripts/cosmos_1call_v6.py`):

| `GUIDANCE` | Guidance file | Questions |
|---|---|---|
| `phase5` | `_varkA/robocasa_guidance_phase_v3.txt` | 4 |
| `phase6` | `_varkA/robocasa_guidance_phase_v5.txt` | 5, one per axis |

LIBERO and dexjoco keep their own guidance and questions under `_libero/` and
`_dexjoco/`. Earlier versions are kept because each one is the control for the
next.

## Running it

```bash
# 1. Label
sbatch vlm_gate/run_scripts/label/sbatch_phase6_full.sh

# 2. Verify before training on it — never skip this
bin/qgate labels v6b_phase6_s16 --expected 247887

# 3. Aggregate, then make the computed flags continuous
python vlm_gate/scripts/aggregate_phase6.py
python vlm_gate/scripts/recompute_soft_phase6.py

# 4. Score the labels against what compression actually costs — never skip this
bin/qgate labelcheck assets/labels/robocasa/<new>.parquet \
    --dataset <lerobot root> --reference assets/labels/robocasa/v6b_phase5_softA.parquet

# 5. Train the student, smoke first
bash   vlm_gate/run_scripts/train/_smoke_train_phase6.sh
sbatch vlm_gate/run_scripts/train/sbatch_train_phase6_softA.sh

# 6. Evaluate, then judge
bin/qgate tradeoff robocasa --fast baseline_compress_K2 \
    --slow baseline_full_v2_with_action_steps
```

**Steps 2 and 4 are not optional.** These labelling jobs end on a `kill`, so they exit
0 no matter what happened before it, and a shard that gets preempted and
requeued can re-emit rows it already wrote. The verdict has to come from the
rows on disk. `qgate labels` counts them per shard, finds duplicate
`(episode, frame)` pairs, and reports the spread of every answer slot.

Step 4 is the one that was missing. A label set is only worth training on if it
ranks tasks the way compression damage ranks them — `qgate labelcheck` measures
that against the uncompressed and blanket-K=2 runs you already have. The phase6
generation passed every other check, revived the dead question it was written to
revive, and scored +0.019 against phase5's +0.420. Lively questions are a
property of the questions, not of the labels.

## Reading results

Results are not stored here. Point the tools at the workspace that holds them:

```bash
export QGATE_WS=/path/to/quantization_agent_workspace
bin/qgate results robocasa
```

**Success rate alone ranks gates backwards.** Compression trades success for
speed, so the gate that compresses nothing scores highest and is worthless.
Uncompressed and blanket compression are two points in the (steps, success)
plane; the line between them is the trade random chunk selection already
gives you, and only distance above that line is evidence a gate picks the
right chunks. `qgate tradeoff` computes it, restricted to the tasks every run
actually finished — a task with 2 of 50 episodes has a success rate quantised
to 0.5 and will invent differences if you let it in.

The traps worth knowing are in [docs/TOOLKIT.md](docs/TOOLKIT.md). Where each
benchmark currently stands — prompt, labelling, student, evaluation, one row
per run — is [docs/STATUS.md](docs/STATUS.md). Trained weights and their
HuggingFace locations are in [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## If you are picking this up cold

Read in this order:

| | |
|---|---|
| [docs/HANDOFF.md](docs/HANDOFF.md) | what is decided, what is running, what is open |
| [docs/WORKSPACE_MAP.md](docs/WORKSPACE_MAP.md) | where everything lives on the cluster, and which side of the repo line it is on |
| [docs/REMOTE_AGENT.md](docs/REMOTE_AGENT.md) | driving the cluster from a laptop |
| [docs/STATUS.md](docs/STATUS.md) | one row per run: prompt, labelling, student, evaluation |
| [docs/TOOLKIT.md](docs/TOOLKIT.md) | the measurements and the traps in them |

## Not included

Evaluation results, label parquets, trained weights, logs. The weights are
public on HuggingFace and `docs/CHECKPOINTS.md` gives every path and URL.
