# 다른 서버 셋업 — 에이전트에게 줄 프롬프트

아래 블록을 새 서버의 코딩 에이전트에게 그대로 붙여넣으면 된다.
워커 노드가 필요 없는 작업 — 스모크 테스트, 추론 시험, 라벨링 시험, 디버깅 — 을
그쪽에서 할 수 있게 만드는 것이 목적이다.

---

## 붙여넣을 프롬프트

You are setting up a second machine for an ongoing robotics research project. The
work is "quantizability-gated action quantization": for each 16-step action chunk a
VLA policy emits, a gate decides whether the chunk can be executed at a reduced rate
(2×, and on some embodiments 2.5× or 3×) without losing task success. Nothing here
needs to be re-derived — the design is settled and documented. Your job is to make
this machine able to run the parts that do not need a GPU cluster: import smoke
tests, single-episode inference checks, small labeling trials, and debugging.

### Start by reading

Clone both repositories and read their documentation before touching anything.

```
git clone -b action-quantization-gate-v2 \
  git@github.com:rakybond007/GR00T-action-quantization.git groot-quant
git clone git@github.com:rakybond007/gr00t-n17-quant-gate.git n17-gate
```

In `groot-quant`, read `vlm_gate/docs/RESEARCH_CONTEXT.md` first — it is the whole
project in one document: the claim, what was tried, what is settled, what is still
open, and the failure modes that keep recurring. Then, in this order:
`vlm_gate/local/STATE.md` (where the research
stands), `vlm_gate/local/PATHS.md` (environments, slurm policy, storage rules),
`vlm_gate/local/CHEATSHEET.md`, `vlm_gate/local/SETUP.md`, and every file under
`vlm_gate/local/memory/` — those are accumulated lessons, several of which cost days
to learn. Then `vlm_gate/docs/DEXJOCO_SETUP.md` and `vlm_gate/docs/DEXJOCO_DATA.md`.

In `n17-gate`, read `README.md` and `docs/CONTEXT.md`.

### Environments to build

Four Python environments, kept separate on purpose. Do not merge them and do not
"upgrade to fix" — the pins are load-bearing.

| env | purpose | hard pins |
|---|---|---|
| `quant_gate` | GR00T-N1.5 policy server | numpy 1.23.5, transformers 4.51.3 |
| `quant_gate_eval` | simulation client, student training, analysis | numpy 1.23.5, transformers 4.51.3 |
| `cosmos_judge_venv` | Cosmos3-Nano VLM judge | transformers 5.x, torch built for the local CUDA |
| `dexjoco` | dexjoco simulator | numpy 1.26.4, mujoco 3.4.0, EGL for headless render |

RoboCasa requires numpy 1.23.x. Raising numpy or transformers in the first two envs
has broken evaluation for days at a time, in ways that look like unrelated failures
(a missing processor config, an action array that silently loads through the wrong
pickle path). If a library seems too old, add it as a **PYTHONPATH overlay for one
process**, never as an upgrade — `n17-gate/setup/00_env_overlay.sh` and
`02_env_extras.sh` show the pattern, including which packages must be deleted from
the overlay afterwards so they do not shadow the environment's numpy and torch.

For N1.7 work also follow `n17-gate/setup/01_apply_to_n17.sh`, which clones NVIDIA's
public `n1.7-release` tag and applies our patch — we deliberately do not vendor that
tree.

### Weights to fetch

All from HuggingFace. The two NVIDIA ones need access approval on the account, and
the Cosmos backbone is **not** bundled inside the N1.7 checkpoint — it must be
fetched separately or the model will not construct.

```
nvidia/GR00T-N1.5-3B                                  5.1 GB
nvidia/GR00T-N1.7-3B                                  6.5 GB   (approval)
nvidia/Cosmos-Reason2-2B                              4.6 GB   (approval, separate)
facebook/dinov3-vits16-pretrain-lvd1689m               83 MB   (gated, accept licence)
nvidia/Cosmos3-Nano                                            (the labeling judge)
```

Set `HF_HUB_DISABLE_XET=1`. Note that recent `transformers` contacts the HF API even
when a model is fully cached, so a fully offline machine needs `HF_HUB_OFFLINE=1`
plus a complete cache, and some code paths still fail — see the N1.7 notes.

### Datasets to fetch

```
DexJoCo/DexJoCo-Datasets-LeRobot        HF dataset, public
    six single-arm tasks only: water_plant, hammer_nail, pick_bucket,
    pinch_tongs, fold_glasses, click_mouse — 100 episodes each, ~5.2 GB
    NOTE: packed layout (one parquet per task, concatenated videos), not the
    per-episode layout older scripts assume. DEXJOCO_DATA.md explains the handling.

RoboCasa Kitchen (LeRobot, robocasa_mg_gr00t_300)    ~180 GB
    the main dataset. Only fetch it if you intend to relabel or retrain; for
    smoke tests and debugging a handful of episodes is enough.

allex recordings                                      ~2 GB
    internal, not on HuggingFace — copy from the origin machine if needed.
```

Teacher labels for RoboCasa (247,887 chunks) are committed inside `n17-gate/labels/`,
so labeling does not have to be repeated to train or evaluate a student.

The 37 GB RoboCasa frame cache is a derived artifact; regenerate it rather than
copying it, and keep it on local storage that is not subject to archival.

### Storage rule that has already cost us data

Anything under a path that archives unaccessed data will disappear — a 37 GB frame
cache was lost this way, and the dexjoco training set built by an earlier conversion
is simply gone. Reading a file does not protect it. Keep working assets in the home
directory and treat everything else as regenerable.

### What to verify, in order

1. Both repos clone and every documented script parses (`ast.parse`, `bash -n`).
2. Each environment imports its stack and reports the pinned versions.
3. The N1.5 policy loads from a checkpoint and returns an action chunk of the
   expected shape for one observation.
4. The Cosmos judge serves and answers one labeling call, with the slot-scored
   `A) YES` / `A) NO` format parsing into per-question probabilities.
5. One dexjoco episode decodes: a frame of the expected shape and an action array
   whose dtype and key names match `meta/modality.json`.
6. A student gate checkpoint loads and serves a confidence for one observation.

Judge each step by the artifact it produces, never by exit code. That rule exists
because several failures in this project completed successfully while doing nothing:
a training run whose gate loss reached no parameters, an evaluation whose task list
was empty, a labeling job whose questions all collapsed to one answer.

### What NOT to do

- Do not upgrade numpy or transformers in `quant_gate` / `quant_gate_eval`.
- Do not run recursive `find`, `du`, or `grep -r` over shared network mounts.
- Do not write into datasets owned by other users; mirror the metadata and symlink
  the bulk instead.
- Do not submit long cluster jobs from this machine unless asked — it is meant for
  the work that does not need them.

Report what built, what is missing, and anything the documentation got wrong.

---

## 이 프롬프트를 쓸 때

저쪽 에이전트가 물어올 만한 것 — HuggingFace 토큰, 승인이 필요한 리포 두 개
(`GR00T-N1.7-3B`, `Cosmos-Reason2-2B`)의 접근 권한, 그리고 allex 녹화본은
HuggingFace에 없으므로 원본 머신에서 복사해야 한다는 점.
