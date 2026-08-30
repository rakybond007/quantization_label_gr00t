# vlm_gate — Self-Evolving External VLM Gate for Action Quantization

Migrated out of `multigpu_workspace/Isaac-GR00T` (the ATQ codebase) into its own
workspace. This holds the **external VLM gate** layer (a frozen VLM decides, per
control block, whether the upcoming actions can be temporally quantized, and
biases the MoE router toward the compressed decoders) plus the **self-evolving
guidance** loop. The ATQ model/router itself stays in Isaac-GR00T.

## Layout
- `scripts/`
  - `vlm_gate.py` — frozen VLM judge server (`--serve`) + `VLMGate` client (Gemma, YES/NO-logit confidence).
  - `inference_service_fair_moe.py` — MoE policy server with `_BiasAdapter` (injects `obs["_compress_bias"]` → `head.external_compress_bias`).
  - `robocasa_service_moe.py` — robocasa eval client; per-chunk gate query → signed router bias (`--bias-mode signed`).
  - `robocasa_service_compress.py` — base-GR00T (no internal router) gate eval client.
  - `evolve_gate_prompt.py` / `evolve_gate_prompt_moe.py` — one self-evolution cycle (base / MoE).
  - `evolving_guide_prompt.txt` — fixed meta-policy for the evolver.
  - `probe_router_bias.py` — controlled same-obs probe (verifies bias reaches the router).
- `run_scripts/eval/` — sbatch/smoke drivers + `vlm_gate_guidance*.txt`. (`BASE_DIR` already repointed here.)
- `analysis/_evolver/` — meta-policy, `evolution_log*.jsonl`, `guidance_versions{,_moe}/`.
- `output/robocasa/` — new eval outputs land here. **Baseline dirs are symlinks** to
  `Isaac-GR00T/output/robocasa` (read-only references used by the evolver):
  `baseline_full_v2_with_action_steps`, `baseline_compress_K2`, `moe_router_confp07_ctrl`.

## Conda envs (private — fully separated)
| env | use |
|---|---|
| `quant_gate` | MoE/base policy server (`inference_service_fair_moe.py`). `gr00t` **editable-installed → this workspace's private `../Isaac-GR00T/`** (carries the `external_compress_bias` patch). Cloned from `gr00t`. |
| `vlm_judge` | Gemma gate (`vlm_gate.py --serve`). torch 2.7.1+cu126, transformers 5.10.2, `HF_HUB_OFFLINE=1`. (unchanged — no `gr00t` dep) |
| `quant_gate_eval` | robocasa eval clients (`robocasa_service_*.py`, `probe_router_bias.py`). `gr00t` editable → private `../Isaac-GR00T/` (so `gr00t.eval.*` harness edits stay local). Cloned from `robocasa_gr00t`. `robosuite`/`robocasa` simulators remain shared editable installs → `~/multigpu_workspace/{robosuite,robocasa}` (sim repos, not part of the gr00t fork). |

Checkpoints: HuggingFace cache (`~/.cache/huggingface`, shared). MoE ckpt repo:
`prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k` (auto `snapshot_download`).

## Run
```bash
# smoke (interactive GPU only — NOT sbatch background):
srun --jobid=<alloc> --overlap bash run_scripts/eval/_smoke_robocasa_moe_vlm_router.sh "OpenDrawer CoffeePressButton" 2 0.5 0.5
# full eval (signed gate, confp07):
sbatch --export=ALL,MOE_CONF=0.7,BIAS=0.5,BIAS_MODE=signed,GUIDANCE_FILE=$PWD/analysis/_evolver/guidance_versions/v9_manual.txt run_scripts/eval/eval_robocasa_moe_vlm_router.sh
# self-evolving loop (MoE):
nohup bash run_scripts/eval/self_evolve_loop_moe.sh 3 1 > analysis/_evolver/loop_moe.log 2>&1 &
```

## Coupling to ATQ (gr00t) — fully decoupled ✅
This workspace now owns a **private `gr00t` fork** at `../Isaac-GR00T/` (sibling of
`vlm_gate/`), independent of the shared `~/multigpu_workspace/Isaac-GR00T`:

- **Source**: `git clone` of the shared repo on branch `action-quantization-impl`,
  history + GitHub remotes (`origin`, `quant`) preserved. The `external_compress_bias`
  router-bias patch is **committed in-source** (`7dc3bfa`) in **both** heads
  (`flow_matching_action_head_fair_moe.py` = MoE, `flow_matching_action_head.py` = base).
- **Envs**: `quant_gate` (policy server) and `quant_gate_eval` (robocasa clients) both
  have `gr00t` editable-installed → this private fork. Verified: `gr00t.__file__`
  resolves under `quantization_agent_workspace/Isaac-GR00T`, and the patch is present
  in the loaded module.
- **Not forked (intentionally shared)**: `robosuite`/`robocasa` simulator repos
  (`~/multigpu_workspace/{robosuite,robocasa}`) — separate from the gr00t source;
  cloning them would mean ~tens of GB of assets. Checkpoints stay on the shared HF
  cache (`~/.cache/huggingface`); nothing copied.

The shared `~/multigpu_workspace/Isaac-GR00T` is no longer imported by anything here
and was left untouched. To pull upstream ATQ changes into the fork later:
`git -C ../Isaac-GR00T fetch origin && git -C ../Isaac-GR00T rebase origin/action-quantization-impl`
(the patch commit `7dc3bfa` rides on top).
