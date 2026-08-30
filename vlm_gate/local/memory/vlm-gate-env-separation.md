---
name: vlm-gate-env-separation
description: How the vlm_gate / ATQ action-quantization work is isolated from the shared Isaac-GR00T (private source + private conda envs)
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
---

The `quantization_agent_workspace/vlm_gate` (external-VLM-gate + self-evolving-prompt
action quantization) is **fully separated** from the shared `~/multigpu_workspace/Isaac-GR00T`.

- **Private gr00t source**: `quantization_agent_workspace/Isaac-GR00T` — git clone of the
  shared repo, branch `action-quantization-impl`, GitHub remotes preserved (`origin`=ismty0805,
  `quant`=rakybond007). The `external_compress_bias` router-bias patch is committed in-source
  (commit `7dc3bfa`) in BOTH heads: `flow_matching_action_head_fair_moe.py` (MoE) and
  `flow_matching_action_head.py` (base). The patch adds bias to router logits idx1=m8/idx2=m4
  before the softmax.
- **Private conda envs** (cloned, then `pip install -e ../Isaac-GR00T --no-deps`):
  - `quant_gate` (from `gr00t`) — policy server (`inference_service_fair_moe.py`), torch 2.5.1+cu124.
  - `quant_gate_eval` (from `robocasa_gr00t`) — robocasa eval clients + `probe_router_bias.py`.
  - `vlm_judge` — Gemma gate, unchanged (no gr00t dep).
- **Intentionally still shared**: `robosuite`/`robocasa` simulators (editable →
  `~/multigpu_workspace/{robosuite,robocasa}`, separate repos, not part of the gr00t fork);
  HF checkpoint cache (`~/.cache/huggingface`). MoE ckpt: `prehj/GR00T-N1.5-robocasa-moe4-v1-K4-b-only-no-metaq-60k`.
- **run_scripts/eval/*.sh** already rewired: policy server → `quant_gate`, eval clients →
  `quant_gate_eval`, judge → `vlm_judge`.

Testing runs on an interactive SLURM alloc via `srun --jobid=<job> --overlap --gres=gpu:2 bash ...`.
Verified 2026-06-23: `probe_router_bias.py` shows m8/m4 probs rise monotonically with bias and the
argmax flips to the compressed decoder — patch reaches the router through the separated stack.
See `vlm_gate/README.md` for the authoritative layout. Related: [[rats-codebase]]
