---
name: cosmos-judge-model
description: "Cosmos VLM judge = nvidia/Cosmos3-Nano via the transformers reasoner path (cu124), NOT vLLM; how it's wired and why"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
---

The second VLM-judge backend for `vlm_gate` (alternative to the Gemma-4 judge) is
**`nvidia/Cosmos3-Nano`** (16B "Cosmos 3" omnimodal, Reasoner surface). Working setup
(validated end-to-end 2026-06-24):

- **Path: Transformers, NOT vLLM.** Use the official NVIDIA/cosmos repo recipe
  "Reasoner with Transformers" (repo cloned at `quantization_agent_workspace/cosmos`).
  Model class `Cosmos3OmniForConditionalGeneration` + `AutoProcessor`, in-process,
  loads only the reasoner tower. The judge reads {YES,NO} first-token logits → softmax
  → continuous P(YES), exactly like the Gemma judge (`scripts/vlm_gate_cosmos.py`,
  imports SYSTEM/build_messages/VLMGate from `vlm_gate.py`). Same POST /judge contract.
- **Why not vLLM:** vllm 0.23 pulls torch 2.11 / **CUDA 13**, but the cluster GPU driver
  is **550.x = CUDA 12.4** → "NVIDIA driver too old" at CUDA init. Major version (13 vs 12)
  is not covered by minor-version compat. The vLLM shim is parked at
  `scripts/vlm_gate_cosmos.py.vllm.bak` for a future CUDA-13 node.
- **Env: `cosmos_judge_venv`** (uv venv, python 3.13) at
  `quantization_agent_workspace/cosmos_judge_venv`. Built with
  `uv pip install --torch-backend=cu124 torch torchvision transformers>=5.11.0 accelerate av pillow safetensors>=0.8.0`.
  Resolved: **torch 2.6.0+cu124, torchvision 0.21.0+cu124, transformers 5.12.1**.
  torchvision IS required (Qwen3VL video processor). cu124 = matches the 12.4 driver
  (CUDA *runtime* lives in the env; *driver* is the host ceiling and can't be raised from there).
- **Smoke / eval drivers:** `run_scripts/eval/_smoke_cosmos_judge.sh` (judge only) and
  `_smoke_robocasa_cosmos_gate.sh` (base GR00T GPU0 + Cosmos judge GPU1 + compress client).
  Base GR00T ckpt stays on the shared tree
  (`~/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000`).
- **Migration gaps fixed:** copied `scripts/inference_service.py` (base server) and
  `scripts/robocasa_service_selective.py` (compress-client dep) from the shared tree.

See [[vlm-gate-env-separation]], [[ops-download-and-bg-task-lessons]].
