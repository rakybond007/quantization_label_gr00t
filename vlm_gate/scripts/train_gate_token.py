#!/usr/bin/env python
"""Launcher for quantizability-gate-token joint VLA training (plan C).

The actual implementation lives in the private fork:
  - model: Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py
      attach_quant_gate / _quant_gate_stream / get_action_and_gate
      + gate BCE wiring in FlowmatchingActionHead.forward
  - training entry: Isaac-GR00T/scripts/gr00t_finetune.py
      (--use-quant-gate --quant-gate-labels ... --quant-gate-loss-weight ...)

This wrapper simply execs gr00t_finetune.py with the fork on sys.path, so
`train_gate_token.py <args...>` == `gr00t_finetune.py <args...>`.
Submission scripts: run_scripts/train/sbatch_vla_gateC_{cosmos,gemma}_60k.sh
"""
import os
import sys

PRIV = os.path.expanduser("~/quantization_agent_workspace/Isaac-GR00T")
script = os.path.join(PRIV, "scripts", "gr00t_finetune.py")
os.execv(sys.executable, [sys.executable, "-u", script] + sys.argv[1:])
