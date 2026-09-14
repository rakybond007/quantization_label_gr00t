"""Does our clip patch actually reach the controller that clips?

`_patch_clip_bounds` multiplies input/output bounds on every controller it can
find. If it finds none, or finds ones that are not the one clipping the action,
the run records `clip_scale: 3.0` in its header and changes nothing — which is
what LIBERO's numbers look like (876 vs 863 successes out of 2000).

Read-only on the main repo: imports the function, does not edit it.
"""
import importlib.util
import os
import sys

import numpy as np

EVAL = os.path.expanduser(
    "~/quantization_agent_workspace/Isaac-GR00T/gr00t/eval/libero/eval_taskwise_gr00t_quantize.py")
spec = importlib.util.spec_from_file_location("ev", EVAL)
ev = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ev)
except Exception as e:
    print(f"[probe] 평가 모듈 임포트 실패, 함수만 발췌: {e}", flush=True)
    ev = None

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(0)
init = suite.get_task_init_states(0)
bddl = os.path.join(
    __import__("libero.libero", fromlist=["get_libero_path"]).get_libero_path("bddl_files"),
    task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(**{"bddl_file_name": bddl, "camera_heights": 128, "camera_widths": 128})
env.seed(7)
env.reset()

def controllers(e):
    base = e
    for _ in range(8):
        if hasattr(base, "robots"):
            break
        base = getattr(base, "env", base)
    out = []
    for r in getattr(base, "robots", []):
        c = getattr(r, "controller", None)
        if c is not None:
            out.append(("robot.controller", c))
        cc = getattr(r, "composite_controller", None)
        if cc is not None:
            pc = getattr(cc, "part_controllers", None) or getattr(cc, "controllers", None) or {}
            for k, v in (pc.items() if hasattr(pc, "items") else enumerate(pc)):
                out.append((f"composite[{k}]", v))
    return out

cs = controllers(env)
print(f"\n[probe] 발견된 컨트롤러 {len(cs)}개")
for name, c in cs:
    im = getattr(c, "input_max", None)
    print(f"  {name:24s} {type(c).__name__:22s} input_max={np.asarray(im).ravel()[:3] if im is not None else None}"
          f" action_scale={'cached' if getattr(c,'action_scale',None) is not None else 'None'}")

if ev is not None and hasattr(ev, "_patch_clip_bounds"):
    n = ev._patch_clip_bounds(env, 3.0)
    print(f"\n[probe] _patch_clip_bounds(env, 3.0) -> {n} 개 패치")
    for name, c in controllers(env):
        im = getattr(c, "input_max", None)
        print(f"  {name:24s} input_max={np.asarray(im).ravel()[:3] if im is not None else None}"
              f" action_scale={'cached' if getattr(c,'action_scale',None) is not None else 'None'}")

# 실제로 잘리는지: ±1 을 넘는 명령을 넣고 컨트롤러가 받은 값을 본다
print("\n[probe] scale_action 이 실제로 자르는가")
for name, c in controllers(env):
    if not hasattr(c, "scale_action"):
        continue
    dim = np.asarray(getattr(c, "input_max")).size
    big = np.full(dim, 2.5)
    try:
        out = c.scale_action(big.copy())
        lin = (big - np.asarray(c.action_input_transform)) * np.asarray(c.action_scale) \
              + np.asarray(c.action_output_transform)
        clipped = not np.allclose(out, lin)
        print(f"  {name:24s} 입력 2.5 -> {'잘림' if clipped else '안 잘림'} "
              f"(out[0]={np.asarray(out).ravel()[0]:.4f}, 선형이면 {np.asarray(lin).ravel()[0]:.4f})")
    except Exception as e:
        print(f"  {name:24s} scale_action 실패: {e}")
env.close()
