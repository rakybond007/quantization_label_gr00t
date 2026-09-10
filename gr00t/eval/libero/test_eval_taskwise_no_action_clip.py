"""AST-level tests keep this lightweight: importing the eval client needs LIBERO."""
import ast
import types
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).with_name("eval_taskwise_gr00t.py")


def _helpers():
    tree = ast.parse(SOURCE.read_text())
    names = {"_iter_controllers", "_libero_delta_arm_controllers", "_finite_unclipped_scale_action",
             "patch_no_action_clip", "action_clip_status"}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
                        type_ignores=[])
    scope = {"np": np, "types": types}
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope


class _Arm:
    use_delta = True
    def __init__(self):
        self.input_min = np.full(6, -1.0)
        self.input_max = np.full(6, 1.0)
        self.output_min = np.full(6, -0.05)
        self.output_max = np.full(6, 0.05)
        self.action_scale = None
        self.action_input_transform = None
        self.action_output_transform = None

    # Exact affine/clipping structure of robosuite-1.4.1 base_controller.py.
    def scale_action(self, action):
        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        action = np.clip(action, self.input_min, self.input_max)
        return (action - self.action_input_transform) * self.action_scale + self.action_output_transform


_Arm.__module__ = "robosuite.controllers.osc"


class _Gripper(_Arm):
    pass


_Gripper.__module__ = "robosuite.controllers.gripper"


class _Robot:
    def __init__(self):
        self.controller = _Arm()
        self.composite_controller = type("Composite", (), {"part_controllers": {"gripper": _Gripper()}})()


class _Env:
    def __init__(self): self.robots = [_Robot()]


def test_actual_eval_ast_scaler_is_finite_unclipped_arm_only_and_reset_safe():
    h = _helpers(); env = _Env(); arm, gripper = env.robots[0].controller, env.robots[0].composite_controller.part_controllers["gripper"]
    in_range = np.array([.5, 0, 0, 0, 0, 0.])
    large = np.array([40.7, 0, 0, 0, 0, 0.])
    native_in_range, native_large = arm.scale_action(in_range), arm.scale_action(large)
    assert h["patch_no_action_clip"](env) == 1
    assert np.allclose(arm.scale_action(in_range), native_in_range)
    assert native_large[0] == .05
    assert np.isclose(arm.scale_action(large)[0], 2.035)
    assert np.isfinite(arm.scale_action(large)).all()
    assert gripper.scale_action(large)[0] == .05
    status = h["action_clip_status"](env)
    assert len(status) == 1 and status[0]["no_action_clip"] and status[0]["finite"]
    # A hard reset creates another v1.4.1 controller; another patch restores mode.
    env.robots[0].controller = _Arm()
    assert h["patch_no_action_clip"](env) == 1
    assert np.isclose(env.robots[0].controller.scale_action(large)[0], 2.035)
