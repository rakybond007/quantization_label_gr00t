"""SimplerEnv gym wrappers adapted to Isaac-GR00T's SimplerEnv data configs.

Key schema differs from the AlinVLA reference:
  Fractal/Google: state.x/y/z/rx/ry/rz/rw/gripper (8) ; action.x/y/z/roll/pitch/yaw/gripper (7)
  Bridge/WidowX:  state.x/y/z/roll/pitch/yaw/pad/gripper (8) ; action.x/y/z/roll/pitch/yaw/gripper (7)

Both are split-scalar (one key per dim) so they match data_config.py's
SimplerEnvFractalDataConfig / SimplerEnvBridgeDataConfig.
"""
import os
from collections import OrderedDict

import cv2
import gymnasium as gym
import numpy as np
import simpler_env
from gymnasium.envs.registration import register
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from transforms3d import euler as te, quaternions as tq


def _scalar_box(low, high):
    return gym.spaces.Box(low=np.float32(low), high=np.float32(high), shape=(), dtype=np.float32)


class GoogleFractalEnv(gym.Env):
    """Google/Fractal env. Uses scalar state keys + 7D action with quat-style state."""

    STATE_KEYS = ["state.x", "state.y", "state.z",
                  "state.rx", "state.ry", "state.rz", "state.rw",
                  "state.gripper"]
    ACTION_KEYS = ["action.x", "action.y", "action.z",
                   "action.roll", "action.pitch", "action.yaw",
                   "action.gripper"]
    VIDEO_KEY = "video.image"

    def __init__(self, env_name: str, image_size: tuple[int, int]):
        os.environ["DISPLAY"] = ""
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        env = simpler_env.make(env_name)
        env._max_episode_steps = 10000
        self.env = env
        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(OrderedDict([
            (self.VIDEO_KEY, gym.spaces.Box(low=0, high=255,
                shape=(image_size[0], image_size[1], 3), dtype=np.uint8)),
            ("state.x", _scalar_box(obs_low[0], obs_high[0])),
            ("state.y", _scalar_box(obs_low[1], obs_high[1])),
            ("state.z", _scalar_box(obs_low[2], obs_high[2])),
            ("state.rx", _scalar_box(-1.0, 1.0)),
            ("state.ry", _scalar_box(-1.0, 1.0)),
            ("state.rz", _scalar_box(-1.0, 1.0)),
            ("state.rw", _scalar_box(-1.0, 1.0)),
            ("state.gripper", _scalar_box(0.0, 1.0)),
            ("annotation.human.action.task_description", gym.spaces.Text(max_length=512)),
        ]))
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(OrderedDict([
            ("action.x", _scalar_box(action_low[0], action_high[0])),
            ("action.y", _scalar_box(action_low[1], action_high[1])),
            ("action.z", _scalar_box(action_low[2], action_high[2])),
            ("action.roll", _scalar_box(action_low[3], action_high[3])),
            ("action.pitch", _scalar_box(action_low[4], action_high[4])),
            ("action.yaw", _scalar_box(action_low[5], action_high[5])),
            ("action.gripper", _scalar_box(action_low[6], action_high[6])),
        ]))
        self.image_size = image_size
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

    def reset(self, seed=None, options=None):
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        return observation, info

    def step(self, action):
        action_vector = np.array([
            float(action["action.x"]),
            float(action["action.y"]),
            float(action["action.z"]),
            float(action["action.roll"]),
            float(action["action.pitch"]),
            float(action["action.yaw"]),
            self._postprocess_gripper(float(action["action.gripper"])),
        ], dtype=np.float32)
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]   # length 8: x,y,z, qx,qy,qz,qw, gripper
        return {
            self.VIDEO_KEY: cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.x":  np.float32(proprio[0]),
            "state.y":  np.float32(proprio[1]),
            "state.z":  np.float32(proprio[2]),
            "state.rx": np.float32(proprio[3]),
            "state.ry": np.float32(proprio[4]),
            "state.rz": np.float32(proprio[5]),
            "state.rw": np.float32(proprio[6]),
            "state.gripper": np.float32(proprio[7]),
            "annotation.human.action.task_description": self.env.unwrapped.get_language_instruction(),
        }

    def _postprocess_gripper(self, gv: float):
        # Same sticky-gripper logic as AlinVLA reference (Fractal-specific).
        if self.previous_gripper_action is None:
            self.previous_gripper_action = gv
        relative_gripper_action = self.previous_gripper_action - gv
        self.previous_gripper_action = gv
        if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        return relative_gripper_action


class WidowXBridgeEnv(gym.Env):
    """WidowX/Bridge env. State has a `pad` placeholder dim matching dataset schema."""

    STATE_KEYS = ["state.x", "state.y", "state.z",
                  "state.roll", "state.pitch", "state.yaw",
                  "state.pad",
                  "state.gripper"]
    ACTION_KEYS = ["action.x", "action.y", "action.z",
                   "action.roll", "action.pitch", "action.yaw",
                   "action.gripper"]
    VIDEO_KEY = "video.image_0"

    def __init__(self, env_name: str, image_size: tuple[int, int]):
        env = simpler_env.make(env_name)
        env._max_episode_steps = 10000
        self.env = env
        obs_low = env.observation_space["agent"]["eef_pos"].low
        obs_high = env.observation_space["agent"]["eef_pos"].high
        self.observation_space = gym.spaces.Dict(OrderedDict([
            (self.VIDEO_KEY, gym.spaces.Box(low=0, high=255,
                shape=(image_size[0], image_size[1], 3), dtype=np.uint8)),
            ("state.x", _scalar_box(obs_low[0], obs_high[0])),
            ("state.y", _scalar_box(obs_low[1], obs_high[1])),
            ("state.z", _scalar_box(obs_low[2], obs_high[2])),
            ("state.roll",  _scalar_box(-np.pi, np.pi)),
            ("state.pitch", _scalar_box(-np.pi, np.pi)),
            ("state.yaw",   _scalar_box(-np.pi, np.pi)),
            ("state.pad",   _scalar_box(0.0, 0.0)),
            ("state.gripper", _scalar_box(0.0, 1.0)),
            ("annotation.human.action.task_description", gym.spaces.Text(max_length=512)),
        ]))
        action_low = env.action_space.low
        action_high = env.action_space.high
        self.action_space = gym.spaces.Dict(OrderedDict([
            ("action.x", _scalar_box(action_low[0], action_high[0])),
            ("action.y", _scalar_box(action_low[1], action_high[1])),
            ("action.z", _scalar_box(action_low[2], action_high[2])),
            ("action.roll",  _scalar_box(action_low[3], action_high[3])),
            ("action.pitch", _scalar_box(action_low[4], action_high[4])),
            ("action.yaw",   _scalar_box(action_low[5], action_high[5])),
            ("action.gripper", _scalar_box(action_low[6], action_high[6])),
        ]))
        self.image_size = image_size
        self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset()
        observation = self._process_observation(observation)
        info["success"] = False
        return observation, info

    def step(self, action):
        action_vector = np.array([
            float(action["action.x"]),
            float(action["action.y"]),
            float(action["action.z"]),
            float(action["action.roll"]),
            float(action["action.pitch"]),
            float(action["action.yaw"]),
            self._postprocess_gripper(float(action["action.gripper"])),
        ], dtype=np.float32)
        observation, reward, done, truncated, info = self.env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = done
        return observation, reward, done, truncated, info

    def _process_observation(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = obs["agent"]["eef_pos"]   # length 8: x,y,z, qx,qy,qz,qw, gripper
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge = te.mat2euler(rm_bridge @ self.default_rot.T)
        return {
            self.VIDEO_KEY: cv2.resize(img, (self.image_size[1], self.image_size[0])),
            "state.x":  np.float32(proprio[0]),
            "state.y":  np.float32(proprio[1]),
            "state.z":  np.float32(proprio[2]),
            "state.roll":  np.float32(rpy_bridge[0]),
            "state.pitch": np.float32(rpy_bridge[1]),
            "state.yaw":   np.float32(rpy_bridge[2]),
            "state.pad":   np.float32(0.0),
            "state.gripper": np.float32(proprio[7]),
            "annotation.human.action.task_description": self.env.unwrapped.get_language_instruction(),
        }

    def _postprocess_gripper(self, gv: float):
        # Trained with [0, 1], 0 close, 1 open -> SimplerEnv expects [-1, 1]
        return 2.0 * (gv > 0.5) - 1.0


def register_simpler_envs():
    for env_name in [
        "google_robot_pick_coke_can",
        "google_robot_pick_object",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_in_closed_drawer",
    ]:
        register(
            id=f"simpler_env_google/{env_name}",
            entry_point="gr00t.eval.sim.SimplerEnv.simpler_env:GoogleFractalEnv",
            # Match training data: fractal20220817_data_lerobot has (H=224, W=256).
            kwargs={"env_name": env_name, "image_size": (224, 256)},
        )

    for env_name in [
        "widowx_spoon_on_towel",
        "widowx_carrot_on_plate",
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
        "widowx_put_eggplant_in_sink",
        "widowx_open_drawer",
        "widowx_close_drawer",
    ]:
        register(
            id=f"simpler_env_widowx/{env_name}",
            entry_point="gr00t.eval.sim.SimplerEnv.simpler_env:WidowXBridgeEnv",
            kwargs={"env_name": env_name, "image_size": (256, 256)},
        )
