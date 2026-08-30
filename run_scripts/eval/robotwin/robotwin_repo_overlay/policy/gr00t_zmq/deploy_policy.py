"""RoboTwin 2.0 policy adapter for our GR00T-N1.5 zmq inference server.

eval_policy.py imports this module by --policy_name=gr00t_zmq.
Contract (matches policy/pi0/deploy_policy.py):
  get_model(usr_args)   -> model object
  eval(TASK_ENV, model, observation) -> runs one action chunk
  reset_model(model)    -> resets per-episode state
"""

from collections import deque
from io import BytesIO

import numpy as np
import torch
import zmq


class _ZmqInferenceClient:
    """Minimal inline copy of gr00t.eval.service.ExternalRobotInferenceClient.

    Avoids importing the gr00t package (which pulls decord via
    gr00t.data.dataset -> gr00t.utils.video) so this adapter runs in any
    env that just has torch + pyzmq.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _to_bytes(self, data: dict) -> bytes:
        buf = BytesIO()
        torch.save(data, buf)
        return buf.getvalue()

    def _from_bytes(self, data: bytes):
        buf = BytesIO(data)
        return torch.load(buf, weights_only=False)

    def call_endpoint(self, endpoint: str, data=None, requires_input: bool = True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.socket.send(self._to_bytes(req))
        msg = self.socket.recv()
        if msg == b"ERROR":
            raise RuntimeError("Server error")
        return self._from_bytes(msg)

    def get_action(self, observations: dict) -> dict:
        return self.call_endpoint("get_action", observations)


class GR00TZmqClient:
    """Thin wrapper: holds zmq client + per-episode state.

    Observation layout sent to server (matches RoboTwinAgilexDataConfig):
        video.cam_high          (1, H, W, 3) uint8
        video.cam_left_wrist    (1, H, W, 3) uint8
        video.cam_right_wrist   (1, H, W, 3) uint8
        state.left_joints       (1, 6)  float
        state.left_gripper      (1, 1)  float
        state.right_joints      (1, 6)  float
        state.right_gripper     (1, 1)  float
        annotation.human.action.task_description  list[str]

    Action chunk decoded from server response (action_keys reassembled into
    a single 14-dim qpos vector in joint order:
        [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, action_horizon: int = 16):
        self.client = _ZmqInferenceClient(host=host, port=port)
        self.action_horizon = int(action_horizon)
        self.task_description: str | None = None
        self.action_chunk: np.ndarray | None = None  # (H, 14)
        self.chunk_step: int = 0

    def reset(self):
        self.task_description = None
        self.action_chunk = None
        self.chunk_step = 0


def _resize_to_train(img, target_wh=(640, 480)):
    # cv2.resize expects (W, H). Training pipeline expects (480, 640) shape
    # (H=480, W=640). RoboTwin sim default D435 head/wrist is (240, 320).
    import cv2
    if img.shape[1] == target_wh[0] and img.shape[0] == target_wh[1]:
        return img
    return cv2.resize(img, target_wh, interpolation=cv2.INTER_AREA)


def _encode_obs(observation, instruction: str):
    head = _resize_to_train(np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8))
    left = _resize_to_train(np.asarray(observation["observation"]["left_camera"]["rgb"], dtype=np.uint8))
    right = _resize_to_train(np.asarray(observation["observation"]["right_camera"]["rgb"], dtype=np.uint8))
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)  # (14,)
    return {
        "video.cam_high":        head[None],
        "video.cam_left_wrist":  left[None],
        "video.cam_right_wrist": right[None],
        "state.left_joints":   state[0:6][None],
        "state.left_gripper":  state[6:7][None],
        "state.right_joints":  state[7:13][None],
        "state.right_gripper": state[13:14][None],
        "annotation.human.action.task_description": [str(instruction)],
    }


def _decode_action(action_dict):
    # Server may return gripper as (H,) instead of (H, 1); reshape defensively.
    lj = np.asarray(action_dict["action.left_joints"]).reshape(-1, 6)
    lg = np.asarray(action_dict["action.left_gripper"]).reshape(-1, 1)
    rj = np.asarray(action_dict["action.right_joints"]).reshape(-1, 6)
    rg = np.asarray(action_dict["action.right_gripper"]).reshape(-1, 1)
    H = lj.shape[0]
    out = np.zeros((H, 14), dtype=np.float32)
    out[:, 0:6]   = lj
    out[:, 6:7]   = lg
    out[:, 7:13]  = rj
    out[:, 13:14] = rg
    return out


def get_model(usr_args):
    host = usr_args.get("host", "127.0.0.1")
    port = int(usr_args.get("port", 5555))
    action_horizon = int(usr_args.get("action_horizon", 16))
    return GR00TZmqClient(host=host, port=port, action_horizon=action_horizon)


def encode_obs(observation):
    return observation


def reset_model(model: GR00TZmqClient):
    model.reset()


def eval(TASK_ENV, model: GR00TZmqClient, observation):
    if model.task_description is None:
        model.task_description = str(TASK_ENV.get_instruction())

    obs = _encode_obs(observation, model.task_description)
    action_dict = model.client.get_action(obs)
    chunk = _decode_action(action_dict)  # (H, 14)
    H = min(model.action_horizon, len(chunk))

    for i in range(H):
        TASK_ENV.take_action(chunk[i])
        observation = TASK_ENV.get_obs()
