"""Isolated SmallGate copy with current state and strictly past executed commands.

Visual trunk copied from scripts/train_gate_module.py:SmallGate. No imports from
the mutable production implementation. Inputs use LeRobot raw column ordering.
"""
from collections import deque

import numpy as np
import torch
from torch import nn


def pack_history(actions, frame, history=16):
    """Observation at frame f precedes command a[f]; only a[:f] is legal.

    Left padding has mask zero, valid commands are oldest to newest. Offline
    commands are demonstrations, NOT reconstructed outputs of the GR00T policy.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or not 0 <= frame <= len(actions) or history < 1:
        raise ValueError("invalid action array, frame, or history")
    out = np.zeros((history, actions.shape[1]), dtype=np.float32)
    mask = np.zeros(history, dtype=np.float32)
    count = min(frame, history)
    if count:
        out[-count:] = actions[frame-count:frame]
        mask[-count:] = 1
    return out, mask


class ExecutedActionHistory:
    """Client-owned per-environment ring buffer. Append AFTER env.step succeeds.

    Append the actual post-compression command in the same 12-D convention as
    the training data, not an unexecuted planned chunk. Reset on episode reset.
    Compression changes the command distribution; a closed-loop evaluation is
    still required. This class does not change any production evaluation client.
    """
    def __init__(self, history=16, action_dim=12):
        self.history, self.action_dim = history, action_dim
        self.rows = deque(maxlen=history)

    def reset(self):
        self.rows.clear()

    def append_executed(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("executed command has invalid shape or values")
        self.rows.append(action.copy())

    def arrays(self):
        rows = np.asarray(list(self.rows), dtype=np.float32).reshape(-1, self.action_dim)
        return pack_history(rows, len(rows), self.history)


class SmallGateProprioHistory(nn.Module):
    def __init__(self, state_dim=53, action_dim=12, history=16, text_dim=384, width=32):
        super().__init__()
        self.config = dict(state_dim=state_dim, action_dim=action_dim,
                           history=history, text_dim=text_dim, width=width)
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(9, width), blk(width, width*2),
                                 blk(width*2, width*4), blk(width*4, width*4),
                                 nn.AdaptiveAvgPool2d(1))
        self.motion = nn.Sequential(nn.Linear(state_dim + history*(action_dim+1), 64),
                                    nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width*4 + text_dim + 64, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def set_normalization(self, states, actions, mask):
        """Fit on training rows ONLY; action padding is excluded from statistics."""
        for prefix, values in (("state", states), ("action", actions[mask.astype(bool)])):
            if not len(values) or not np.isfinite(values).all():
                raise ValueError("normalization requires finite training data")
            mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = values.std(axis=0, dtype=np.float64).astype(np.float32)
            std = np.where(std < 1e-5, 1.0, std)
            getattr(self, prefix+"_mean").copy_(torch.from_numpy(mean))
            getattr(self, prefix+"_std").copy_(torch.from_numpy(std))

    def forward(self, images, text, state, previous_actions, history_mask):
        c = self.config
        b = images.shape[0]
        if state.shape != (b, c["state_dim"]) or text.shape != (b, c["text_dim"]):
            raise ValueError("state/text shape mismatch")
        if previous_actions.shape != (b, c["history"], c["action_dim"]):
            raise ValueError("previous_actions shape mismatch")
        if history_mask.shape != (b, c["history"]):
            raise ValueError("history_mask shape mismatch")
        s = (state-self.state_mean)/self.state_std
        a = ((previous_actions-self.action_mean)/self.action_std)*history_mask.unsqueeze(-1)
        motion = self.motion(torch.cat((s, a.flatten(1), history_mask), dim=1))
        visual = self.net(images).flatten(1)
        return self.head(torch.cat((visual, text, motion), dim=1))


class SmallGateBaseline(nn.Module):
    """Unmodified architecture copied for same-process timing, not retrained here."""
    def __init__(self, text_dim=384, width=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(9,width), blk(width,width*2), blk(width*2,width*4),
                                 blk(width*4,width*4), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(width*4+text_dim,128), nn.ReLU(),
                                  nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))

    def forward(self, images, text, state=None, previous_actions=None, history_mask=None):
        """The extra arguments are accepted and ignored.

        The comparison has to differ in exactly one thing -- the inputs. Sharing
        the call signature lets both arms run through the same loader, the same
        split, the same loss and the same checkpoint selection, so a difference
        in the numbers is the inputs and not the harness.
        """
        return self.head(torch.cat((self.net(images).flatten(1), text), dim=1))
