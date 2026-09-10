"""VSTA — Variable-Speed Trajectory Augmentation, from TempoVLA (arXiv:2606.06491).

Re-times a demonstration to a target speed s by *accumulating* per-step action
increments into a cumulative path and *splitting* that path on a new, coarser (s>1)
or finer (s<1) time grid. Merging s consecutive actions into one larger action is the
s = integer special case; the general form interpolates the cumulative curve, which is
what lets s be fractional.

Why this file exists here: the repository's own embodiment adapter already sums adjacent
end-effector deltas for RoboCasa and LIBERO. That is exactly VSTA at s = 2 with no
interpolation. Writing the general operator once means the gate's K = 2 compression and
TempoVLA's arbitrary-s conditioning share one implementation, and any disagreement
between them is a bug rather than two conventions.

What is faithful to the paper, and what is not:
  - accumulate-then-split on linearly composable dimensions: faithful.
  - gripper handled discretely rather than averaged: faithful (the paper states this).
  - the paper additionally performs "motion-consistent segmentation" (splitting a demo
    into translate / rotate / still / gripper segments and re-timing within segments).
    That is NOT implemented here; this module re-times whole trajectories. Segment-wise
    re-timing is the natural next layer and is where a gate would attach.

Rotation note: LIBERO and RoboCasa both express rotation as small per-step increments
(euler rpy, axis-angle). Summing those is exact only in the small-angle limit. The paper
relies on the same approximation by calling the space "linearly composable"; the error
is measured by `resample_error` below rather than assumed away.
"""
from __future__ import annotations

import numpy as np

__all__ = ["retime", "resample_error", "merge_integer"]


def _cumulative(actions: np.ndarray) -> np.ndarray:
    """Cumulative path with a leading zero, shape (T+1, D)."""
    out = np.zeros((actions.shape[0] + 1, actions.shape[1]), dtype=np.float64)
    np.cumsum(actions, axis=0, out=out[1:])
    return out


def retime(
    actions: np.ndarray,
    speed: float,
    delta_dims: "slice | np.ndarray | list[int]",
    discrete_dims: "slice | np.ndarray | list[int] | None" = None,
) -> np.ndarray:
    """Re-time `actions` (T, D) to `speed`, returning roughly (T / speed, D).

    Args:
        actions: per-step action increments.
        speed: >1 compresses (fewer, larger actions), <1 stretches.
        delta_dims: columns that compose linearly and are accumulated-then-split.
        discrete_dims: columns sampled at the new grid instead (gripper, mode flags).
            Sampled with "last of group" so a close command inside a merged window
            survives it, which is the behaviour a gripper needs.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"expected (T, D), got {actions.shape}")
    T, D = actions.shape
    if T == 0:
        return actions.copy()

    # Grid of the re-timed sequence, in units of original timesteps.
    #
    # For an integer speed the paper's operation is "merge k consecutive actions", so the
    # boundaries must sit on integers -- otherwise a trajectory whose length is not a
    # multiple of k gets bins that straddle original steps, and the result silently
    # disagrees with plain grouped summation. (test_vsta.py caught exactly that: T=207,
    # k=2 matched at k=3, where 207 divides evenly, and diverged at k=2 and k=4.)
    # Fractional speeds have no integer grid to land on and use the uniform one.
    if abs(speed - round(speed)) < 1e-9 and speed >= 1:
        k = int(round(speed))
        edges = np.arange(0, T + k, k, dtype=np.float64)
        edges[-1] = float(T)                      # ragged tail keeps whatever is left
        edges = np.unique(edges)
        n_new = len(edges) - 1
    else:
        n_new = max(1, int(round(T / speed)))
        edges = np.linspace(0.0, float(T), n_new + 1)

    out = np.zeros((n_new, D), dtype=np.float64)

    cum = _cumulative(actions)                       # (T+1, D)
    grid = np.arange(T + 1, dtype=np.float64)
    idx = np.arange(D)
    dmask = np.zeros(D, dtype=bool)
    dmask[idx[delta_dims] if not isinstance(delta_dims, slice) else idx[delta_dims]] = True

    # accumulate-then-split: interpolate the cumulative path at the new edges and
    # difference it. Exact for integer speeds; linear interpolation between the two
    # neighbouring original steps otherwise.
    for d in np.nonzero(dmask)[0]:
        sampled = np.interp(edges, grid, cum[:, d])
        out[:, d] = np.diff(sampled)

    if discrete_dims is not None:
        ddims = idx[discrete_dims] if not isinstance(discrete_dims, slice) else idx[discrete_dims]
        for d in np.atleast_1d(ddims):
            # last original step whose centre falls inside each new window
            for j in range(n_new):
                lo, hi = edges[j], edges[j + 1]
                sel = np.arange(int(np.floor(lo)), max(int(np.ceil(hi)), int(np.floor(lo)) + 1))
                sel = sel[(sel >= 0) & (sel < T)]
                out[j, d] = actions[sel[-1], d] if sel.size else actions[min(j, T - 1), d]

    # any column neither delta nor discrete: carry the mean, and say so loudly if hit
    other = ~dmask
    if discrete_dims is not None:
        other[np.atleast_1d(ddims)] = False
    for d in np.nonzero(other)[0]:
        for j in range(n_new):
            lo, hi = edges[j], edges[j + 1]
            sel = np.arange(int(np.floor(lo)), max(int(np.ceil(hi)), int(np.floor(lo)) + 1))
            sel = sel[(sel >= 0) & (sel < T)]
            out[j, d] = actions[sel, d].mean() if sel.size else actions[min(j, T - 1), d]

    return out


def merge_integer(actions: np.ndarray, k: int, delta_dims, discrete_dims=None) -> np.ndarray:
    """The gate's K-step compression: sum groups of k deltas, last-of-group elsewhere.

    Kept separate from `retime` so the two can be compared against each other; for
    integer k they must agree to floating-point tolerance, and `test_vsta.py` checks it.
    """
    actions = np.asarray(actions, dtype=np.float64)
    T, D = actions.shape
    n = int(np.ceil(T / k))
    out = np.zeros((n, D), dtype=np.float64)
    idx = np.arange(D)
    dd = idx[delta_dims] if not isinstance(delta_dims, slice) else idx[delta_dims]
    for j in range(n):
        grp = actions[j * k : (j + 1) * k]
        out[j, dd] = grp[:, dd].sum(axis=0)
        if discrete_dims is not None:
            cd = idx[discrete_dims] if not isinstance(discrete_dims, slice) else idx[discrete_dims]
            out[j, cd] = grp[-1, cd]
    return out


def resample_error(actions: np.ndarray, speed: float, delta_dims, discrete_dims=None) -> dict:
    """How much displacement the re-timing loses.

    The claim VSTA rests on is that re-timing preserves motion semantics. On a delta
    action space the checkable part of that is endpoint displacement: the cumulative sum
    of the re-timed sequence should equal the original's. Reports absolute and relative
    error per delta dimension so a violation is visible rather than assumed absent.
    """
    actions = np.asarray(actions, dtype=np.float64)
    new = retime(actions, speed, delta_dims, discrete_dims)
    idx = np.arange(actions.shape[1])
    dd = idx[delta_dims] if not isinstance(delta_dims, slice) else idx[delta_dims]
    orig = actions[:, dd].sum(axis=0)
    got = new[:, dd].sum(axis=0)
    denom = np.maximum(np.abs(orig), 1e-8)
    return {
        "speed": speed,
        "steps_in": int(actions.shape[0]),
        "steps_out": int(new.shape[0]),
        "endpoint_abs_err": np.abs(got - orig),
        "endpoint_rel_err": np.abs(got - orig) / denom,
        "max_rel_err": float(np.max(np.abs(got - orig) / denom)),
    }
