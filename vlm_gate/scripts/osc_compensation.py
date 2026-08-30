"""Model-based inverse compensation for merged (time-quantized) OSC deltas.

Uses the ARX(1) response model identified by sysid_osc_response.py:

    dx_t = a * v_prev + b * u_t  (+ b2 * u_t|u_t|/OUTPUT_MAX if quadratic fit)

with dx, v, u in meters (u = clip(d, -1, 1) * OUTPUT_MAX).

ModelCompensator: given the observed EE velocity proxy v_now (last per-step
displacement, m/step) and the block's fine deltas (action units), it
(1) rolls the model forward over the fine deltas to get the displacement the
fine execution would have achieved, then (2) inverts the one-step model to
find the single merged command d' that achieves that displacement:

    u' = (target - a * v_now) / b       (linear case)
    d' = clip(u' / OUTPUT_MAX, -1, 1)   (clipped amount is reported)

ScalarCompensator (ablation): d' = c * sum(fine deltas).
"""
import json
import numpy as np

OUTPUT_MAX = 0.05
AXIS_ORDER = "xyz"


def load_sysid(path):
    with open(path) as f:
        return json.load(f)


def _axis_coeffs(sysid):
    """Per-axis (a, b, b2) arrays, falling back to pooled fit."""
    pooled = sysid.get("pooled", {})
    a = np.zeros(3)
    b = np.zeros(3)
    b2 = np.zeros(3)
    for i, ax in enumerate(AXIS_ORDER):
        f = sysid.get("per_axis", {}).get(ax) or pooled
        a[i] = f.get("a", pooled.get("a", 0.0))
        b[i] = f.get("b", pooled.get("b", 1.0))
        b2[i] = f.get("b2", 0.0)
    return a, b, b2


class ModelCompensator:
    """Inverse compensation with a VIRTUAL fine-path state.

    The fine target must be rolled out from the velocity the FINE execution
    would have had (self.v_f, model-propagated), not from the observed merged
    velocity: using the observed v for the target couples target inflation to
    the merged path's own (higher) velocity — positive feedback that diverges
    for a + a^2 near 1 (a≈0.55 here; empirically 150-200% overshoot). Observed
    v_now enters only the inversion term (-a*v_now), which is negative
    feedback. Call begin_chunk(v_now) when a fresh policy chunk arrives (the
    policy re-observes, so the fine path restarts from the true state)."""

    def __init__(self, sysid, clip_limit=1.0):
        if isinstance(sysid, str):
            sysid = load_sysid(sysid)
        self.a, self.b, self.b2 = _axis_coeffs(sysid)
        self.v_f = None
        self.clip_limit = float(clip_limit)

    def begin_chunk(self, v_now):
        self.v_f = np.asarray(v_now, dtype=float).copy()

    def _forward(self, v, u):
        """One-step model, all (3,) arrays in meters."""
        return self.a * v + self.b * u + self.b2 * u * np.abs(u) / OUTPUT_MAX

    def _invert(self, rhs):
        """Solve b*u + b2*u|u|/OM = rhs per axis (odd in u -> solve |rhs|)."""
        u = np.empty(3)
        for i in range(3):
            r, b, q = abs(float(rhs[i])), self.b[i], self.b2[i] / OUTPUT_MAX
            if abs(q) < 1e-9:
                m = r / b if b != 0 else 0.0
            else:
                disc = b * b + 4.0 * q * r
                m = (-b + np.sqrt(max(disc, 0.0))) / (2.0 * q) if disc >= 0 else r / b
            u[i] = np.sign(rhs[i]) * m
        return u

    def merged_command(self, v_now, fine_deltas):
        """v_now: (3,) m/step observed. fine_deltas: (k,3) action units.
        Returns (d_prime (3,) action units clipped, clip_excess (3,))."""
        v_now = np.asarray(v_now, dtype=float)
        if self.v_f is None:
            self.v_f = v_now.copy()
        v = self.v_f.copy()
        target = np.zeros(3)
        for d in np.asarray(fine_deltas, dtype=float).reshape(-1, 3):
            u = np.clip(d, -1, 1) * OUTPUT_MAX
            dx = self._forward(v, u)
            target += dx
            v = dx
        self.v_f = v  # fine-path state advances by the model, not by observation
        u_prime = self._invert(target - self.a * v_now)
        d_prime = u_prime / OUTPUT_MAX
        clipped = np.clip(d_prime, -self.clip_limit, self.clip_limit)
        return clipped, d_prime - clipped


class ScalarCompensator:
    def __init__(self, c, clip_limit=1.0):
        self.c = float(c); self.clip_limit = float(clip_limit)

    def begin_chunk(self, v_now):
        pass

    def merged_command(self, v_now, fine_deltas):
        d_sum = np.asarray(fine_deltas, dtype=float).reshape(-1, 3).sum(axis=0)
        d_prime = self.c * d_sum
        clipped = np.clip(d_prime, -self.clip_limit, self.clip_limit)
        return clipped, d_prime - clipped


def make_compensator(mode, sysid_json=None, scalar_c=None, clip_limit=1.0):
    if mode == "model":
        return ModelCompensator(sysid_json, clip_limit)
    if mode == "scalar":
        if scalar_c is None and sysid_json:
            rr = load_sysid(sysid_json).get("implied_K2_reach_rate")
            scalar_c = 1.0 / rr if rr else 1.0
        return ScalarCompensator(scalar_c if scalar_c is not None else 1.0, clip_limit)
    return None
