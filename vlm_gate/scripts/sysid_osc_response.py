"""System identification of the robocasa OSC end-effector translation response.

Motivation (see probe_quantization_dynamics.py): the robosuite OSC uses
achieved-mode goal re-anchoring + a non-convergent PD, so a merged (summed)
delta command reaches only ~89% of the displacement that the same deltas
executed sequentially would. To invert-compensate merged commands we fit a
per-axis 1-step response model in displacement space:

    dx_t = a * v_prev + b * u_t          (ARX(1))

where
    dx_t   : EE displacement achieved during env step t            [m]
    v_prev : EE displacement of the previous env step (= velocity  [m/step]
             proxy observable from state at policy rate)
    u_t    : commanded delta in METERS = clip(d, -1, 1) * output_max(0.05)

Data collection: from identical resets, inject scripted delta sequences on
one axis at a time — constant runs of varying magnitude/sign, random walks,
and zero-delta (settle) steps that isolate the `a` term. All commands stay
inside the [-1, 1] controller clip so the fit region is the linear regime.

Output: JSON with per-axis and pooled (a, b, R^2); if pooled R^2 < 0.8 a
quadratic term b2 * u|u| is added and refit.

Run inside the `quant_gate_eval` conda env (same as the robocasa clients).
"""
import argparse
import json
import numpy as np

from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env

OUTPUT_MAX = 0.05  # m per env step at command=1.0 (robosuite OSC output_max)
AXES = {"x": 0, "y": 1, "z": 2}


def make_env(env_name, seed):
    env = load_robocasa_gym_env(
        env_name, seed=seed, robots="PandaOmron", camera_widths=256,
        camera_heights=256, render_onscreen=False, obj_instance_split="A",
        generative_textures=None, randomize_cameras=False,
        layout_ids=None, style_ids=None, collect_data=False)
    return RoboCasaWrapper(env)


def rel_pos(obs):
    """EE position in the robot BASE frame (state.end_effector_position_relative).
    The OSC delta command acts in this frame — measuring displacement in the
    world frame aliases axes whenever the base yaw != 0 (observed: action-x
    moved world-y), which destroys the per-axis fit."""
    return np.asarray(obs["state.end_effector_position_relative"], dtype=float).reshape(3)


def build_sequences(rng):
    """Delta-command sequences (action units, clip=1). Each stays mean-reverting
    so the arm remains in free space. Zero steps identify the decay term a."""
    seqs = []
    mags = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for m in mags:
        # constant run +, settle, constant run -, settle (net ~0 drift)
        seqs.append([m] * 4 + [0.0] * 3 + [-m] * 4 + [0.0] * 3)
    for m in [0.1, 0.25, 0.5]:
        # alternating (high velocity-reversal excitation)
        seqs.append([m, -m] * 5)
    for _ in range(4):
        # bounded random walk with occasional zeros
        s, acc = [], 0.0
        for _ in range(16):
            d = float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.5]) * rng.choice([-1, 1]))
            if abs(acc + d) > 1.2:
                d = -np.sign(acc) * abs(d)
            acc += d
            s.append(d)
        seqs.append(s)
    return seqs


def collect(env_name, seed, axis_idx, seqs):
    """Returns per-sequence lists of tuples (v_prev[m], u[m], dx[m])."""
    all_seq_rows = []
    env = make_env(env_name, seed)
    for si, seq in enumerate(seqs):
        obs, _ = env.reset()
        p_prev = rel_pos(obs)
        v_prev = 0.0  # at reset the arm is (near-)static
        rows = []
        for d in seq:
            vec = np.zeros(3)
            vec[axis_idx] = d
            act = {"action.end_effector_position": vec,
                   "action.end_effector_rotation": np.zeros(3),
                   "action.base_motion": np.zeros(4),
                   "action.control_mode": np.zeros(1),
                   "action.gripper_close": np.zeros(1)}
            obs, *_ = env.step(act)
            p = rel_pos(obs)
            dx = float(p[axis_idx] - p_prev[axis_idx])
            u = float(np.clip(d, -1, 1) * OUTPUT_MAX)
            rows.append((v_prev, u, dx))
            v_prev = dx
            p_prev = p
        all_seq_rows.append(rows)
        print(f"  axis {axis_idx} seq {si + 1}/{len(seqs)}: {len(seq)} steps", flush=True)
    env.close()
    return all_seq_rows


def fit_sim(seq_rows, a_grid=None):
    """SIMULATION-error fit of dx_t = a*v_prev + b*u_t with v_prev taken from
    the MODEL (not the measurement): dx_t = b*s_t, s_t = u_t + a*s_{t-1},
    s reset to 0 at each sequence start. One-step OLS on measured v_prev is
    biased here (feedback + collinearity in constant runs inflated a to ~0.6
    while the step response and the measured K2 merged reach imply ~0.25-0.35);
    the simulation fit is consistent for this model class. For each a on a
    grid, b has a closed form; pick (a, b) minimizing SSE."""
    if a_grid is None:
        a_grid = np.arange(0.0, 0.96, 0.01)
    y = np.concatenate([[r[2] for r in rows] for rows in seq_rows])
    best = None
    for a in a_grid:
        s_all = []
        for rows in seq_rows:
            s = 0.0
            for r in rows:
                s = r[1] + a * s
                s_all.append(s)
        s_all = np.asarray(s_all)
        denom = float(s_all @ s_all)
        b = float(s_all @ y) / denom if denom > 0 else 0.0
        sse = float(((y - b * s_all) ** 2).sum())
        if best is None or sse < best[2]:
            best = (float(a), b, sse)
    a, b, sse = best
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"a": a, "b": b, "r2": 1.0 - sse / max(ss_tot, 1e-12),
            "n": int(len(y)), "fit": "simulation-error"}


def fit(rows, quadratic=False):
    v = np.array([r[0] for r in rows])
    u = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    cols = [v, u] + ([u * np.abs(u) / OUTPUT_MAX] if quadratic else [])
    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    out = {"a": float(coef[0]), "b": float(coef[1]), "r2": r2, "n": len(rows)}
    if quadratic:
        out["b2"] = float(coef[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="CloseDrawer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--axes", default="xyz")
    ap.add_argument("--out", default="analysis/sysid_osc_response.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    result = {"env_name": args.env_name, "seed": args.seed,
              "output_max": OUTPUT_MAX, "model": "dx = a*v_prev + b*u",
              "units": "v_prev,u,dx in meters (u = clip(d)*output_max)",
              "per_axis": {}}
    all_seq_rows = []
    raw = {}
    for ax in args.axes:
        idx = AXES[ax]
        print(f"[sysid] collecting axis {ax}", flush=True)
        seqs = build_sequences(rng)
        seq_rows = collect(args.env_name, args.seed, idx, seqs)
        all_seq_rows += seq_rows
        raw[ax] = [[list(map(float, r)) for r in rows] for rows in seq_rows]
        f = fit_sim(seq_rows)
        flat = [r for rows in seq_rows for r in rows]
        f_ols = fit(flat)
        if f_ols["r2"] < 0.8:
            f_ols = fit(flat, quadratic=True)
        f["ols_diagnostic"] = f_ols
        result["per_axis"][ax] = f
        print(f"[sysid] axis {ax}: {f}", flush=True)
    fp = fit_sim(all_seq_rows)
    result["pooled"] = fp
    result["raw_rows"] = raw  # (v_prev, u, dx) per sequence, for offline refits
    print(f"[sysid] pooled: {fp}", flush=True)
    # implied K2 merged reach rate at steady state (v=0 entry, u constant):
    a, b = fp["a"], fp["b"]
    # fine 2 steps from v0=0: d1=b*u, d2=a*b*u+b*u -> sum=b*u*(2+a)
    # merged 1 step: b*(2u) -> ratio merged/fine = 2/(2+a)
    result["implied_K2_reach_rate"] = 2.0 / (2.0 + a) if (2.0 + a) != 0 else None
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[sysid] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
