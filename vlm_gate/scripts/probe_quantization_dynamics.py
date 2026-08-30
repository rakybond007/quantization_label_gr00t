"""Empirical A/B test: does summing two delta commands reproduce the endpoint
of executing them sequentially, under the real robocasa OSC controller?

From an identical reset (same seed), executes three scripted regimes of pure
EE-translation deltas and reports final EE displacement:
  fine     : d applied N times            (N env.steps)
  quant    : (K*d) applied N/K times      (N/K env.steps)  <- our K2 merge
  quant+set: same as quant, then (N - N/K) zero-delta settle steps
             (equal wall-clock comparison: does the gap close given time?)
Also prints per-step tracking (achieved / commanded) to quantify the
"achieved-mode + non-convergent PD" displacement loss the OOD doc describes.
"""
import argparse
import numpy as np
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env


def make_env(env_name, seed):
    env = load_robocasa_gym_env(
        env_name, seed=seed, robots="PandaOmron", camera_widths=256,
        camera_heights=256, render_onscreen=False, obj_instance_split="A",
        generative_textures=None, randomize_cameras=False,
        layout_ids=None, style_ids=None, collect_data=False)
    return RoboCasaWrapper(env)


def rel_pos(obs):
    """EE position in the robot BASE frame — the frame the OSC delta acts in.
    (World-frame measurement aliases axes when the base yaw != 0.)"""
    return np.asarray(obs["state.end_effector_position_relative"], dtype=float).reshape(3)


def run(env_name, seed, deltas, tag, compensator=None, fine_blocks=None):
    """If compensator is given, `deltas` entries are replaced at execution time
    by compensator.merged_command(v_now, fine_blocks[i]) where v_now is the
    observed last per-step EE displacement (m/step)."""
    env = make_env(env_name, seed)
    obs, _ = env.reset()
    p0 = rel_pos(obs)
    traj = [p0.copy()]
    v_now = np.zeros(3)
    clip_excess = 0.0
    if compensator is not None:
        compensator.begin_chunk(v_now)
    for i, d in enumerate(deltas):
        d_exec = d.copy()
        if compensator is not None and fine_blocks is not None and fine_blocks[i] is not None:
            d_exec, exc = compensator.merged_command(v_now, fine_blocks[i])
            clip_excess += float(np.abs(exc).sum())
        act = {"action.end_effector_position": np.asarray(d_exec, dtype=float),
               "action.end_effector_rotation": np.zeros(3),
               "action.base_motion": np.zeros(4),
               "action.control_mode": np.zeros(1),
               "action.gripper_close": np.zeros(1)}
        obs, *_ = env.step(act)
        p = rel_pos(obs)
        v_now = p - traj[-1]
        traj.append(p)
    p_end = traj[-1]
    env.close()
    disp = p_end - p0
    if compensator is not None and clip_excess > 0:
        print(f"[{tag}] total clip excess (action units): {clip_excess:.3f}")
    return disp, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="CloseDrawer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d", type=float, default=0.25, help="per-step x-delta (action units, clip=1)")
    ap.add_argument("--n", type=int, default=8, help="fine steps")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--compensate", choices=["none", "model", "scalar"], default="none")
    ap.add_argument("--sysid-json", default="analysis/sysid_osc_response.json")
    ap.add_argument("--scalar-c", type=float, default=None)
    args = ap.parse_args()

    d = np.array([args.d, 0.0, 0.0])
    N, K = args.n, args.k
    fine = [d] * N
    quant = [d * K] * (N // K)
    quant_settle = quant + [np.zeros(3)] * (N - N // K)

    disp_f, traj_f = run(args.env_name, args.seed, fine, "fine")

    comp = None
    fine_blocks = None
    if args.compensate != "none":
        from osc_compensation import make_compensator
        comp = make_compensator(args.compensate, args.sysid_json, args.scalar_c)
        # each merged step corresponds to K fine deltas
        fine_blocks = [np.stack([d] * K) for _ in range(N // K)]
        print(f"[compensate] mode={args.compensate}")
    disp_q, _ = run(args.env_name, args.seed, quant, "quant", comp, fine_blocks)
    disp_s, _ = run(args.env_name, args.seed, quant_settle, "quant+settle", comp,
                    (fine_blocks + [None] * (N - N // K)) if fine_blocks else None)

    cmd_total = args.d * N  # in action units; report ratios instead of meters
    print(f"\n=== {args.env_name} seed={args.seed} d={args.d} N={N} K={K} ===")
    print(f"fine        : disp={disp_f.round(4)}  |x|={disp_f[0]:.4f}")
    print(f"quant K{K}    : disp={disp_q.round(4)}  |x|={disp_q[0]:.4f}  vs fine {100*disp_q[0]/max(disp_f[0],1e-9):.1f}%")
    print(f"quant+settle: disp={disp_s.round(4)}  |x|={disp_s[0]:.4f}  vs fine {100*disp_s[0]/max(disp_f[0],1e-9):.1f}%")
    # per-step tracking under fine
    steps = np.diff(np.array(traj_f)[:, 0])
    print(f"fine per-step x-advance: {steps.round(4)} (같은 delta 반복인데 속도상태 따라 변동하면 F가 상태의존이라는 뜻)")


if __name__ == "__main__":
    main()
