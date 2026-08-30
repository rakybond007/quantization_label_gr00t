"""K2 temporal quantization dynamics probe for dexjoco (absolute targets).

Arm: absolute EE pose targets (mocap + OSC opspace torque control).
Hand: absolute Allegro joint position targets (position actuators).

Fine: 2N commands, one target increment per tick.
K2:   N commands, every other target, each held 2 ticks (same final target).
"""
import json, sys, os
import numpy as np

sys.path.insert(0, "/sjw_alinlab/home/hojin2/multigpu_workspace/external_dependencies/dexjoco/dexjoco")
from dexjoco.sim.envs.panda_hammer_nail_env import PandaHammerNailGymEnv, _N_ALLEGRO

N = 25          # K2 command count -> 2N=50 ticks (1.0 s at 50 Hz control)
SETTLE = 25     # extra ticks holding final target (settled reference)

def make_env():
    env = PandaHammerNailGymEnv(seed=0, render_mode="rgb_array")
    env.hz = 1e9  # disable real-time sleep
    env._compute_observation = lambda: {}  # skip rendering
    env.reset(seed=0)
    return env

def get_ee(env):
    return env._data.site_xpos[env._site_id].copy()

def get_hand(env):
    return env._data.qpos[env._allegro_dof_ids].copy()

def run_arm(env, direction, amp, mode):
    """Ramp mocap target from p0 to p0+amp*dir over 2N ticks."""
    p0 = env._data.mocap_pos[env._panda_mocap_id].copy()
    q0 = env._data.mocap_quat[env._panda_mocap_id].copy()
    a0 = get_hand(env)  # hold hand at current pose
    d = np.zeros(3); d[direction] = 1.0
    targets = [p0 + d * amp * (t + 1) / (2 * N) for t in range(2 * N)]
    traj = []
    for t in range(2 * N):
        if mode == "fine":
            tgt = targets[t]
        else:  # k2: hold target of odd indices (1,3,...) for 2 ticks each
            tgt = targets[2 * (t // 2) + 1]
        act = np.concatenate([tgt, q0, a0])
        env.step(act)
        traj.append(get_ee(env))
    end = get_ee(env)
    for _ in range(SETTLE):
        env.step(np.concatenate([targets[-1], q0, a0]))
    settled = get_ee(env)
    return p0, np.array(traj), end, settled

def run_hand(env, amp, mode):
    """Ramp all allegro joints from a0 by +amp rad (clipped to ctrl range)."""
    a0 = get_hand(env)
    lo = env._model.actuator_ctrlrange[env._allegro_ctrl_ids, 0]
    hi = env._model.actuator_ctrlrange[env._allegro_ctrl_ids, 1]
    a1 = np.clip(a0 + amp, lo, hi)
    zeros7 = np.zeros(7)  # zero pose -> env holds current mocap target
    targets = [a0 + (a1 - a0) * (t + 1) / (2 * N) for t in range(2 * N)]
    traj = []
    for t in range(2 * N):
        tgt = targets[t] if mode == "fine" else targets[2 * (t // 2) + 1]
        env.step(np.concatenate([zeros7, tgt]))
        traj.append(get_hand(env))
    end = get_hand(env)
    for _ in range(SETTLE):
        env.step(np.concatenate([zeros7, targets[-1]]))
    settled = get_hand(env)
    return a0, a1, np.array(traj), end, settled

def arm_case(direction, amp):
    envf = make_env(); p0, trf, endf, setf = run_arm(envf, direction, amp, "fine")
    envk = make_env(); _,  trk, endk, setk = run_arm(envk, direction, amp, "k2")
    d = np.zeros(3); d[direction] = 1.0
    df, dk = float((endf - p0) @ d), float((endk - p0) @ d)
    sf, sk = float((setf - p0) @ d), float((setk - p0) @ d)
    diff = np.linalg.norm(trk - trf, axis=1)
    return {
        "axis": "xyz"[direction], "amplitude_m": amp,
        "achieved_fine_m": df, "achieved_k2_m": dk,
        "ratio_k2_over_fine_at_end": dk / df if df else None,
        "settled_fine_m": sf, "settled_k2_m": sk,
        "ratio_settled": sk / sf if sf else None,
        "traj_rms_dev_m": float(np.sqrt((diff ** 2).mean())),
        "traj_max_dev_m": float(diff.max()),
        "tracking_lag_fine_m": amp - df, "tracking_lag_k2_m": amp - dk,
    }

def hand_case(amp):
    envf = make_env(); a0, a1, trf, endf, setf = run_hand(envf, amp, "fine")
    envk = make_env(); _,  _,  trk, endk, setk = run_hand(envk, amp, "k2")
    span = a1 - a0
    m = np.abs(span) > 1e-6
    rat_end = float(np.mean(((endk - a0)[m]) / ((endf - a0)[m]))) if ((endf - a0)[m] != 0).all() else None
    rat_set = float(np.mean(((setk - a0)[m]) / ((setf - a0)[m])))
    diff = np.abs(trk - trf)
    return {
        "amplitude_rad": amp, "n_joints_moving": int(m.sum()),
        "mean_span_rad": float(np.abs(span[m]).mean()),
        "ratio_k2_over_fine_at_end": rat_end, "ratio_settled": rat_set,
        "mean_abs_final_gap_fine_rad": float(np.abs(a1 - endf)[m].mean()),
        "mean_abs_final_gap_k2_rad": float(np.abs(a1 - endk)[m].mean()),
        "traj_rms_dev_rad": float(np.sqrt((diff[:, m] ** 2).mean())),
        "traj_max_dev_rad": float(diff[:, m].max()),
    }

out = {
    "probe": "dexjoco K2 temporal quantization dynamics",
    "env": "PandaHammerNailGymEnv (control_dt=0.02, physics_dt=0.002, 10 substeps/tick)",
    "protocol": {"N_k2_commands": N, "ticks_total": 2 * N, "settle_ticks": SETTLE,
                 "fine": "2N absolute targets, 1/tick",
                 "k2": "N targets (every other), each held 2 ticks; identical final target"},
    "arm_abs_ee_pose": [], "hand_abs_joint": [],
}
for ax in (0, 1, 2):
    for amp in (0.02, 0.05, 0.10):
        r = arm_case(ax, amp); out["arm_abs_ee_pose"].append(r)
        print("arm", r["axis"], amp, "ratio_end=%.4f settled=%.4f rms=%.5f max=%.5f" %
              (r["ratio_k2_over_fine_at_end"], r["ratio_settled"], r["traj_rms_dev_m"], r["traj_max_dev_m"]), flush=True)
for amp in (0.1, 0.3, 0.6):
    r = hand_case(amp); out["hand_abs_joint"].append(r)
    print("hand", amp, "ratio_end=%.4f settled=%.4f rms=%.5f max=%.5f" %
          (r["ratio_k2_over_fine_at_end"], r["ratio_settled"], r["traj_rms_dev_rad"], r["traj_max_dev_rad"]), flush=True)

arm_r = [r["ratio_k2_over_fine_at_end"] for r in out["arm_abs_ee_pose"]]
hand_r = [r["ratio_k2_over_fine_at_end"] for r in out["hand_abs_joint"]]
out["summary"] = {
    "arm_ratio_end_mean": float(np.mean(arm_r)), "arm_ratio_end_range": [min(arm_r), max(arm_r)],
    "hand_ratio_end_mean": float(np.mean(hand_r)), "hand_ratio_end_range": [min(hand_r), max(hand_r)],
    "robocasa_reference_ratio": "0.89-0.90 (delta EE + achieved-mode re-anchoring)",
}
path = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/analysis/dexjoco_k2_dynamics_probe.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", path)
