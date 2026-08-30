"""Verify: dexjoco action[t] = absolute target pose ≈ state[t+1]?
If yes, our state-anchored delta reparametrization is well-defined.
If no, action may be something else (delta, controller setpoint different from reached pose, etc.)."""
import glob
import numpy as np
import pandas as pd

PARQS = sorted(glob.glob("/sjw_alinlab2/home/hojin2/.cache/dexjoco_lerobot/v20/hammer_nail/data/chunk-*/episode_*.parquet"))[:5]
for p in PARQS:
    df = pd.read_parquet(p)
    state = np.stack(df["observation.state"].values)     # (T, 23)
    action = np.stack(df["action"].values)               # (T, 22)
    T = state.shape[0]
    print(f"\n=== {p.split('/')[-1]}  T={T} ===")
    print(f"  state shape={state.shape}  action shape={action.shape}")

    # Hypothesis: action[t]._pos ≈ state[t+1]._pos
    pos_diff = np.linalg.norm(state[1:, :3] - action[:-1, :3], axis=-1)
    print(f"  ||state[t+1].pos - action[t].pos||  mean={pos_diff.mean():.5f}  max={pos_diff.max():.5f}  median={np.median(pos_diff):.5f}")

    # For arm_rot: state is QUAT (4-dim), action is ROTVEC (3-dim). Convert action rotvec -> quat, compare.
    from scipy.spatial.transform import Rotation as R
    state_rot_quat = state[1:, 3:7]   # (T-1, 4)  scalar-first
    act_rot_rv    = action[:-1, 3:6]  # (T-1, 3)
    act_rot_quat  = R.from_rotvec(act_rot_rv).as_quat(scalar_first=True)  # (T-1, 4)
    # Quat distance: 1 - |<q1,q2>|
    dot = np.abs(np.sum(state_rot_quat * act_rot_quat, axis=-1))
    rot_diff = 1.0 - dot
    print(f"  rot distance (1-|<q,q>|)  mean={rot_diff.mean():.5f}  max={rot_diff.max():.5f}  median={np.median(rot_diff):.5f}")

    # Hand (16 dim each): state[t+1].hand vs action[t].hand
    hnd_diff = np.linalg.norm(state[1:, 7:23] - action[:-1, 6:22], axis=-1)
    print(f"  ||state[t+1].hand - action[t].hand||  mean={hnd_diff.mean():.5f}  max={hnd_diff.max():.5f}  median={np.median(hnd_diff):.5f}")

    # Alternative hypothesis: action[t] ≈ state[t] (controller setpoint = current state, then small motion)
    # If this is closer than state[t+1], abs target hypothesis is weaker.
    pos_diff_curr = np.linalg.norm(state[:-1, :3] - action[:-1, :3], axis=-1)
    print(f"  ||state[t].pos - action[t].pos||      mean={pos_diff_curr.mean():.5f}  (sanity: action vs CURRENT state)")

    # Show first 3 timesteps in detail
    print(f"  --- first 3 timesteps ---")
    for t in range(3):
        print(f"    t={t}: state.pos={state[t, :3]}  action.pos={action[t, :3]}  state[t+1].pos={state[t+1, :3]}")
print("\nABS_CHECK_DONE")
