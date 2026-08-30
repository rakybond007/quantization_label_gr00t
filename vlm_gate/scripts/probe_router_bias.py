"""Controlled probe: does obs['_compress_bias'] actually move the MoE router?

For a handful of REAL robocasa observations, call the inference server on the
SAME obs with several bias values and print the returned moe_probs (computed by
the head AFTER the bias is added, so it is the decisive plumbing signal) plus the
resulting chunk length H. If probs shift toward idx 1 (m8) / idx 2 (m4) as bias
rises, the coupling works; if probs are identical across bias, the bias never
reaches the head.

Run in the robocasa_gr00t env against a running inference_service_fair_moe server.
"""
import argparse
import os

import numpy as np
from robosuite.controllers import load_composite_controller_config

from gr00t.eval.robocasa_simulation import SimulationInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env


def H_of(action):
    return next(np.asarray(v).shape[0]
                for k, v in action.items()
                if not k.startswith(("moe_", "_moe"))
                and np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] > 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", default="OpenDrawer")
    p.add_argument("--port", type=int, default=11000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--n_probes", type=int, default=6)
    p.add_argument("--gap", type=int, default=8, help="env steps (bias 0) between probes")
    args = p.parse_args()

    client = SimulationInferenceClient(host=args.host, port=args.port)
    env = load_robocasa_gym_env(
        args.env_name, seed=42, robots="PandaOmron",
        camera_widths=256, camera_heights=256, render_onscreen=False,
        obj_instance_split="A", generative_textures="100p",
        randomize_cameras=False, layout_ids=-1,
        style_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11], collect_data=False,
    )
    env = RoboCasaWrapper(env)
    env = MultiStepWrapper(env, video_delta_indices=np.arange(1),
                           state_delta_indices=np.arange(1), n_action_steps=1)

    biases = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]
    obs, _ = env.reset()
    print(f"# env={args.env_name}  probing {args.n_probes} obs x {len(biases)} biases")
    print(f"# moe_probs = [raw16, m8, m4, n8]  (bias adds to idx1=m8, idx2=m4)")

    for pi in range(args.n_probes):
        obs["video.left_view"] = np.flip(obs["video.left_view"], axis=1)
        obs["video.right_view"] = np.flip(obs["video.right_view"], axis=1)
        obs["video.wrist_view"] = np.flip(obs["video.wrist_view"], axis=1)
        print(f"\n=== probe obs #{pi} ===")
        base_probs = None
        for b in biases:
            o = dict(obs)
            o["_compress_bias"] = b
            a = client.get_action(o)
            probs = a.get("_moe_probs", a.get("moe_probs"))
            probs = np.asarray(probs).reshape(-1) if probs is not None else None
            H = H_of(a)
            ps = "[" + " ".join(f"{x:.3f}" for x in probs) + "]" if probs is not None else "None"
            arg = int(np.argmax(probs)) if probs is not None else -1
            tag = ""
            if probs is not None and b == 0.0:
                base_probs = probs.copy()
            if probs is not None and base_probs is not None and b != 0.0:
                dcomp = (probs[1] + probs[2]) - (base_probs[1] + base_probs[2])
                tag = f"  d(m8+m4 vs bias0)={dcomp:+.3f}"
            print(f"  bias={b:+.1f}  H={H:2d}  argmax={arg}  probs={ps}{tag}")
        # advance the env a few steps with no bias so the next probe is a new state
        for _ in range(args.gap):
            o = dict(obs); o["_compress_bias"] = 0.0
            a = client.get_action(o)
            H = H_of(a)
            done = False
            for j in range(H):
                one = {}
                for k, v in a.items():
                    if k.startswith(("moe_", "_moe")):
                        continue
                    v = np.asarray(v)
                    if v.ndim == 0 or v.shape[0] != H:
                        continue
                    one[k] = np.asarray([[v[j]]]) if v.ndim == 1 else np.asarray([v[j]])
                obs, _, term, trunc, _ = env.step(one)
                done = term or trunc
                if done:
                    break
            if done:
                obs, _ = env.reset()
                break
    env.close()
    print("\n# DONE. If d(m8+m4) rises with bias, the coupling reaches the router.")


if __name__ == "__main__":
    main()
