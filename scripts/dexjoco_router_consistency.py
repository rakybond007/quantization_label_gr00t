"""Router-consistency analysis for DexJoCo single-arm MoE.

Three phases (single task, e.g. hammer_nail):

  phase_a (collect_prior): run MoE balance-ON + conf0p7. Record per-chunk
    router decoder pick (0=raw16, 1=n8, 2=m8, 3=m4) for each episode.
    Output: rollouts_A.jsonl

  phase_b (forced_from_prior): run BASELINE (head=main, raw 16). Use the
    prior P(compressed | chunk_idx) computed from rollouts_A.jsonl to sample
    a per-chunk compression schedule with the SAME overall compression rate.
    Apply K=2 block-last compression at scheduled chunks.
    Output: rollouts_B.jsonl

  phase_c (random_schedule): same as B but per-chunk compression decision is
    a Bernoulli with the overall mean rate (chunk_idx-agnostic).
    Output: rollouts_C.jsonl

All phases are RESUMABLE: each episode appends a single JSON line; on restart
the script counts existing lines and skips those episode indices.

Server side: separate shell launches.
  - phase_a server : serve_policy_dexjoco.py with MoE ckpt + --head moe
                     --moe-stochastic --moe-confidence-threshold 0.7
  - phase_b/c srv  : serve_policy_dexjoco.py with BASELINE ckpt + --head main
"""
import json
import os
import sys
import random
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from openpi_client import image_tools, websocket_client_policy
from dexjoco_openpi_client.dexjoco_openpi_env import DexJoCoOpenPIEnv


def _process_obs_native(self, env_obs: dict):
    obs = {}
    for policy_key, env_key in self.camera_mapping.items():
        obs[policy_key] = image_tools.convert_to_uint8(env_obs[env_key])
    obs["state"] = env_obs["state"][:46 if self.dual_arm else 23]
    obs["prompt"] = self.prompt
    return obs

DexJoCoOpenPIEnv._process_obs = _process_obs_native


def _is_compressed_pick(pick: int) -> bool:
    """Decoder index convention: 0=raw16, 1=n8, 2=m8, 3=m4. m8/m4 = compressed."""
    return pick in (2, 3)


def _block_last_compress(chunk: np.ndarray, K: int) -> np.ndarray:
    """K-block fixed-ratio block-last compression for absolute actions.
    16 -> 16//K + (16 mod K) (raw tail). E.g., K=2 -> 8, K=3 -> 6.
    """
    H = chunk.shape[0]
    num_full = H // K
    out = []
    for i in range(num_full):
        out.append(chunk[i * K:(i + 1) * K][-1])
    for j in range(num_full * K, H):
        out.append(chunk[j])
    return np.stack(out)


def _count_existing(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    with open(jsonl_path) as f:
        return sum(1 for _ in f)


def _load_prior(jsonl_path: Path) -> tuple[dict, float]:
    """Compute P(compressed | chunk_idx) and overall mean rate from Phase A."""
    counts: dict[int, list[int]] = {}  # chunk_idx -> list of compressed (0/1)
    n_total = n_compressed = 0
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            for pk in d.get("picks", []):
                idx = int(pk["chunk_idx"])
                comp = 1 if _is_compressed_pick(int(pk["picked"])) else 0
                counts.setdefault(idx, []).append(comp)
                n_total += 1
                n_compressed += comp
    prior = {i: float(np.mean(v)) for i, v in counts.items()}
    overall = n_compressed / max(n_total, 1)
    return prior, overall


def _sample_schedule_from_prior(num_chunks_max: int, prior: dict, default_rate: float,
                                rng: random.Random) -> list[bool]:
    """For each chunk_idx, sample bernoulli(prior[i] or default)."""
    out = []
    for i in range(num_chunks_max):
        p = prior.get(i, default_rate)
        out.append(rng.random() < p)
    return out


def _sample_schedule_random(num_chunks_max: int, rate: float,
                            rng: random.Random) -> list[bool]:
    return [rng.random() < rate for _ in range(num_chunks_max)]


def _run_episode(env, client, ep_idx: int, mode: str, max_steps: int,
                 compress_k: int, schedule: list[bool] | None):
    """Run one episode. Returns dict with per-chunk log + final success/steps."""
    env.reset()
    picks_log = []          # phase_a: chunk-level decoder pick
    sched_log = []          # phase_b/c: chunk-level compressed bool
    chunk_idx = 0
    steps = 0
    # Pre-roll for click_mouse not needed for hammer_nail; keep generic.
    while steps < max_steps:
        result = client.infer(env.get_obs())
        chunk = np.asarray(result["actions"])  # (H, D)
        if mode == "phase_a":
            picked = int(result.get("_moe_picked", -1))
            picks_log.append({"chunk_idx": chunk_idx,
                              "picked": picked,
                              "chunk_len_pre": int(chunk.shape[0])})
            # MoE server already returns the chosen decoder's chunk; just execute
            exec_chunk = chunk
        else:
            do_compress = bool(schedule[chunk_idx]) if chunk_idx < len(schedule) else False
            if do_compress and compress_k > 1:
                exec_chunk = _block_last_compress(chunk, compress_k)
            else:
                exec_chunk = chunk
            sched_log.append({"chunk_idx": chunk_idx,
                              "compressed": do_compress,
                              "chunk_len_pre": int(chunk.shape[0]),
                              "chunk_len_exec": int(exec_chunk.shape[0])})

        for a in exec_chunk:
            env.step(np.array(a))
            steps += 1
            if env.is_done or steps >= max_steps:
                break
        chunk_idx += 1
        if env.is_done:
            break

    return {
        "ep": ep_idx,
        "success": bool(env.is_success),
        "steps": int(steps),
        "n_chunks": int(chunk_idx),
        **({"picks": picks_log} if mode == "phase_a" else {"schedule": sched_log}),
    }


def main(
    config: Path,
    output_jsonl: Path,
    mode: Literal["phase_a", "phase_b", "phase_c"],
    prior_jsonl: Path | None = None,  # phase_b only
    episodes: int = 50,
    seed: int = 0,
    port: int = 8000,
    host: str = "127.0.0.1",
    max_episode_steps: int = 1500,
    compress_k: int = 2,
    pad_state_dim46: bool = False,
    override_rate: float | None = None,  # phase_c: explicit rate (else derive from prior)
):
    import yaml
    os.environ.setdefault("MUJOCO_GL", "egl")

    cfg = yaml.safe_load(open(config))
    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    dual_arm = cfg["robot_type"] == "dual_arm"
    prompt = cfg["prompt"]

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count existing lines
    done = _count_existing(output_jsonl)
    if done >= episodes:
        print(f"[{mode}] already have {done} episodes, target={episodes}; nothing to do.")
        return
    print(f"[{mode}] resuming from ep {done} -> {episodes}; writing to {output_jsonl}")

    env = DexJoCoOpenPIEnv(
        env_name=env_name, camera_mapping=camera_mapping, seed=seed,
        rand_full=False, randomize_dynamics=False,
        dual_arm=dual_arm, prompt=prompt, render_mode="rgb_array",
        pad_state_dim46=pad_state_dim46, password=cfg.get("password", None),
    )
    env.start()
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    # Load prior + compute rate for phase_b/c
    schedule_rate = 0.0
    prior = {}
    if mode in ("phase_b", "phase_c"):
        if mode == "phase_b":
            assert prior_jsonl is not None, "phase_b requires --prior-jsonl"
            prior, overall = _load_prior(Path(prior_jsonl))
            schedule_rate = override_rate if override_rate is not None else overall
            print(f"[{mode}] prior loaded: {len(prior)} chunk_idx bins, mean rate={overall:.3f}, used_rate={schedule_rate:.3f}")
        else:  # phase_c
            schedule_rate = override_rate
            if schedule_rate is None:
                # Try to compute from prior_jsonl for consistency with B
                if prior_jsonl is not None and Path(prior_jsonl).exists():
                    _, schedule_rate = _load_prior(Path(prior_jsonl))
                else:
                    schedule_rate = 0.3  # fallback
            print(f"[{mode}] schedule uniform-random rate={schedule_rate:.3f}")

    NUM_CHUNKS_MAX = 200  # ample upper bound for a 1500-step episode (worst case raw16: 94 chunks)

    for ep in range(done, episodes):
        # Deterministic per-episode RNG for reproducibility / resume
        rng = random.Random(seed * 10000 + ep)

        if mode == "phase_b":
            schedule = _sample_schedule_from_prior(NUM_CHUNKS_MAX, prior, schedule_rate, rng)
        elif mode == "phase_c":
            schedule = _sample_schedule_random(NUM_CHUNKS_MAX, schedule_rate, rng)
        else:
            schedule = None

        print(f"[{mode}] ep {ep}/{episodes}")
        rec = _run_episode(env, client, ep, mode, max_episode_steps, compress_k, schedule)
        with open(output_jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        succ_str = "✓" if rec["success"] else "✗"
        n_compressed = (sum(1 for p in rec.get("picks", []) if _is_compressed_pick(int(p["picked"])))
                        if mode == "phase_a" else
                        sum(1 for s in rec.get("schedule", []) if s["compressed"]))
        print(f"  {succ_str} steps={rec['steps']} n_chunks={rec['n_chunks']} n_compressed={n_compressed}")

    print(f"[{mode}] DONE: wrote {episodes} episodes to {output_jsonl}")


if __name__ == "__main__":
    tyro.cli(main)
