"""Train the merge gate via GRPO on top of a frozen baseline GR00T policy.

Usage:
    python scripts/train_gate_rl.py \
        --model-path ckpt/.../checkpoint-60000 \
        --env-name TurnOnStove \
        --iters 100 \
        --group-size 4 \
        --gpu 0
"""

from dataclasses import dataclass
from pathlib import Path
import os
import time

import numpy as np
import torch
import tyro

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.rl.gate import GateMLP
from gr00t.rl.episode_runner import make_env, run_episode
from gr00t.rl.grpo_trainer import episode_reward, grpo_loss, group_advantages
from gr00t.rl.wandb_video import annotate_episode, merge_timeline_image


@dataclass
class Args:
    model_path: str = "ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"
    env_name: str = "TurnOnStove"
    data_config: str = "single_panda_gripper"
    embodiment_tag: str = "new_embodiment"
    denoising_steps: int = 4

    iters: int = 50                       # number of training iterations (groups)
    group_size: int = 4                    # K rollouts per iteration (GRPO group)
    max_env_steps: int = 1500
    seed: int = 42
    n_envs: int = 1                        # 1 = sequential (legacy). >1 = AsyncVectorEnv parallel.

    lr: float = 3e-4
    clip_eps: float = 0.2
    kl_weight: float = 0.05
    entropy_weight: float = 0.01
    prior_logit: float = -2.0
    init_logit: float = -2.0
    hidden_dim: int = 256
    alpha_len: float = 0.1
    alpha_merge: float = 0.05

    gpu: int = 0
    log_video_every: int = 5               # log a video every N iters
    eval_every: int = 20                   # run an eval every N iters (stochastic gate, fresh seeds)
    eval_n_episodes: int = 32              # total eval episodes (1 sequential w/ video + rest vec for metric)
    save_path: str = "ckpt/rl/gate_baseline"

    # Wandb
    wandb_project: str = "GR00T-RL-gate"
    wandb_run_name: str = ""               # default: f"{env_name}_seed{seed}"
    no_wandb: bool = False


def build_policy(args: Args) -> Gr00tPolicy:
    cfg = DATA_CONFIG_MAP[args.data_config]
    return Gr00tPolicy(
        model_path=args.model_path,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=f"cuda:{args.gpu}",
    )


def main(args: Args):
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"[Setup] Loading base policy from {args.model_path}")
    policy = build_policy(args)
    # Freeze the entire base model
    for p in policy.model.parameters():
        p.requires_grad_(False)
    policy.model.eval()

    # Probe VL feature dim by running one fake forward
    print("[Setup] Probing VL feature dimension...")
    env = make_env(args.env_name, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    from gr00t.rl.episode_runner import _flip_views, _add_time_axis
    from gr00t.rl.policy_with_features import policy_forward
    obs_flipped = _flip_views(obs)
    obs_T = _add_time_axis(obs_flipped)
    _, vl_feats = policy_forward(policy, obs_T)
    feat_dim = vl_feats.shape[-1]
    print(f"[Setup] VL feature dim = {feat_dim}, T = {vl_feats.shape[1]}")

    gate = GateMLP(feature_dim=feat_dim, hidden_dim=args.hidden_dim,
                   init_logit=args.init_logit).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)

    # Wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = args.wandb_run_name or f"{args.env_name}_seed{args.seed}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train] Starting GRPO. iters={args.iters}, group_size={args.group_size}")
    for it in range(args.iters):
        t0 = time.time()
        # Sample K episodes from same seed (group). Each has its own gate sampling.
        capture = (it % args.log_video_every == 0)
        if args.n_envs > 1:
            from gr00t.rl.vec_episode_runner import run_vec_episodes
            # All K envs share same seed → trajectories diverge only via gate sampling
            seeds = [args.seed] * args.group_size
            # group_size must equal n_envs in the simple vec mode; if larger, run multiple sub-batches
            episodes = []
            for sub in range(0, args.group_size, args.n_envs):
                sub_seeds = seeds[sub:sub + args.n_envs]
                sub_episodes = run_vec_episodes(
                    env_name=args.env_name, seeds=sub_seeds,
                    policy=policy, gate=gate, device=device,
                    max_env_steps=args.max_env_steps,
                    sample_gate=True,
                    capture_idx=0 if (capture and sub == 0) else -1,
                )
                episodes.extend(sub_episodes)
        else:
            episodes = []
            for k in range(args.group_size):
                res = run_episode(
                    env=env, policy=policy, gate=gate, device=device,
                    seed=args.seed,    # same seed per group
                    max_env_steps=args.max_env_steps,
                    sample_gate=True,
                    capture_frames=(capture and k == 0),
                )
                episodes.append(res)

        rewards = [episode_reward(ep, max_env_steps=args.max_env_steps,
                                  alpha_len=args.alpha_len, alpha_merge=args.alpha_merge)
                   for ep in episodes]
        adv = group_advantages(rewards)

        # Filter out empty episodes (no chunks)
        valid_idx = [i for i, ep in enumerate(episodes) if ep.chunks]
        if not valid_idx:
            print(f"[iter {it}] All episodes had 0 chunks, skipping update.")
            continue

        eps_used = [episodes[i] for i in valid_idx]
        adv_used = [adv[i] for i in valid_idx]

        loss_pkg = grpo_loss(
            gate=gate, episodes=eps_used, advantages_per_ep=adv_used,
            clip_eps=args.clip_eps, kl_weight=args.kl_weight,
            entropy_weight=args.entropy_weight,
            prior_logit=args.prior_logit, device=device,
        )

        opt.zero_grad()
        loss_pkg.total.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()

        # Stats
        success_rate = float(np.mean([ep.success for ep in episodes]))
        avg_steps = float(np.mean([ep.total_env_steps for ep in episodes]))
        avg_merge_ratio = float(np.mean([
            (sum(c.decision for c in ep.chunks) / max(len(ep.chunks), 1))
            for ep in episodes if ep.chunks
        ]))
        avg_p = float(np.mean([np.mean([c.prob for c in ep.chunks]) for ep in episodes if ep.chunks]))

        elapsed = time.time() - t0
        print(f"[iter {it:3d}] success={success_rate:.2f} steps={avg_steps:5.0f} "
              f"merge={avg_merge_ratio:.2f} p={avg_p:.2f} | "
              f"loss(p={loss_pkg.policy_loss.item():+.3f} kl={loss_pkg.kl_loss.item():.3f} "
              f"H={loss_pkg.entropy.item():.3f}) | dt={elapsed:.1f}s")

        if use_wandb:
            log = {
                "iter": it,
                "success_rate": success_rate,
                "avg_env_steps": avg_steps,
                "avg_merge_ratio": avg_merge_ratio,
                "avg_p_merge": avg_p,
                "loss/policy": loss_pkg.policy_loss.item(),
                "loss/kl": loss_pkg.kl_loss.item(),
                "loss/entropy": loss_pkg.entropy.item(),
                "loss/total": loss_pkg.total.item(),
                "reward/mean": float(np.mean(rewards)),
                "reward/std": float(np.std(rewards)),
                "wall_dt_s": elapsed,
            }
            if capture and episodes[0].frames:
                video = annotate_episode(episodes[0])
                # wandb expects (T, C, H, W) or (T, H, W, C) with format
                video_thwc = video  # (T, H, W, 3) uint8
                log["video/episode"] = wandb.Video(
                    video_thwc.transpose(0, 3, 1, 2),  # (T, C, H, W)
                    fps=20, format="mp4",
                    caption=f"iter {it} | success={episodes[0].success}",
                )
                log["timeline/merge_decisions"] = wandb.Image(merge_timeline_image(episodes[0]))
            wandb.log(log)

        # Save gate periodically
        if (it + 1) % 10 == 0:
            ckpt = save_dir / f"gate_iter_{it + 1}.pt"
            torch.save({"gate": gate.state_dict(), "args": vars(args), "iter": it + 1}, ckpt)
            print(f"[ckpt] Saved {ckpt}")

        # ===== Eval phase (stochastic gate, fresh seeds) =====
        # Stochastic gate (Bernoulli at P) — matches how the policy would be
        # deployed in inference; deterministic threshold would distort behavior.
        # Speedup: first ep runs sequentially in main-process env so we can grab
        # render frames for video; remaining N-1 episodes run via vec sub-batches
        # without frame capture (much faster, batched policy forward).
        if (it + 1) % args.eval_every == 0:
            eval_seeds = [args.seed + 1000 + j for j in range(args.eval_n_episodes)]
            eval_eps = []
            # 1) one sequential ep with video
            ep_video = run_episode(
                env=env, policy=policy, gate=gate, device=device,
                seed=eval_seeds[0], max_env_steps=args.max_env_steps,
                sample_gate=True, capture_frames=True,
            )
            eval_eps.append(ep_video)
            # 2) the rest via vec sub-batches (no frame capture)
            rest_seeds = eval_seeds[1:]
            if rest_seeds:
                if args.n_envs > 1:
                    from gr00t.rl.vec_episode_runner import run_vec_episodes
                    for sub in range(0, len(rest_seeds), args.n_envs):
                        sub_seeds = rest_seeds[sub:sub + args.n_envs]
                        sub_eps = run_vec_episodes(
                            env_name=args.env_name, seeds=sub_seeds,
                            policy=policy, gate=gate, device=device,
                            max_env_steps=args.max_env_steps,
                            sample_gate=True, capture_idx=-1,
                        )
                        eval_eps.extend(sub_eps)
                else:
                    for s in rest_seeds:
                        eval_eps.append(run_episode(
                            env=env, policy=policy, gate=gate, device=device,
                            seed=s, max_env_steps=args.max_env_steps,
                            sample_gate=True, capture_frames=False,
                        ))

            es = float(np.mean([ep.success for ep in eval_eps]))
            est = float(np.mean([ep.total_env_steps for ep in eval_eps]))
            emr = float(np.mean([
                sum(c.decision for c in ep.chunks) / max(len(ep.chunks), 1)
                for ep in eval_eps if ep.chunks
            ])) if any(ep.chunks for ep in eval_eps) else 0.0
            print(f"[EVAL it={it + 1}] success={es:.2f} steps={est:5.0f} merge={emr:.2f}")

            # Save annotated mp4 to disk for offline monitoring
            if eval_eps[0].frames:
                try:
                    import cv2
                    video = annotate_episode(eval_eps[0])
                    out_mp4 = save_dir / f"eval_video_iter{it + 1}.mp4"
                    H, W = video.shape[1:3]
                    writer = cv2.VideoWriter(str(out_mp4),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             20, (W, H))
                    for f in video:
                        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    writer.release()
                    print(f"[EVAL] saved {out_mp4}")
                except Exception as e:
                    print(f"[EVAL] disk video save failed: {e}")

            if use_wandb:
                eval_log = {
                    "eval/success_rate": es,
                    "eval/avg_env_steps": est,
                    "eval/merge_ratio": emr,
                    "iter": it + 1,
                }
                if eval_eps[0].frames:
                    v = annotate_episode(eval_eps[0])
                    eval_log["eval/video"] = wandb.Video(
                        v.transpose(0, 3, 1, 2), fps=20, format="mp4",
                        caption=f"EVAL iter {it + 1} | success={eval_eps[0].success}",
                    )
                    eval_log["eval/timeline"] = wandb.Image(merge_timeline_image(eval_eps[0]))
                wandb.log(eval_log)

    env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
