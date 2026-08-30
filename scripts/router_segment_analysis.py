"""Router segment analysis — simplified per user spec.

For one task (e.g. CloseDoubleDoor) from kimtaey/robocasa_mg_gr00t_300:
  - Pick an episode. Split into N segments by time.
  - For each segment: subsample timesteps, for each sampled t:
      * force-pick each of K decoders -> compute MSE vs per-decoder compressed GT
      * free-run -> obtain router prob distribution
  - Aggregate per segment: mean loss (compressed vs raw), mean router probs.
  - Save per-episode figure: cols = N segments,
      row0: frame at segment start
      row1: 2-bar (compressed vs raw avg loss)
      row2: 4-bar (router prob per decoder)

Resumable: per-(ep, seg, t) JSONL cache. Cherry-pick favorable episodes by hand
after browsing the produced figures.

Robocasa decoder ordering (MoE4 v1, B-only, no-metaq):
  0: raw16  (T=16, env=16)
  1: m8     (T=8,  env=16, sum-pair of raw 16)
  2: m4     (T=4,  env=8,  sum-pair of first 8)
  3: n8     (T=8,  env=8,  raw first 8)
Compressed = {1, 2}; Raw = {0, 3}.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import pyarrow.parquet as pq
from PIL import Image

ACTION_DIM = 12
DISCRETE_DIMS = (4, 11)  # control_mode (idx 4), gripper_close (idx 11)
H = 16
DECODER_NAMES = ["raw16", "m8", "m4", "n8"]
DECODER_T = [16, 8, 4, 8]
DECODER_ENV_STEPS = [16, 16, 8, 8]
COMPRESSED_IDX = (1, 2)
RAW_IDX = (0, 3)

# Action slice layout in the raw 12-dim action vector (per modality.json)
ACTION_KEYS = [
    ("action.base_motion", 0, 4),
    ("action.control_mode", 4, 5),
    ("action.end_effector_position", 5, 8),
    ("action.end_effector_rotation", 8, 11),
    ("action.gripper_close", 11, 12),
]

# State slices (only those listed in single_panda_gripper modality_config)
STATE_KEYS = [
    ("state.base_position", 0, 3),
    ("state.base_rotation", 3, 7),
    ("state.end_effector_position_relative", 10, 13),
    ("state.end_effector_rotation_relative", 17, 21),
    ("state.gripper_qpos", 21, 23),
]


def compress_gt(actions: np.ndarray, k: int) -> np.ndarray:
    """Build compressed-GT target for decoder k. actions: (16, 12)."""
    if k == 0:  # raw16
        return actions[:16].copy()
    if k == 3:  # n8 = raw first 8
        return actions[:8].copy()
    # k=1 m8: sum-pair of 16  ->  8
    # k=2 m4: sum-pair of first 8  ->  4
    src = actions[:16] if k == 1 else actions[:8]
    T_in = src.shape[0]
    T_out = T_in // 2
    out = np.zeros((T_out, actions.shape[1]), dtype=actions.dtype)
    for i in range(T_out):
        blk = src[2 * i:2 * i + 2]
        out[i] = blk.sum(axis=0)
        for d in DISCRETE_DIMS:
            out[i, d] = blk[-1, d]  # discrete: last
    return out


def load_model(ckpt_repo: str, data_config_name: str = "single_panda_gripper"):
    import sys
    sys.path.insert(0, "/sjw_alinlab2/home/hojin2/multigpu_workspace/Isaac-GR00T")
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy_fair_moe import Gr00tPolicyFairMoe
    from gr00t.data.embodiment_tags import EmbodimentTag

    cfg = DATA_CONFIG_MAP[data_config_name]
    policy = Gr00tPolicyFairMoe(
        model_path=ckpt_repo,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(backbone_model_type="eagle"),
        denoising_steps=4,
    )
    policy.model.action_head.discrete_action_dims = list(DISCRETE_DIMS)
    policy.inference_head = "moe"   # route through MoE expert decoders (single_pick)
    return policy


def list_episodes_for_task(repo_root: Path, task_substring: str, max_eps: int = 999999):
    """Return episode indices whose task description contains substring (case-insensitive)."""
    sub = task_substring.lower()
    out = []
    with open(repo_root / "meta/episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            tasks = ep.get("tasks", [])
            if isinstance(tasks, list):
                hit = any(sub in str(t).lower() for t in tasks)
            else:
                hit = sub in str(tasks).lower()
            if hit:
                out.append(ep["episode_index"])
            if len(out) >= max_eps:
                break
    return out


def load_episode_df(repo_root: Path, ep_idx: int):
    p = repo_root / f"data/chunk-{ep_idx // 300:03d}/episode_{ep_idx:06d}.parquet"
    return pq.read_table(str(p)).to_pandas()


def grab_frame(repo_root: Path, ep_idx: int, view: str, t: int) -> np.ndarray:
    p = repo_root / f"videos/chunk-{ep_idx // 300:03d}/observation.images.{view}/episode_{ep_idx:06d}.mp4"
    if not p.exists():
        p = repo_root / f"videos/chunk-{ep_idx // 300:03d}/video.{view}/episode_{ep_idx:06d}.mp4"
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t = min(max(0, t), max(0, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cv2 failed at {p} t={t}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_obs(state_53: np.ndarray, frames: dict, task_str: str) -> dict:
    obs = {}
    for k, s, e in STATE_KEYS:
        obs[k] = state_53[None, s:e].astype(np.float32)
    for view in ("left_view", "right_view", "wrist_view"):
        obs[f"video.{view}"] = frames[view][None]   # (1, H, W, 3) uint8
    obs["annotation.human.action.task_description"] = [task_str]
    return obs


def flatten_action_dict(d: dict) -> np.ndarray:
    """Reconstruct (T, 12) from per-key dict, preserving raw action layout."""
    T = None
    for key, s, e in ACTION_KEYS:
        if key in d:
            v = np.asarray(d[key])
            T = v.shape[0] if v.ndim >= 1 else 1
            break
    if T is None:
        return np.zeros((0, ACTION_DIM))
    out = np.zeros((T, ACTION_DIM), dtype=np.float32)
    for key, s, e in ACTION_KEYS:
        if key not in d:
            continue
        v = np.asarray(d[key])
        if v.ndim == 1:
            v = v[:, None]
        out[:, s:e] = v[:T, :e - s]
    return out


def per_t_inference(policy, obs, gt_chunk: np.ndarray):
    """Returns (losses[4], probs[4]) for this single timestep obs."""
    head = policy.model.action_head
    losses = []
    for k in range(4):
        head.moe_force_pick = k
        head.moe_inference_stochastic = False
        out = policy.get_action(obs)
        flat = flatten_action_dict(out)
        T_k = DECODER_T[k]
        if flat.shape[0] < T_k:
            losses.append(float("nan")); continue
        pred = flat[:T_k]
        tgt = compress_gt(gt_chunk, k)
        mse = float(np.mean((pred - tgt[:T_k]) ** 2))
        losses.append(mse)
    head.moe_force_pick = -1
    head.moe_inference_stochastic = False
    out = policy.get_action(obs)
    probs_raw = out.get("_moe_probs", None)
    if probs_raw is None:
        probs = [float("nan")] * 4
    else:
        probs = list(np.asarray(probs_raw).flatten()[:4].astype(float))
    return losses, probs


def cmd_collect(args):
    repo_root = Path(args.gt_root)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"; frames_dir.mkdir(exist_ok=True)
    cache_path = out_dir / "samples.jsonl"

    # resume keys: (ep, seg, t)
    done = set()
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                d = json.loads(line)
                done.add((d["ep"], d["seg"], d["t"]))
        print(f"[resume] {len(done)} samples cached")

    print(f"[load] {args.ckpt}")
    policy = load_model(args.ckpt, args.data_config)

    eps = list_episodes_for_task(repo_root, args.task)
    if args.episode_indices:
        wanted = [int(x) for x in args.episode_indices.split(",")]
        eps = [e for e in eps if e in wanted]
    else:
        eps = eps[args.ep_start: args.ep_start + args.n_episodes]
    print(f"[task] '{args.task}' -> {len(eps)} eps: {eps}")

    for ep_idx in eps:
        try:
            df = load_episode_df(repo_root, ep_idx)
        except Exception as e:
            print(f"  [skip ep {ep_idx}] {e}"); continue
        n_steps = len(df)
        # segment boundaries
        bounds = np.linspace(0, n_steps, args.n_segments + 1, dtype=int)
        # save segment-start frames once
        seg_starts = []
        for s in range(args.n_segments):
            ts = int(bounds[s])
            seg_starts.append(ts)
            fp = frames_dir / f"ep{ep_idx:04d}_seg{s}.png"
            if not fp.exists():
                try:
                    img = grab_frame(repo_root, ep_idx, "wrist_view", ts)
                    Image.fromarray(img).save(fp)
                except Exception as e:
                    print(f"  [warn ep {ep_idx} seg {s}] frame save: {e}")
        # iterate per segment, subsample timesteps
        for seg in range(args.n_segments):
            ts, te = int(bounds[seg]), int(bounds[seg + 1])
            ts_list = list(range(ts, max(ts + 1, te - H), max(1, args.subsample_stride)))
            for t in ts_list:
                if (ep_idx, seg, t) in done:
                    continue
                if t + H > n_steps:
                    continue
                gt = np.stack([np.asarray(df.iloc[t + i]["action"], dtype=np.float32) for i in range(H)])
                state = np.asarray(df.iloc[t]["observation.state"], dtype=np.float32)
                try:
                    frames = {
                        v: grab_frame(repo_root, ep_idx, v, t)
                        for v in ("left_view", "right_view", "wrist_view")
                    }
                except Exception as e:
                    print(f"  [skip ep {ep_idx} seg {seg} t {t}] frames: {e}"); continue
                obs = build_obs(state, frames, args.task_prompt)
                try:
                    losses, probs = per_t_inference(policy, obs, gt)
                except Exception as e:
                    print(f"  [ep {ep_idx} seg {seg} t {t}] infer: {e}"); raise
                rec = {
                    "ep": int(ep_idx), "seg": int(seg), "t": int(t),
                    "n_steps": int(n_steps), "seg_starts": seg_starts,
                    "losses": [float(x) for x in losses],
                    "probs":  [float(x) for x in probs],
                }
                with open(cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n"); f.flush()
                L = ",".join(f"{n}={l:.4f}" for n, l in zip(DECODER_NAMES, losses))
                P = ",".join(f"{n}={p:.2f}" for n, p in zip(DECODER_NAMES, probs))
                print(f"  ep{ep_idx} seg{seg} t={t:4d}: L[{L}] P[{P}]")

    print(f"[collect DONE] {cache_path}")


def cmd_plot(args):
    import matplotlib.pyplot as plt
    cache_path = Path(args.output_dir) / "samples.jsonl"
    figs_dir = Path(args.output_dir) / "figs"; figs_dir.mkdir(exist_ok=True)
    frames_dir = Path(args.output_dir) / "frames"

    by_ep: dict[int, list[dict]] = {}
    with open(cache_path) as f:
        for line in f:
            r = json.loads(line)
            by_ep.setdefault(r["ep"], []).append(r)
    print(f"[plot] {len(by_ep)} episodes in cache  (style={args.style})")

    # ----- style: timeseries -----
    def render_ep_timeseries(ep_idx: int, records: list[dict]):
        # sort by t
        records = sorted(records, key=lambda r: r["t"])
        ts = np.array([r["t"] for r in records])
        L = np.array([r["losses"] for r in records])  # (n, 4)
        P = np.array([r["probs"] for r in records])
        p_comp = P[:, COMPRESSED_IDX[0]] + P[:, COMPRESSED_IDX[1]]
        comp_mean = np.nanmean(L[:, list(COMPRESSED_IDX)], axis=1)
        raw_mean = np.nanmean(L[:, list(RAW_IDX)], axis=1)
        loss_gap = raw_mean - comp_mean   # positive = compressed wins
        # segment-start frame timesteps
        seg_starts = next((r["seg_starts"] for r in records if "seg_starts" in r), None)
        n_seg = len(seg_starts) if seg_starts else 5

        # figure: top row = N frame thumbnails, bottom row = curve(s)
        # use gridspec so frames are a thin row above wide curve panel
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(2.2 * n_seg, 5.0))
        gs = GridSpec(2, n_seg, figure=fig, height_ratios=[1.0, 1.4], hspace=0.35, wspace=0.15)
        # top: frames
        for s in range(n_seg):
            ax = fig.add_subplot(gs[0, s])
            ts0 = seg_starts[s] if seg_starts else 0
            fp = frames_dir / f"ep{ep_idx:04d}_seg{s}.png"
            if fp.exists():
                ax.imshow(Image.open(fp))
            else:
                ax.text(0.5, 0.5, "no frame", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t={ts0}", fontsize=8)
        # bottom: curve (spans full width)
        ax = fig.add_subplot(gs[1, :])
        ax.plot(ts, p_comp, color="#1a8", lw=1.8, marker="o", ms=3, label="p(compressed)")
        ax.axhline(0.5, color="#888", ls="--", lw=0.7)
        ax.set_xlabel("timestep")
        ax.set_ylabel("p(compressed)", color="#1a8")
        # auto-zoom Y so trend variation is visible
        pmin, pmax = float(p_comp.min()), float(p_comp.max())
        pad = max(0.03, (pmax - pmin) * 0.2)
        ax.set_ylim(max(0, pmin - pad), min(1, pmax + pad))
        ax.tick_params(axis='y', labelcolor="#1a8")
        ax.grid(alpha=0.25)
        # vertical guides at segment starts
        if seg_starts:
            for ts0 in seg_starts:
                ax.axvline(ts0, color="#bbb", lw=0.5)
        # optional loss_gap overlay
        if not args.hide_loss_gap:
            ax2 = ax.twinx()
            ax2.plot(ts, loss_gap, color="#c33", lw=1.2, marker="s", ms=2.5, alpha=0.7,
                     label="loss_gap (raw-comp)")
            ax2.axhline(0.0, color="#c33", ls=":", lw=0.5, alpha=0.5)
            ax2.set_ylabel("loss gap (raw - comp)", color="#c33")
            ax2.tick_params(axis='y', labelcolor="#c33")
            # combine legends
            lns1, lbl1 = ax.get_legend_handles_labels()
            lns2, lbl2 = ax2.get_legend_handles_labels()
            ax.legend(lns1 + lns2, lbl1 + lbl2, loc="upper left", fontsize=8)
        else:
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Episode {ep_idx} — p(compressed) over time", fontsize=11)
        out = figs_dir / f"ep{ep_idx:04d}_timeseries.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        # quick aggregate
        frac_high = float(np.mean(p_comp > 0.5))
        print(f"  ep{ep_idx} -> {out}  (frac p(comp)>0.5: {frac_high:.2f}, p(comp) mean={p_comp.mean():.2f})")

    # ----- style: segments (original) -----
    def render_ep_segments(ep_idx: int, records: list[dict]):
        seg_data: dict[int, list[dict]] = {}
        for r in records:
            seg_data.setdefault(r["seg"], []).append(r)
        seg_starts_per = next((r["seg_starts"] for r in records if "seg_starts" in r), None)
        n_seg = max(seg_data.keys()) + 1
        n_cols = n_seg

        fig, axes = plt.subplots(3, n_cols, figsize=(2.2 * n_cols, 7.0), squeeze=False)
        # row0 = frame, row1 = comp vs raw avg loss, row2 = per-decoder router probs

        comp_losses_per_seg = []
        raw_losses_per_seg = []
        probs_per_seg = []

        for s in range(n_seg):
            rs = seg_data.get(s, [])
            if not rs:
                comp_losses_per_seg.append(np.nan)
                raw_losses_per_seg.append(np.nan)
                probs_per_seg.append([np.nan] * 4)
                continue
            L = np.array([r["losses"] for r in rs])  # (n, 4)
            P = np.array([r["probs"] for r in rs])
            comp = np.nanmean(L[:, list(COMPRESSED_IDX)])
            raw = np.nanmean(L[:, list(RAW_IDX)])
            comp_losses_per_seg.append(comp)
            raw_losses_per_seg.append(raw)
            probs_per_seg.append(list(np.nanmean(P, axis=0)))

            # row 0: frame
            ax = axes[0][s]
            ts0 = seg_starts_per[s] if seg_starts_per else rs[0]["t"]
            fp = frames_dir / f"ep{ep_idx:04d}_seg{s}.png"
            if fp.exists():
                ax.imshow(Image.open(fp))
            else:
                ax.text(0.5, 0.5, "no frame", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"seg {s}\nt={ts0}", fontsize=9)

            # row 1: comp vs raw loss
            ax = axes[1][s]
            ax.bar(["comp\n(m8+m4)", "raw\n(r16+n8)"], [comp, raw],
                   color=["#1a8", "#888"])
            ax.set_ylabel("MSE" if s == 0 else "")
            ax.tick_params(axis='x', labelsize=7); ax.tick_params(axis='y', labelsize=7)
            # annotate ratio
            if raw > 0:
                ax.set_title(f"comp/raw={comp/raw:.2f}", fontsize=8)

            # row 2: per-decoder router probs
            ax = axes[2][s]
            ax.bar(DECODER_NAMES, probs_per_seg[s],
                   color=["#888", "#1a8", "#1a8", "#888"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("router prob" if s == 0 else "")
            ax.tick_params(axis='x', labelsize=7); ax.tick_params(axis='y', labelsize=7)
            # annotate compressed fraction
            comp_frac = probs_per_seg[s][1] + probs_per_seg[s][2]
            ax.set_title(f"p(comp)={comp_frac:.2f}", fontsize=8)

        fig.suptitle(f"Episode {ep_idx} — router segment analysis", fontsize=11)
        plt.tight_layout()
        out = figs_dir / f"ep{ep_idx:04d}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        # quick aggregate score for this episode
        score_easy = 0; score_hard = 0
        for s in range(n_seg):
            c, r = comp_losses_per_seg[s], raw_losses_per_seg[s]
            if np.isnan(c) or np.isnan(r): continue
            p_comp = probs_per_seg[s][1] + probs_per_seg[s][2]
            if c < r and p_comp > 0.5:
                score_easy += 1
            if c > r and p_comp < 0.5:
                score_hard += 1
        print(f"  ep{ep_idx} -> {out}  (consistent_easy={score_easy}, consistent_hard={score_hard})")

    render_fn = render_ep_timeseries if args.style == "timeseries" else render_ep_segments
    for ep_idx, recs in sorted(by_ep.items()):
        render_fn(ep_idx, recs)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--gt-root", required=True)
    c.add_argument("--ckpt", required=True)
    c.add_argument("--data-config", default="single_panda_gripper")
    c.add_argument("--task", default="CloseDoubleDoor",
                   help="Substring to match in tasks list (e.g. CloseDoubleDoor).")
    c.add_argument("--task-prompt", default="close the cabinet doors",
                   help="Language annotation string fed to the policy.")
    c.add_argument("--n-episodes", type=int, default=5)
    c.add_argument("--ep-start", type=int, default=0)
    c.add_argument("--episode-indices", default="")
    c.add_argument("--n-segments", type=int, default=5)
    c.add_argument("--subsample-stride", type=int, default=8,
                   help="Sample every Nth timestep within each segment.")
    c.add_argument("--output-dir", required=True)

    pl = sub.add_parser("plot")
    pl.add_argument("--output-dir", required=True)
    pl.add_argument("--style", choices=["segments", "timeseries"], default="timeseries")
    pl.add_argument("--hide-loss-gap", action="store_true",
                    help="In timeseries style, drop the secondary loss_gap curve.")

    cb = sub.add_parser("combine",
                        help="Render multi-episode combined timeseries figure.")
    cb.add_argument("--inputs", required=True,
                    help="Comma-separated list of 'output_dir:ep_idx:label' triples. "
                         "Label is optional (falls back to dir basename).")
    cb.add_argument("--captions", default="",
                    help="Semicolon-separated per-item caption sets. Each set is "
                         "pipe-separated captions (one per segment), in the same "
                         "order as --inputs. Empty set '' uses no captions for that "
                         "item. Example: 'approach|reach|carry|over sink|release;;...'")
    cb.add_argument("--output", required=True)
    cb.add_argument("--hide-loss-gap", action="store_true")
    cb.add_argument("--gt-root", default="",
                    help="If set, auto-detect strongest p_comp dip per episode and "
                         "grab that timestep's frame from the source video, "
                         "inserting it (time-sorted) into the thumbnail row.")
    cb.add_argument("--n-dips", type=int, default=1,
                    help="Number of dip frames to add per episode (default 1).")

    kit = sub.add_parser("render_kit",
                         help="For each input episode emit a directory containing "
                              "individual frame PNGs + curve.png + combined.png. "
                              "Each kit dir is self-contained for downstream editing.")
    kit.add_argument("--inputs", required=True,
                     help="Comma-separated 'output_dir:ep_idx:label' triples.")
    kit.add_argument("--captions", default="",
                     help="Semicolon-separated per-item caption sets (pipe-separated).")
    kit.add_argument("--gt-root", default="",
                     help="If set, auto-detect strongest p_comp dip and grab that frame.")
    kit.add_argument("--n-dips", type=int, default=1)
    kit.add_argument("--out-root", required=True,
                     help="Root dir; one sub-dir per episode kit will be created.")
    kit.add_argument("--hide-loss-gap", action="store_true")

    args = p.parse_args()
    if args.cmd == "collect": cmd_collect(args)
    elif args.cmd == "combine": cmd_combine(args)
    elif args.cmd == "render_kit": cmd_render_kit(args)
    else: cmd_plot(args)


def cmd_render_kit(args):
    """For each input episode, write a self-contained kit dir:
        <out_root>/<label>/
            frames/seg{N}_t{T}.png
            frames/dip_t{T}.png       (if dip found)
            curve.png                  (p_comp curve, full width)
            combined.png               (frame strip + curve panel)
            meta.json                  (record of timesteps, captions, source dir)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import shutil

    items = []
    for spec in args.inputs.split(","):
        parts = spec.split(":")
        d = Path(parts[0]); ep = int(parts[1])
        label = parts[2] if len(parts) > 2 else d.name
        items.append((d, ep, label))
    caption_sets = []
    if args.captions:
        for s in args.captions.split(";"):
            caption_sets.append([c.strip() for c in s.split("|")] if s else [])
    while len(caption_sets) < len(items):
        caption_sets.append([])

    gt_root = Path(args.gt_root) if args.gt_root else None
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    for i, (src_dir, ep, label) in enumerate(items):
        recs = []
        with open(src_dir / "samples.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["ep"] == ep: recs.append(r)
        if not recs:
            print(f"[skip] {src_dir}:ep{ep} no records"); continue
        recs.sort(key=lambda r: r["t"])
        seg_starts = next((r["seg_starts"] for r in recs if "seg_starts" in r), None)
        n_seg = len(seg_starts) if seg_starts else 5

        kit_dir = out_root / label
        kit_dir.mkdir(parents=True, exist_ok=True)
        (kit_dir / "frames").mkdir(exist_ok=True)

        caps = caption_sets[i] if i < len(caption_sets) else []
        # ----- frames -----
        frames_meta = []
        src_frames = src_dir / "frames"
        for s in range(n_seg):
            ts0 = seg_starts[s]
            src = src_frames / f"ep{ep:04d}_seg{s}.png"
            dst = kit_dir / "frames" / f"seg{s}_t{ts0}.png"
            if src.exists():
                shutil.copy(src, dst)
            frames_meta.append({"t": ts0, "label": f"t={ts0}",
                                "caption": caps[s] if s < len(caps) and caps[s] else "",
                                "is_dip": False, "path": str(dst.relative_to(kit_dir))})

        ts_arr = np.array([r["t"] for r in recs])
        probs = np.array([r["probs"] for r in recs])
        losses = np.array([r["losses"] for r in recs])
        p_comp = probs[:, COMPRESSED_IDX[0]] + probs[:, COMPRESSED_IDX[1]]
        if gt_root is not None and args.n_dips > 0:
            used = {f["t"] for f in frames_meta}
            for idx in np.argsort(p_comp):
                if sum(1 for f in frames_meta if f["is_dip"]) >= args.n_dips: break
                t_dip = int(ts_arr[idx])
                if any(abs(t_dip - u) < 20 for u in used): continue
                try:
                    arr = grab_frame(gt_root, ep, "wrist_view", t_dip)
                except Exception as e:
                    print(f"  [warn] dip {ep} t={t_dip}: {e}"); continue
                dst = kit_dir / "frames" / f"dip_t{t_dip}.png"
                Image.fromarray(arr).save(dst)
                frames_meta.append({"t": t_dip, "label": f"t={t_dip}",
                                    "caption": f"p(comp) dip ({p_comp[idx]:.2f})",
                                    "is_dip": True, "path": str(dst.relative_to(kit_dir))})
                used.add(t_dip)
        frames_meta.sort(key=lambda f: f["t"])

        # ----- curve.png -----
        fig, ax = plt.subplots(figsize=(2.4 * n_seg, 2.2))
        ax.plot(ts_arr, p_comp, color="#1a8", lw=1.8, marker="o", ms=3, label="p(compressed)")
        ax.axhline(0.5, color="#888", ls="--", lw=0.7)
        pmin, pmax = float(p_comp.min()), float(p_comp.max())
        pad = max(0.03, (pmax - pmin) * 0.2)
        ax.set_ylim(max(0, pmin - pad), min(1, pmax + pad))
        ax.set_ylabel("p(compressed)", color="#1a8")
        ax.tick_params(axis='y', labelcolor="#1a8")
        ax.grid(alpha=0.25)
        if seg_starts:
            for ts0 in seg_starts: ax.axvline(ts0, color="#bbb", lw=0.5)
        for f in frames_meta:
            if f["is_dip"]:
                ax.axvline(f["t"], color="#c33", lw=1.0, alpha=0.6, ls="--")
        if not args.hide_loss_gap:
            comp_mean = np.nanmean(losses[:, list(COMPRESSED_IDX)], axis=1)
            raw_mean = np.nanmean(losses[:, list(RAW_IDX)], axis=1)
            loss_gap = raw_mean - comp_mean
            ax2 = ax.twinx()
            ax2.plot(ts_arr, loss_gap, color="#c33", lw=1.2, marker="s", ms=2.5,
                     alpha=0.7, label="loss_gap (raw-comp)")
            ax2.axhline(0.0, color="#c33", ls=":", lw=0.5, alpha=0.5)
            ax2.set_ylabel("loss gap", color="#c33")
            ax2.tick_params(axis='y', labelcolor="#c33")
            lns1, lbl1 = ax.get_legend_handles_labels()
            lns2, lbl2 = ax2.get_legend_handles_labels()
            ax.legend(lns1 + lns2, lbl1 + lbl2, loc="upper left", fontsize=8)
        else:
            ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{label} — ep{ep}", fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(kit_dir / "curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # ----- combined.png -----
        THUMB_W = 3.2
        n_cols = max(1, len(frames_meta))
        fig = plt.figure(figsize=(THUMB_W * n_cols, THUMB_W + 2.0))
        gs = GridSpec(2, n_cols, figure=fig,
                      height_ratios=[THUMB_W, 2.0], hspace=0.9, wspace=0.06)
        for s, f in enumerate(frames_meta):
            ax = fig.add_subplot(gs[0, s])
            fp = kit_dir / f["path"]
            if fp.exists():
                ax.imshow(Image.open(fp))
            else:
                ax.text(0.5, 0.5, "no frame", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            t_str = f["label"]
            if f["caption"]:
                t_str += f"\n{f['caption']}"
            color = "#c33" if f["is_dip"] else "black"
            ax.set_title(t_str, fontsize=9, color=color)
        ax = fig.add_subplot(gs[1, :])
        ax.plot(ts_arr, p_comp, color="#1a8", lw=1.8, marker="o", ms=3, label="p(compressed)")
        ax.axhline(0.5, color="#888", ls="--", lw=0.7)
        ax.set_ylim(max(0, pmin - pad), min(1, pmax + pad))
        ax.set_ylabel("p(compressed)", color="#1a8")
        ax.tick_params(axis='y', labelcolor="#1a8")
        ax.grid(alpha=0.25)
        if seg_starts:
            for ts0 in seg_starts: ax.axvline(ts0, color="#bbb", lw=0.5)
        for f in frames_meta:
            if f["is_dip"]:
                ax.axvline(f["t"], color="#c33", lw=1.0, alpha=0.6, ls="--")
        if not args.hide_loss_gap:
            comp_mean = np.nanmean(losses[:, list(COMPRESSED_IDX)], axis=1)
            raw_mean = np.nanmean(losses[:, list(RAW_IDX)], axis=1)
            loss_gap = raw_mean - comp_mean
            ax2 = ax.twinx()
            ax2.plot(ts_arr, loss_gap, color="#c33", lw=1.2, marker="s", ms=2.5,
                     alpha=0.7, label="loss_gap (raw-comp)")
            ax2.axhline(0.0, color="#c33", ls=":", lw=0.5, alpha=0.5)
            ax2.set_ylabel("loss gap", color="#c33")
            ax2.tick_params(axis='y', labelcolor="#c33")
            lns1, lbl1 = ax.get_legend_handles_labels()
            lns2, lbl2 = ax2.get_legend_handles_labels()
            ax.legend(lns1 + lns2, lbl1 + lbl2, loc="upper left", fontsize=8)
        else:
            ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{label} — ep{ep}", fontsize=10, loc="left")
        fig.savefig(kit_dir / "combined.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # ----- meta.json -----
        meta = {
            "label": label, "src_dir": str(src_dir), "ep": ep,
            "frames": frames_meta,
            "p_comp_min": float(p_comp.min()),
            "p_comp_max": float(p_comp.max()),
            "p_comp_dynamic_range": float(p_comp.max() - p_comp.min()),
            "p_comp_mean": float(p_comp.mean()),
        }
        with open(kit_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [kit] {kit_dir}  dr={meta['p_comp_dynamic_range']:.3f}")
    print(f"[render_kit DONE] {len(items)} kits at {out_root}")


def cmd_combine(args):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    items = []  # list of (dir, ep, label)
    for spec in args.inputs.split(","):
        parts = spec.split(":")
        d = Path(parts[0])
        ep = int(parts[1])
        label = parts[2] if len(parts) > 2 else d.name
        items.append((d, ep, label))

    # Parse captions: one set per item, sets separated by ';', captions by '|'.
    caption_sets = []
    if args.captions:
        for s in args.captions.split(";"):
            caption_sets.append([c.strip() for c in s.split("|")] if s else [])
    while len(caption_sets) < len(items):
        caption_sets.append([])

    # Load each
    per_item = []
    for d, ep, lbl in items:
        recs = []
        with open(d / "samples.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["ep"] == ep:
                    recs.append(r)
        recs.sort(key=lambda r: r["t"])
        if not recs:
            print(f"[warn] no records for {d}:ep{ep}")
            continue
        seg_starts = next((r["seg_starts"] for r in recs if "seg_starts" in r), None)
        n_seg = len(seg_starts) if seg_starts else 5
        per_item.append({"dir": d, "ep": ep, "label": lbl, "recs": recs,
                         "seg_starts": seg_starts, "n_seg": n_seg})

    N = len(per_item)
    if N == 0:
        print("[combine] nothing to render")
        return

    # Build per-item ordered list of (t, frame_label, caption, frame_path_or_array)
    gt_root = Path(args.gt_root) if args.gt_root else None
    for i, it in enumerate(per_item):
        recs = it["recs"]
        ts_arr = np.array([r["t"] for r in recs])
        probs = np.array([r["probs"] for r in recs])
        p_comp = probs[:, COMPRESSED_IDX[0]] + probs[:, COMPRESSED_IDX[1]]
        frames_dir = it["dir"] / "frames"
        seg_starts = it["seg_starts"] or [recs[0]["t"]] * it["n_seg"]
        caps = caption_sets[i] if i < len(caption_sets) else []
        # Segment-start frames
        frames = []
        for s in range(it["n_seg"]):
            cap = caps[s] if s < len(caps) and caps[s] else ""
            frames.append({"t": seg_starts[s], "label": f"t={seg_starts[s]}",
                           "caption": cap,
                           "img": frames_dir / f"ep{it['ep']:04d}_seg{s}.png",
                           "is_dip": False})
        # Auto-detect dips and grab their frames live
        if gt_root is not None and args.n_dips > 0:
            used_ts = {f["t"] for f in frames}
            dip_order = np.argsort(p_comp)   # ascending p_comp
            picked = 0
            for idx in dip_order:
                if picked >= args.n_dips: break
                t_dip = int(ts_arr[idx])
                # require min distance from already-chosen frame timesteps
                if any(abs(t_dip - u) < 20 for u in used_ts): continue
                try:
                    arr = grab_frame(gt_root, it["ep"], "wrist_view", t_dip)
                except Exception as e:
                    print(f"  [warn] dip frame grab failed ep{it['ep']} t={t_dip}: {e}")
                    continue
                frames.append({"t": t_dip,
                               "label": f"t={t_dip}",
                               "caption": f"p(comp) dip ({p_comp[idx]:.2f})",
                               "img": arr, "is_dip": True})
                used_ts.add(t_dip); picked += 1
        # Sort frames by t for natural reading
        frames.sort(key=lambda f: f["t"])
        it["frames_list"] = frames

    n_cols = max(len(it["frames_list"]) for it in per_item)
    # Bigger thumbnails: 3.2" wide each
    THUMB_W = 3.2
    fig_w = THUMB_W * n_cols
    fig = plt.figure(figsize=(fig_w, (THUMB_W + 2.0) * N))
    gs = GridSpec(2 * N, n_cols, figure=fig,
                  height_ratios=[THUMB_W, 2.0] * N, hspace=0.9, wspace=0.06)

    for i, it in enumerate(per_item):
        fl = it["frames_list"]
        for s, f in enumerate(fl):
            ax = fig.add_subplot(gs[2 * i, s])
            img_src = f["img"]
            try:
                if isinstance(img_src, Path):
                    ax.imshow(Image.open(img_src))
                else:
                    ax.imshow(img_src)
            except Exception:
                ax.text(0.5, 0.5, "no frame", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            t_str = f["label"]
            if f["caption"]:
                t_str += f"\n{f['caption']}"
            color = "#c33" if f["is_dip"] else "black"
            ax.set_title(t_str, fontsize=9, color=color)
        # row 2: curve
        ax = fig.add_subplot(gs[2 * i + 1, :])
        ts = np.array([r["t"] for r in it["recs"]])
        L = np.array([r["losses"] for r in it["recs"]])
        P = np.array([r["probs"] for r in it["recs"]])
        p_comp = P[:, COMPRESSED_IDX[0]] + P[:, COMPRESSED_IDX[1]]
        ax.plot(ts, p_comp, color="#1a8", lw=1.8, marker="o", ms=3, label="p(compressed)")
        ax.axhline(0.5, color="#888", ls="--", lw=0.7)
        pmin, pmax = float(p_comp.min()), float(p_comp.max())
        pad = max(0.03, (pmax - pmin) * 0.2)
        ax.set_ylim(max(0, pmin - pad), min(1, pmax + pad))
        ax.set_ylabel("p(compressed)", color="#1a8")
        ax.tick_params(axis='y', labelcolor="#1a8")
        # Drop the redundant 'timestep' xlabel — the next-episode "t=" frame
        # titles below already make the axis meaning obvious, and the label
        # otherwise collides with those titles.
        ax.grid(alpha=0.25)
        seg_starts_i = it["seg_starts"] or []
        for ts0 in seg_starts_i:
            ax.axvline(ts0, color="#bbb", lw=0.5)
        # red guide for each dip frame
        for f in it.get("frames_list", []):
            if f.get("is_dip"):
                ax.axvline(f["t"], color="#c33", lw=1.0, alpha=0.6, ls="--")
        if not args.hide_loss_gap:
            comp_mean = np.nanmean(L[:, list(COMPRESSED_IDX)], axis=1)
            raw_mean = np.nanmean(L[:, list(RAW_IDX)], axis=1)
            loss_gap = raw_mean - comp_mean
            ax2 = ax.twinx()
            ax2.plot(ts, loss_gap, color="#c33", lw=1.2, marker="s", ms=2.5, alpha=0.7,
                     label="loss_gap (raw-comp)")
            ax2.axhline(0.0, color="#c33", ls=":", lw=0.5, alpha=0.5)
            ax2.set_ylabel("loss gap", color="#c33")
            ax2.tick_params(axis='y', labelcolor="#c33")
            lns1, lbl1 = ax.get_legend_handles_labels()
            lns2, lbl2 = ax2.get_legend_handles_labels()
            ax.legend(lns1 + lns2, lbl1 + lbl2, loc="upper left", fontsize=8)
        else:
            ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{it['label']} — ep{it['ep']}", fontsize=10, loc="left")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[combine] saved {out}")


if __name__ == "__main__":
    main()
