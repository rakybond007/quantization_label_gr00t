"""Render an annotated video from EpisodeResult.frames + chunk decisions.
Each frame gets a corner overlay showing chunk index, timestep, and gate
decision (MERGE / KEEP). Returns a (T, H, W, 3) uint8 numpy array suitable
for wandb.Video(..., fps=20, format='mp4')."""

import numpy as np


def annotate_episode(result, fps_hint: int = 20) -> np.ndarray:
    if not result.frames:
        return np.zeros((1, 64, 64, 3), dtype=np.uint8)
    try:
        import cv2  # type: ignore
    except ImportError:
        # No annotation; return raw frames stacked.
        return np.stack(result.frames, axis=0)

    annotated = []
    # Build per-env-step lookup of (chunk_idx, decision, prob)
    step_to_chunk = {}
    for c in result.chunks:
        for fi in c.frame_indices:
            step_to_chunk[fi] = (c.chunk_idx, c.decision, c.prob)

    success_tag = "SUCCESS" if result.success else "FAIL"
    success_col = (0, 220, 0) if result.success else (50, 50, 220)
    bar_h = 36           # height of the dark info bar at the bottom
    pad = 8

    for i, frame in enumerate(result.frames):
        H, W = frame.shape[:2]
        # Pad the frame downward with a black bar so text doesn't overlap content
        canvas = np.zeros((H + bar_h, W, 3), dtype=np.uint8)
        canvas[:H] = frame
        canvas[H:H + bar_h] = (20, 20, 20)
        ci, dec, p = step_to_chunk.get(i, (-1, -1, 0.0))
        merge = (dec == 1)
        # Top-left: env-step + chunk
        cv2.putText(canvas, f"step {i}  chunk {ci}", (pad, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # Top-right: success indicator (visible throughout episode for context)
        if i == len(result.frames) - 1:
            cv2.putText(canvas, success_tag, (W - 130, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, success_col, 2, cv2.LINE_AA)
        # Bottom bar: gate decision + probability
        gate_text = f"GATE: {'MERGE' if merge else 'KEEP'}   P(merge)={p:.2f}"
        gate_col = (60, 220, 60) if merge else (200, 200, 200)
        cv2.putText(canvas, gate_text, (pad, H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gate_col, 2, cv2.LINE_AA)
        annotated.append(canvas)
    return np.stack(annotated, axis=0)


def merge_timeline_image(result):
    """Build a small timeline image: x = chunk index, y = decision (0/1).
    Returns numpy array (H, W, 3) suitable for wandb.Image."""
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    if not result.chunks:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    decisions = [c.decision for c in result.chunks]
    probs = [c.prob for c in result.chunks]
    n = len(decisions)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.15), 1.6), dpi=120)
    ax.bar(range(n), decisions, color=["#2ca02c" if d else "#aaaaaa" for d in decisions],
           width=0.9, edgecolor=None)
    ax.plot(range(n), probs, "o-", color="#1f77b4", markersize=2.5, lw=0.8,
            label="P(merge)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("chunk idx")
    ax.set_ylabel("decision (bar) / P(merge) (line)")
    ax.set_title(f"merge timeline | success={result.success} | env_steps={result.total_env_steps}",
                 fontsize=8)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    return img
