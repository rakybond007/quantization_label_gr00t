"""Anchor-and-interpolate action compression.

Instead of merging a fixed K steps, let the trajectory choose its own segment
boundaries: keep extending a segment while a straight line between its two
endpoints reproduces the actions in between to within `eps`, and cut where it
stops doing so. Smooth stretches collapse a lot, corners barely at all.

    anchor t, length L = 1
    interpolate a[t+1..t+L-1] linearly between a[t] and a[t+L]
    err = max |interpolated - original|
    err > eps  ->  commit the segment, new anchor
    err <= eps ->  L += 1, try again

Two error spaces, because a delta action and an absolute target mean different
things by "the line between two actions":

  action  — interpolate the commanded values themselves. For deltas this is
            velocity space: it asks whether the speed profile is linear.
  path    — integrate the deltas first and interpolate the cumulative
            displacement. This asks whether the arm's PATH is straight, which
            is what decides where it actually ends up.

`mode="absolute"` skips the integration: the commands already are positions.

What gets emitted per segment is unchanged from the fixed-K aggregator —
continuous keys sum (deltas compose), discrete keys latch to the last value —
so this is a drop-in alternative to `compress_chunk`, differing only in where
the boundaries fall.

The ratio cap is fractional and global rather than per segment. A segment
length is an integer, so no per-segment rule can average 2.5; instead the
segmentation runs free and then the longest segments are split until
T_in / T_out is within the cap. That also makes the cap a guarantee rather
than a hope.
"""
import numpy as np


def _interp_error(seg, space="action"):
    """Max deviation of the interior from the straight line joining the ends.

    seg: (L+1, D) — the anchor, the interior, and the closing frame.
    """
    L = seg.shape[0] - 1
    if L < 2:
        return 0.0                      # nothing in between to be wrong about
    if space == "path":
        # Cumulative displacement, anchored at 0 so both curves start together.
        seg = np.cumsum(seg, axis=0)
        seg = seg - seg[0]
    t = np.linspace(0.0, 1.0, L + 1)[1:-1, None]
    line = seg[0][None, :] + t * (seg[-1] - seg[0])[None, :]
    return float(np.abs(seg[1:-1] - line).max())


def segment(cont, disc, eps, kmax, space="action"):
    """Segment boundaries chosen by interpolation error.

    cont: (T, Dc) continuous actions, disc: (T, Dd) discrete (may be empty).
    Returns a list of (start, end) half-open spans covering 0..T.
    """
    T = cont.shape[0]
    spans, i = [], 0
    while i < T:
        # L is the largest length whose OWN interpolation error is within eps.
        # error(L) reads cont[i .. i+L], so error(1) has no interior and is
        # trivially 0 — a length-1 segment is the original action, uncompressed.
        # Growing to L+1 must be checked against error(L+1) BEFORE accepting it;
        # testing after accepting committed every segment to length >= 2, which
        # pinned the achievable ratio at 2.0 and made the chunk-level cap the
        # only thing deciding anything.
        L = 1
        while i + L + 1 <= T and L + 1 <= kmax:
            # A gripper-like transition ends the segment: interpolating across
            # the moment a grasp opens or closes is not a smoothness question.
            if disc.size and np.abs(disc[i + L] - disc[i + L - 1]).max() > 0.5:
                break
            if _interp_error(cont[i:i + L + 2], space) > eps:
                break
            L += 1
        spans.append((i, min(i + L, T)))
        i += L
    return spans


def _cap_ratio(spans, T, ratio_max):
    """Split the longest spans until T / len(spans) <= ratio_max.

    Splitting the longest first removes the most ratio per split and cuts where
    a straight line was carrying the most steps — the place a fixed-K merge
    would have cut anyway.
    """
    if ratio_max is None or ratio_max <= 0:
        return spans
    need = int(np.ceil(T / float(ratio_max)))
    spans = list(spans)
    while len(spans) < need:
        w = max(range(len(spans)), key=lambda k: spans[k][1] - spans[k][0])
        s, e = spans[w]
        if e - s < 2:
            break                        # nothing left to split
        m = s + (e - s) // 2
        spans[w:w + 1] = [(s, m), (m, e)]
    return spans


def interp_compress_chunk(chunk_dict, eps, ratio_max=2.5, kmax=8,
                          discrete_keys=(), space="action", mode="delta",
                          return_blocks=False):
    """Drop-in alternative to compress_chunk with data-chosen boundaries."""
    keys = list(chunk_dict.keys())
    arrs = {}
    for k, v in chunk_dict.items():
        v = np.asarray(v, dtype=float)
        arrs[k] = v if v.ndim >= 2 else v[..., None]
    T = next(iter(arrs.values())).shape[0]
    cont_k = [k for k in keys if k not in discrete_keys]
    disc_k = [k for k in keys if k in discrete_keys]

    cont = np.concatenate([arrs[k] for k in cont_k], axis=1) if cont_k else np.zeros((T, 0))
    disc = np.concatenate([arrs[k] for k in disc_k], axis=1) if disc_k else np.zeros((T, 0))

    spans = segment(cont, disc, eps, kmax, "path" if mode == "delta" and space == "path" else space)
    spans = _cap_ratio(spans, T, ratio_max)

    out = {k: [] for k in keys}
    for s, e in spans:
        for k in cont_k:
            blk = arrs[k][s:e]
            out[k].append(blk.sum(axis=0) if mode == "delta" else blk[-1])
        for k in disc_k:
            out[k].append(arrs[k][e - 1])
    res = {k: np.stack(v) for k, v in out.items()}
    return (res, spans) if return_blocks else res
