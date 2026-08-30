"""Variable-ratio action compression for ABSOLUTE joint targets (allex).

WHY SUMMING IS WRONG HERE
-------------------------
In the RoboCasa end-effector embodiment an action is a DELTA: "move 3 mm along
+x".  Running two steps as one there means executing their SUM, because the two
deltas compose.  The allex action vector is not a delta - every one of its 48
entries is an ABSOLUTE joint target that the low-level controller servos to.
Summing two absolute targets produces a pose at roughly twice the joint angles,
i.e. a command that is not on the demonstrated trajectory at all (and usually
outside the joint limits).  Compressing an absolute-target stream can only mean
SKIPPING targets: emit a subset of the original targets, and let the controller
travel further per control tick between the ones that survive.

FRACTIONAL RATIOS
-----------------
A ratio K means "emit one target per K input steps".  For integer K that is
plain striding.  For K = 2.5 there is no single stride, so we keep a running
real-valued cursor:

    idx = floor(pos + 0.5);  emit target[idx];  pos += K

which for K = 2.5 walks 0, 2.5, 5, 7.5, ... -> indices 0, 3, 5, 8, ... i.e. the
skip alternates 3, 2, 3, 2 and the AVERAGE ratio is exactly 2.5.

The cursor is carried ACROSS chunk boundaries (`pos` is returned and fed back
in).  Restarting the cursor at every 16-step chunk would round the leftover
fraction away each time and the realised ratio would drift toward the nearest
integer; carrying it makes the episode-level average equal the requested K.

`floor(pos + 0.5)` is used instead of `round()` on purpose: Python's `round`
does banker's rounding, so round(2.5) == 2 and round(7.5) == 8, which makes the
2/3 alternation irregular.

The FIRST target is always emitted (the cursor starts at 0) and the LAST target
of the episode is always emitted: the last target is where the arm is supposed
to end up, so dropping it would leave the robot short of the goal pose.
"""
from __future__ import annotations

import math

ALLOWED_RATIOS = (1.0, 2.0, 2.5, 3.0)


def snap_ratio(k, allowed=ALLOWED_RATIOS):
    """Snap a continuous ratio to the nearest allowed value."""
    return min(allowed, key=lambda a: (abs(a - float(k)), a))


def chunk_indices(chunk_start, chunk_len, K, pos=None):
    """Indices kept inside ONE chunk, plus the cursor to carry to the next.

    chunk_start : index of the chunk's first step in the episode
    chunk_len   : number of steps in the chunk
    K           : ratio for this chunk (any positive float)
    pos         : absolute real-valued cursor left over from the previous chunk.
                  None -> start exactly at chunk_start (episode start).
    returns (indices, next_pos)
    """
    if pos is None:
        pos = float(chunk_start)
    end = chunk_start + chunk_len
    out = []
    while True:
        i = int(math.floor(pos + 0.5))
        if i >= end:            # this emission belongs to the next chunk
            break
        out.append(i)
        pos += float(K)
    return out, pos


def compress_episode(n_steps, ratios, chunk=16, keep_last=True):
    """Full-episode target selection under a per-chunk ratio schedule.

    n_steps : number of absolute joint targets in the episode
    ratios  : list of ratios, one per chunk of `chunk` steps (last chunk may be
              short; if fewer ratios than chunks are given the last is reused)
    returns the sorted list of kept target indices.
    """
    idx, pos = [], None
    n_chunks = (n_steps + chunk - 1) // chunk
    for c in range(n_chunks):
        s = c * chunk
        L = min(chunk, n_steps - s)
        K = float(ratios[c]) if c < len(ratios) else float(ratios[-1] if len(ratios) else 1.0)
        got, pos = chunk_indices(s, L, K, pos)
        idx.extend(got)
    if not idx:
        idx = [0]
    if idx[0] != 0:
        idx.insert(0, 0)
    if keep_last and idx[-1] != n_steps - 1:
        K_last = float(ratios[-1]) if len(ratios) else 1.0
        # Snap the final emission onto the goal pose when that does not open a
        # gap much larger than K; only append when it would.
        if len(idx) > 1 and (n_steps - 1 - idx[-2]) <= math.ceil(1.5 * K_last):
            idx[-1] = n_steps - 1
        else:
            idx.append(n_steps - 1)
    return idx


def realised_ratio(n_steps, idx):
    """Average compression actually achieved: input steps per emitted target."""
    return n_steps / float(len(idx))


def _self_test():
    import numpy as np
    ok = True
    for K in ALLOWED_RATIOS:
        n = 200
        ratios = [K] * ((n + 15) // 16)
        idx = compress_episode(n, ratios, chunk=16)
        r = realised_ratio(n, idx)
        err = abs(r - K) / K
        first_ok = idx[0] == 0
        last_ok = idx[-1] == n - 1
        mono = all(b > a for a, b in zip(idx, idx[1:]))
        print(f"K={K:<4} kept={len(idx):4d}/{n}  realised={r:.4f}  err={err*100:.2f}%  "
              f"first={first_ok} last={last_ok} strictly_increasing={mono}")
        assert err <= 0.01, f"K={K}: realised {r} off by {err*100:.2f}% (>1%)"
        assert first_ok, f"K={K}: first target dropped"
        assert last_ok, f"K={K}: last target dropped"
        assert mono, f"K={K}: indices not strictly increasing"
        ok = ok and first_ok and last_ok
    # mixed schedule: the carry must not be reset at chunk boundaries
    n = 320
    ratios = [2.5, 1.0, 3.0, 2.5, 2.0, 2.5, 3.0, 1.0, 2.5, 2.0,
              2.5, 2.5, 3.0, 2.0, 1.0, 2.5, 2.5, 3.0, 2.0, 2.5]
    idx = compress_episode(n, ratios, chunk=16)
    exp = sum(16.0 / k for k in ratios)
    print(f"mixed schedule: kept={len(idx)} expected~{exp:.1f}  "
          f"first={idx[0]} last={idx[-1]}/{n-1}")
    # tolerance 4 (~2.5%): when K changes mid-cursor the emission that straddles
    # the boundary is charged to one chunk or the other, so the per-chunk sum is
    # only an estimate.  The carry keeps that error bounded instead of cumulative.
    assert abs(len(idx) - exp) <= 4, f"mixed: kept {len(idx)} vs expected {exp:.1f}"
    assert idx[0] == 0 and idx[-1] == n - 1
    # K=2.5 must alternate 3,2,3,2 (not drift to a constant stride)
    idx, _ = chunk_indices(0, 40, 2.5)
    d = list(np.diff(idx))
    print("K=2.5 step pattern:", d[:10])
    assert set(d) == {2, 3}, f"K=2.5 gaps should alternate 2/3, got {set(d)}"
    # skipping, never summing: every emitted target is an original target
    print("ALL RATIO TESTS PASSED")
    return ok


if __name__ == "__main__":
    _self_test()
