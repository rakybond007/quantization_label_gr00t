"""Does packing several chunks into one request change the labels?

Packing k chunks into one call divides the prompt cost by k, which is the
difference between $16 and $2 for a full RoboCasa pass. The question is whether
it costs anything in label quality.

The risk is anchoring: the model sees six consecutive moments of the same
episode at once and answers them as a set, so its answers may drift toward each
other rather than being decided on their own. If that happens, neighbours
*inside* a request agree more than neighbours that straddle a request boundary
— and the difference is the bias, because both pairs are equally close in time.

The API pass packed 6 chunks per request at stride 8, so request boundaries sit
every 6th chunk. Cosmos labelled the same chunks one at a time, which gives a
control: whatever step this test reports for cosmos is the ordinary smoothness
of the signal, not an artefact of packing.
"""
import json
import sys

import numpy as np
import pandas as pd

API = ("/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/output/"
       "_gate_distill/openai_batch_full/labels_api_none_full.jsonl")
COSMOS = ("/sjw_alinlab/home/hojin2/quantization_agent_workspace/assets/labels/"
          "robocasa/v6b_phase5_softA.parquet")
K, STRIDE = 6, 8


def load_api():
    rows = []
    for line in open(API):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("p_yes") is not None:
            rows.append((r["ep"], r["f"], float(r["p_yes"])))
    return pd.DataFrame(rows, columns=["ep", "f", "p"])


def gaps(df, col):
    """|difference| between neighbouring chunks, split by request boundary.

    Position within a request is (f / stride) mod K. A pair whose first member
    sits at position K-1 straddles a boundary; every other pair is internal.
    """
    inside, across = [], []
    for ep, g in df.groupby("ep"):
        g = g.sort_values("f")
        f = g.f.to_numpy()
        v = g[col].to_numpy()
        step = np.diff(f) == STRIDE          # only truly adjacent chunks
        d = np.abs(np.diff(v))
        pos = (f[:-1] // STRIDE) % K
        inside.append(d[step & (pos != K - 1)])
        across.append(d[step & (pos == K - 1)])
    return np.concatenate(inside), np.concatenate(across)


api = load_api()
print("API labels: %d rows, %d episodes" % (len(api), api.ep.nunique()))

cos = pd.read_parquet(COSMOS, columns=["episode_index", "frame_index", "p_yes"])
cos = cos.rename(columns={"episode_index": "ep", "frame_index": "f", "p_yes": "p"})
cos = cos.merge(api[["ep", "f"]], on=["ep", "f"], how="inner")   # same chunks only
print("cosmos labels on the same chunks: %d rows\n" % len(cos))

print("%-22s %10s %10s %9s %9s" % ("", "inside", "across", "ratio", "n inside/across"))
for name, df in (("API (6 per request)", api), ("cosmos (1 at a time)", cos)):
    ins, acr = gaps(df, "p")
    if len(ins) == 0 or len(acr) == 0:
        print("%-22s  (not enough adjacent pairs)" % name)
        continue
    print("%-22s %10.4f %10.4f %9.3f   %d / %d"
          % (name, ins.mean(), acr.mean(), ins.mean() / acr.mean(), len(ins), len(acr)))

print("\ninside  = mean |Δp| between neighbours in the same request")
print("across  = mean |Δp| between neighbours split by a request boundary")
print("ratio much below 1 on API but not on cosmos would mean packing smooths"
      " labels together; ratio near 1 on both means it does not.")
