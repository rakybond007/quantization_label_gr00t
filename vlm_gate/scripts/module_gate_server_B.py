"""Serve the B-variant distilled gate (frozen GR00T Eagle backbone features ->
MLP head) behind the same POST /judge interface as vlm_gate.py.

Per request: 3 view images + instruction -> backbone forward (GPU) ->
masked-mean pooled feature -> FeatureGate head -> confidence. State inputs are
zero-filled: the Eagle backbone consumes only vision+language, state tensors
merely satisfy the transform pipeline (matches how features were extracted for
training, which used dataset state the backbone never sees).
"""
import argparse
import base64
import io
import json
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_gate_backbone_features import (  # noqa: E402
    load_policy, backbone_features, VIDEO_KEYS, STATE_KEYS, LANG_KEY)


class FeatureGate(nn.Module):
    def __init__(self, dim, hidden=(256, 64)):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="gate_head_best.pt")
    p.add_argument("--model-path", default=os.path.expanduser(
        "~/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000"))
    p.add_argument("--port", type=int, default=8130)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()

    policy = load_policy(args.model_path)
    ck = torch.load(args.ckpt, map_location="cpu")
    head = FeatureGate(ck["dim"]).eval()
    head.load_state_dict(ck["model"])

    # state dims from the modality slices the extractor uses
    state_dims = {}
    import json as _j
    meta = _j.load(open(os.path.join(
        "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300",
        "meta", "modality.json")))
    for sk in STATE_KEYS:
        sl = meta["state"][sk.split("state.")[1]]
        state_dims[sk] = sl["end"] - sl["start"]

    def judge(imgs_b64, instruction):
        obs = {}
        for i, dk in enumerate(VIDEO_KEYS):
            if i < len(imgs_b64):
                im = np.array(Image.open(io.BytesIO(base64.b64decode(imgs_b64[i]))).convert("RGB"))
            else:
                im = np.zeros((256, 256, 3), np.uint8)
            obs[dk] = im[None, None]  # (1,1,H,W,3)
        for sk, d in state_dims.items():
            obs[sk] = np.zeros((1, 1, d), dtype=np.float64)
        obs[LANG_KEY] = [[instruction]]
        feat = backbone_features(policy, obs)  # (1, D)
        with torch.no_grad():
            conf = torch.sigmoid(head(torch.from_numpy(feat))).item()
        return {"decision": "YES" if conf >= args.tau else "NO", "confidence": float(conf)}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path != "/judge":
                self.send_response(404); self.end_headers(); return
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n))
            out = judge(d.get("images_b64", []), d.get("instruction", ""))
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer((args.host, args.port), H)
    print(f"JUDGE READY (module gate B, dim={ck['dim']})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
