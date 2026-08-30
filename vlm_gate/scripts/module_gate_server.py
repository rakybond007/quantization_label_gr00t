"""Serve a distilled gate module behind the SAME POST /judge interface as
vlm_gate.py, so the gated eval stack runs unchanged with the lightweight
student in place of the heavy VLM judge.

Input: images_b64 (list, 3 views), instruction, guidance (ignored — the
student was distilled FROM the best guidance, it is baked in).
Output: {"decision", "confidence"} matching the VLM server contract.

Task conditioning: instruction -> MiniLM embedding. Known instructions hit the
precomputed table; unseen ones are embedded on the fly (MiniLM stays loaded,
CPU, ~ms) and cached.
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class SmallGate(nn.Module):
    def __init__(self, ch=32, temb_dim=0):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(9, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4),
                                 blk(ch * 4, ch * 4), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(ch * 4 + temb_dim, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, t=None):
        f = self.net(x).flatten(1)
        if t is not None and t.numel():
            f = torch.cat([f, t], dim=1)
        return self.head(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="gate_module.pt from train_gate_module.py")
    p.add_argument("--task-emb", required=True, help="task_embeddings.npz")
    p.add_argument("--port", type=int, default=8130)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    res = ck["res"]
    if ck.get("arch") == "lang":
        # A'-lang: frozen text-encoder emb -> trained linear proj inside model.
        from train_gate_module_lang import SmallGateLang
        temb_dim = ck["temb_in"]
        model = SmallGateLang(temb_in=ck["temb_in"], proj_dim=ck.get("proj_dim", 32)).to(dev).eval()
    else:
        temb_dim = ck.get("temb_dim", 0)
        if ck.get("encoder") == "dinov3s":
            # DINOv3 ViT-S/16 동결 + attention 풀링. transformers>=4.56 이 필요하므로
            # 이 프로세스는 오버레이를 PYTHONPATH 로 받아야 한다(평가 클라이언트는 4.51.3 고정).
            from train_gate_module import DinoGate
            model = DinoGate(temb_dim=temb_dim).to(dev).eval()
        else:
            model = SmallGate(temb_dim=temb_dim).to(dev).eval()
    model.load_state_dict(ck["model"])

    z = np.load(args.task_emb, allow_pickle=True)
    table = {t: e for t, e in zip(z["tasks"], z["emb"])}
    # MiniLM encoder for instructions absent from the table. Loaded eagerly when
    # possible (a lazy first-hit load would stall the single-threaded server
    # mid-rollout), but never fatally: this env pins numpy 1.23.5 for robocasa,
    # which breaks the AutoTokenizer dispatch in transformers 5.x. MiniLM is a
    # BERT model, so the concrete classes still import; if even that fails we
    # fall back to the nearest table entry rather than killing the eval.
    enc = None
    try:
        from transformers import AutoTokenizer, AutoModel
        enc = {"tok": AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
               "m": AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()}
    except Exception:
        try:
            from transformers.models.bert import BertTokenizerFast, BertModel
            enc = {"tok": BertTokenizerFast.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                   "m": BertModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()}
            print("[judge] MiniLM via BertTokenizerFast (AutoTokenizer unavailable)", flush=True)
        except Exception as e:
            print(f"[judge] WARNING no text encoder ({type(e).__name__}); "
                  "unseen instructions will fall back to the nearest known one", flush=True)

    _tok = {t: set(t.lower().split()) for t in table}

    def _nearest(instr):
        w = set(instr.lower().split())
        best = max(_tok, key=lambda t: len(w & _tok[t]) / max(len(w | _tok[t]), 1))
        print(f"[judge] unseen instruction {instr!r} -> nearest {best!r}", flush=True)
        return table[best]

    def embed(instr):
        if instr in table:
            return table[instr]
        if enc is None:
            table[instr] = _nearest(instr)
            return table[instr]
        with torch.no_grad():
            b = enc["tok"]([instr], padding=True, truncation=True, return_tensors="pt")
            h = enc["m"](**b).last_hidden_state
            mask = b.attention_mask.unsqueeze(-1).float()
            e = torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=-1)
        table[instr] = e[0].numpy().astype(np.float32)
        return table[instr]

    import cv2

    def judge(imgs_b64, instruction):
        planes = []
        for i in range(3):
            if i < len(imgs_b64):
                im = np.array(Image.open(io.BytesIO(base64.b64decode(imgs_b64[i]))).convert("RGB"))
                im = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
            else:
                im = np.zeros((res, res, 3), np.uint8)
            planes.append(im.transpose(2, 0, 1))
        x = torch.from_numpy(np.concatenate(planes)[None].astype(np.float32) / 255.0).to(dev)
        t = None
        if temb_dim:
            t = torch.from_numpy(embed(instruction)[None]).to(dev)
        with torch.no_grad():
            conf = torch.sigmoid(model(x, t)).item()
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
    print(f"JUDGE READY (module gate, {args.ckpt}, temb_dim={temb_dim})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
