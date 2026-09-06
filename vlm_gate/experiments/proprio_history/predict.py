"""Standalone inference adapter. Production servers/clients remain unchanged."""
import numpy as np
import torch

from model import SmallGateProprioHistory


class Predictor:
    def __init__(self, checkpoint, device="cuda"):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("format") != "proprio_history_v1":
            raise ValueError("not a proprio/history checkpoint")
        self.device = device
        self.model = SmallGateProprioHistory(**ck["config"])
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.to(device).eval()
        self.views = ck["views"]
        self.res = ck["res"]
        z = np.load(ck["task_emb_file"], allow_pickle=True)
        self.embeddings = {t:e for t,e in zip(z["tasks"], z["emb"])}

    def predict(self, images, instruction, state, previous_actions, mask, threshold=.5):
        """Three RGB uint8 HWC views, raw state, oldest->newest actual commands.

        All arrays belong to the same current observation/episode. No silent
        zero fallback for missing state/history or approximate text matching.
        Client applies camera orientation identical to its training images.
        A missing instruction embedding raises; extend embeddings explicitly.
        """
        import cv2
        if len(images) != 3 or instruction not in self.embeddings:
            raise ValueError("three views and a known instruction are required")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        planes=[]
        for im in images:
            im=np.asarray(im)
            if im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3:
                raise ValueError("images must be RGB uint8 HWC")
            planes.append(cv2.resize(im,(self.res,self.res),interpolation=cv2.INTER_AREA).transpose(2,0,1))
        arrays=[np.concatenate(planes).astype(np.float32)/255,
                np.asarray(self.embeddings[instruction],np.float32),
                np.asarray(state,np.float32),np.asarray(previous_actions,np.float32),
                np.asarray(mask,np.float32)]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("nonfinite input")
        if not np.isin(arrays[-1], [0,1]).all() or np.any(np.diff(arrays[-1]) < 0):
            raise ValueError("history mask must be left-padded zero then valid ones")
        inputs=[torch.as_tensor(np.ascontiguousarray(a)[None],device=self.device) for a in arrays]
        with torch.inference_mode():
            score=float(self.model(*inputs).sigmoid().item())
        return {"confidence":score,"decision":"YES" if score >= threshold else "NO"}
