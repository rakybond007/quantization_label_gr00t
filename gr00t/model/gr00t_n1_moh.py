# SPDX-License-Identifier: Apache-2.0
"""GR00T_N1_5_MoH: GR00T N1.5 wrapper with the Mixture-of-Horizons action head.

This file is intentionally a thin wrapper that mirrors the base `gr00t_n1.py`
structure. The only differences are:
  - the action head is `FlowmatchingActionHeadMoH` (file-isolated, no shared
    auxiliary-loss code with the sibling base head),
  - `model_type` is `"gr00t_n1_5_moh"` so HF AutoConfig/AutoModel can roundtrip,
  - `from_pretrained` accepts an `gr00t_n1_5` checkpoint and transfers
    the backbone (and, by default, the action-head weights that exist in the
    base head — DiT body, state encoder, action encoder/decoder, vlln, vl
    self-attention, position embedding). Only the MoH gate modules start from
    scratch.
  - `validate_data` relaxes the horizon assertion when dynamic replanning is
    on (the head may return fewer than `action_horizon` steps).
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head_moh import (
    FlowmatchingActionHeadMoH,
    FlowmatchingActionHeadMoHConfig,
)
from .backbone import EagleBackbone, PaligemmaBackbone, Qwen2_5VLBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class GR00T_N1_5_MoH_Config(PretrainedConfig):
    model_type = "gr00t_n1_5_moh"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})
    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})
    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})
    backbone_model_type: str = field(default="eagle", metadata={
        "help": "Type of the backbone model: 'eagle' | 'qwen2_5_vl' | 'qwen2_5_vl_7b' | 'paligemma'."
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class GR00T_N1_5_MoH(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_MoH_Config

    def __init__(self, config: GR00T_N1_5_MoH_Config, local_model_path: str = None):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.local_model_path = local_model_path
        backbone_model_type = getattr(config, "backbone_model_type", "eagle")
        print(f"[DEBUG][MoH]: Backbone model type: {backbone_model_type}")

        if backbone_model_type == "eagle":
            self.backbone = EagleBackbone(**config.backbone_cfg)
        elif backbone_model_type in ("qwen2_5_vl", "qwen2_5_vl_7b"):
            self.backbone = Qwen2_5VLBackbone(**config.backbone_cfg)
        elif backbone_model_type == "paligemma":
            self.backbone = PaligemmaBackbone(**config.backbone_cfg)
        else:
            raise ValueError(
                f"Unsupported backbone model type: {backbone_model_type}. "
                "Supported types are 'eagle', 'qwen2_5_vl', 'qwen2_5_vl_7b', and 'paligemma'."
            )

        action_head_cfg = FlowmatchingActionHeadMoHConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHeadMoH(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.backbone_model_type = backbone_model_type
        if self.backbone_model_type in ("qwen2_5_vl", "qwen2_5_vl_7b"):
            self.backbone_path = config.backbone_cfg.get("qwen_path", None)
        elif self.backbone_model_type == "paligemma":
            self.backbone_path = config.backbone_cfg.get("paligemma_path", None)
        else:
            self.backbone_path = None

    # ------------------------------------------------------------------
    # Input validation (mirrors base GR00T_N1_5)
    # ------------------------------------------------------------------
    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )
        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        if not isinstance(action_head_outputs, BatchFeature):
            raise ValueError(f"{ERROR_MSG}\nactino_head_outputs not BatchFeature")

        if is_training:
            if LOSS_KEY not in action_head_outputs:
                raise ValueError(f"{ERROR_MSG}\n'{LOSS_KEY}' missing during training")
            return

        # Inference: action_pred must be present.
        if ACTION_KEY not in action_head_outputs:
            raise ValueError(f"{ERROR_MSG}\n'{ACTION_KEY}' missing during inference")
        pred = action_head_outputs[ACTION_KEY]
        if pred.shape[2] != self.action_dim:
            raise ValueError(
                f"{ERROR_MSG}\n{pred.shape=} action_dim mismatch ({self.action_dim=})"
            )

        # Dynamic replanning may return < action_horizon steps; allow it.
        dyn = bool(getattr(self.action_head.config, "use_dynamic_replanning", False))
        if dyn:
            if not (1 <= pred.shape[1] <= self.action_horizon):
                raise ValueError(
                    f"{ERROR_MSG}\n[dyn] {pred.shape=} expected 1..{self.action_horizon}"
                )
        else:
            if pred.shape[1] != self.action_horizon:
                raise ValueError(
                    f"{ERROR_MSG}\n{pred.shape=} expected {self.action_horizon}"
                )

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------
    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(self, inputs: dict, head: str = "main") -> BatchFeature:
        """Inference. `head` is forwarded to the action head; MoH only supports
        head='main' but we keep the signature for API parity."""
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(
            backbone_outputs, action_inputs, head=head
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    # ------------------------------------------------------------------
    # Pretrained loading
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        load_action_head: bool = True,
        action_head_kwargs: dict = None,
        **kwargs,
    ):
        """Load an NVIDIA `gr00t_n1_5` (or already-MoH `gr00t_n1_5_moh`)
        checkpoint into a `GR00T_N1_5_MoH` model.

        Workflow:
          1. Download / locate the checkpoint.
          2. Load its config dict via `AutoConfig`.
          3. Force `model_type` -> `gr00t_n1_5_moh` and overlay any
             MoH-specific action-head kwargs (horizons, aux_weight, ...).
          4. Build a fresh `GR00T_N1_5_MoH` from that config.
          5. Transfer backbone weights from the checkpoint (always).
          6. If `load_action_head` is True (default), transfer the action-head
             weights that exist in the base head -- DiT body, encoders,
             decoder, vlln, vl self-attention, positional embedding. The MoH
             gate modules (`gate_out_proj`, `gate_noise_layer`) keep their
             from-scratch zero init.
        """
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"[MoH] Loading checkpoint from {pretrained_model_name_or_path}")
        print(f"[MoH] Tune backbone vision tower: {tune_visual}")
        print(f"[MoH] Tune backbone LLM: {tune_llm}")
        print(f"[MoH] Tune action head projector: {tune_projector}")
        print(f"[MoH] Tune action head DiT: {tune_diffusion_model}")
        print(f"[MoH] load_action_head={load_action_head}")

        # 1. Resolve to a local path
        try:
            local_model_path = snapshot_download(
                pretrained_model_name_or_path, repo_type="model"
            )
        except (HFValidationError, RepositoryNotFoundError):
            print(f"[MoH] Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path

        # 2. Read the checkpoint's config (NVIDIA: gr00t_n1_5; ours: gr00t_n1_5_moh)
        try:
            src_config = AutoConfig.from_pretrained(local_model_path)
            src_dict = src_config.to_dict()
        except Exception:
            import json
            cfg_path = os.path.join(local_model_path, "config.json")
            with open(cfg_path, "r") as f:
                src_dict = json.load(f)

        # 3. Swap model_type and overlay MoH-specific action_head_cfg fields
        src_dict.pop("model_type", None)
        src_dict.pop("architectures", None)
        if action_head_kwargs:
            ah_cfg = dict(src_dict.get("action_head_cfg", {}))
            ah_cfg.update(action_head_kwargs)
            src_dict["action_head_cfg"] = ah_cfg

        moh_config = GR00T_N1_5_MoH_Config(**src_dict)

        # 4. Build a fresh MoH model
        model = cls(moh_config, local_model_path=local_model_path)

        # 5/6. Transfer weights from the checkpoint's state_dict
        src_sd = _load_checkpoint_state_dict(local_model_path)

        # Backbone weights (always transfer; strict=False so we tolerate minor
        # naming drift between backbone families)
        backbone_sd = {
            k[len("backbone."):]: v
            for k, v in src_sd.items()
            if k.startswith("backbone.")
        }
        if backbone_sd:
            missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
            print(f"[MoH] Backbone state_dict loaded. "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[MoH] No backbone weights found in checkpoint (init from scratch).")

        # Action-head weights (optional but default True)
        if load_action_head:
            head_sd = {
                k[len("action_head."):]: v
                for k, v in src_sd.items()
                if k.startswith("action_head.")
            }
            if head_sd:
                # Filter to the keys that exist in our MoH head. The MoH gate
                # modules will simply not appear in head_sd and remain at their
                # zero-init values.
                moh_keys = set(model.action_head.state_dict().keys())
                filtered = {k: v for k, v in head_sd.items() if k in moh_keys}
                dropped = [k for k in head_sd.keys() if k not in moh_keys]
                missing, _ = model.action_head.load_state_dict(filtered, strict=False)
                # `missing` here will contain the MoH-only modules (gate_*).
                print(f"[MoH] Action-head state_dict loaded. "
                      f"transferred={len(filtered)} "
                      f"dropped(from-ckpt-not-in-MoH)={len(dropped)} "
                      f"missing(MoH-from-scratch)={len(missing)}")
                if dropped:
                    print(f"[MoH] Dropped (sample): {dropped[:5]}")
                if missing:
                    print(f"[MoH] From-scratch (sample): {list(missing)[:5]}")
            else:
                print("[MoH] No action_head weights found in checkpoint.")
        else:
            print("[MoH] Initializing action head entirely from scratch.")

        # Apply train/freeze
        model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return model

    @classmethod
    def from_config(cls, config, **kwargs):
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        print(f"[MoH] Building from config (no pretrained weights).")
        model = super()._from_config(config, **kwargs)
        model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return model


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_checkpoint_state_dict(local_model_path: str) -> dict:
    """Load a state_dict from a checkpoint directory.

    Supports:
      - sharded safetensors (model.safetensors.index.json + multiple shards)
      - single safetensors file (model.safetensors)
      - pytorch_model.bin / pytorch_model.bin.index.json
    """
    import json
    sd: dict = {}

    # Sharded safetensors
    idx_st = os.path.join(local_model_path, "model.safetensors.index.json")
    if os.path.exists(idx_st):
        from safetensors.torch import load_file
        with open(idx_st, "r") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        for shard in shards:
            sd.update(load_file(os.path.join(local_model_path, shard), device="cpu"))
        return sd

    # Single safetensors
    single_st = os.path.join(local_model_path, "model.safetensors")
    if os.path.exists(single_st):
        from safetensors.torch import load_file
        sd.update(load_file(single_st, device="cpu"))
        return sd

    # Sharded pytorch bin
    idx_pt = os.path.join(local_model_path, "pytorch_model.bin.index.json")
    if os.path.exists(idx_pt):
        with open(idx_pt, "r") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        for shard in shards:
            sd.update(torch.load(os.path.join(local_model_path, shard), map_location="cpu"))
        return sd

    # Single pytorch bin
    single_pt = os.path.join(local_model_path, "pytorch_model.bin")
    if os.path.exists(single_pt):
        sd.update(torch.load(single_pt, map_location="cpu"))
        return sd

    raise FileNotFoundError(
        f"No model weights found in {local_model_path} "
        "(expected model.safetensors[.index.json] or pytorch_model.bin[.index.json])."
    )


# Register so AutoConfig / AutoModel can resolve "gr00t_n1_5_moh".
AutoConfig.register("gr00t_n1_5_moh", GR00T_N1_5_MoH_Config)
AutoModel.register(GR00T_N1_5_MoH_Config, GR00T_N1_5_MoH)
