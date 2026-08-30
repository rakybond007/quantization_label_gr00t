# Meta-query fork of gr00t_n1.py.  Uses EagleBackboneMetaQ (which appends N
# learnable meta-query tokens to the LM input and returns their output features
# alongside the regular VL features) and FlowmatchingActionHeadMetaQ (which
# pulls router input from meta_q_features instead of vl_embeds.mean()).  All
# other architectural pieces are identical to GR00T_N1_5.

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head_metaq import (
    FlowmatchingActionHeadMetaQ,
    FlowmatchingActionHeadMetaQConfig,
)
from .backbone.eagle_backbone_metaq import EagleBackboneMetaQ

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class GR00T_N1_5_MetaQ_Config(PretrainedConfig):
    """Same shape as GR00T_N1_5_Config but a distinct model_type so checkpoints
    can be loaded with the metaq backbone+head pair without interfering with
    the legacy gr00t_n1_5 path."""

    model_type = "gr00t_n1_5_metaq"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})
    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})
    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})
    backbone_model_type: str = field(default="eagle", metadata={
        "help": "Backbone type. Currently only 'eagle' is supported in the metaq fork."
    })
    n_meta_q: int = field(default=8, metadata={"help": "Number of meta-query tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class GR00T_N1_5_MetaQ(PreTrainedModel):
    """GR00T-N1.5 with meta-query routed router input.  Mirrors GR00T_N1_5 but
    instantiates EagleBackboneMetaQ + FlowmatchingActionHeadMetaQ.  The forward
    contract is identical (returns BatchFeature with loss/action_pred), so
    training and inference scripts can swap GR00T_N1_5 → GR00T_N1_5_MetaQ
    without other changes."""

    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_MetaQ_Config

    def __init__(self, config: GR00T_N1_5_MetaQ_Config, local_model_path: str = None):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.local_model_path = local_model_path
        backbone_model_type = getattr(config, "backbone_model_type", "eagle")
        if backbone_model_type != "eagle":
            raise NotImplementedError(
                f"GR00T_N1_5_MetaQ currently only supports the eagle backbone "
                f"(got '{backbone_model_type}').  Add a metaq fork of the chosen "
                f"backbone before training with it."
            )
        n_meta_q = int(getattr(config, "n_meta_q", 8))
        # Wire n_meta_q through both backbone (owns the embeds + injection) and
        # action head (knows the router uses meta_q_features) so they agree.
        self.backbone = EagleBackboneMetaQ(n_meta_q=n_meta_q, **config.backbone_cfg)
        ah_cfg_dict = dict(config.action_head_cfg)
        ah_cfg_dict["n_meta_q"] = n_meta_q
        action_head_cfg = FlowmatchingActionHeadMetaQConfig(**ah_cfg_dict)
        self.action_head = FlowmatchingActionHeadMetaQ(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype
        self.backbone_model_type = backbone_model_type
        self.backbone_path = None

    # ---- The validate_*, forward, get_action, prepare_input methods below
    # are copied verbatim from GR00T_N1_5.  Only the class identity and
    # underlying backbone/action_head differ.

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (len(action.shape) == 3
                        and action.shape[1] == self.action_horizon
                        and action.shape[2] == self.action_dim)
            if not type_ok:
                error_msg += f"\n{action.dtype=}"; detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"; detected_error = True
        if "action_extended" in inputs:
            ax = inputs["action_extended"]
            ax_ok = isinstance(ax, torch.Tensor) and ax.dim() == 3 and ax.shape[2] == self.action_dim
            if not ax_ok:
                error_msg += f"\n{ax.shape=}"; detected_error = True
        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:  error_msg += f"\n{type(video)=}"; detected_error = True
            if not dtype_ok: error_msg += f"\n{video.dtype=}"; detected_error = True
            if not shape_ok: error_msg += f"\n{video.shape=}"; detected_error = True
        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (not isinstance(backbone_outputs, BatchFeature)
                         or BACKBONE_FEATURE_KEY not in backbone_outputs)
        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            raise ValueError(error_msg)
        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (LOSS_KEY in action_head_outputs and is_training)
            or (ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim))
        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            raise ValueError(error_msg)

    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(self, inputs: dict, head: str = "main") -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs, head=head)
        if head in ("main", "ensemble", "ensemble_fix", "main_and_m8"):
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, load_action_head: bool = True, **kwargs):
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        print(f"Loading pretrained dual brain (metaq) from {pretrained_model_name_or_path}")
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            local_model_path = pretrained_model_name_or_path
        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )
        # Re-init meta_q_emb explicitly: when loading from a base ckpt that
        # doesn't have this parameter, HF's from_pretrained materializes it
        # from meta-device WITHOUT calling _init_weights for raw nn.Parameters
        # → memory contains garbage (e.g. |max|=1e17, NaN downstream).  Re-
        # initialize to a small normal distribution.
        if hasattr(pretrained_model.backbone, "meta_q_emb"):
            with torch.no_grad():
                pretrained_model.backbone.meta_q_emb.data.normal_(mean=0.0, std=0.02)
            print(f"[metaq] re-initialized meta_q_emb (shape="
                  f"{tuple(pretrained_model.backbone.meta_q_emb.shape)}, std=0.02)")
        if not load_action_head:
            print("[metaq] Initializing action head from scratch. Only loading backbone.")
        pretrained_model.backbone.set_trainable_parameters(
            tune_llm=tune_llm, tune_visual=tune_visual,
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model,
        )
        return pretrained_model
