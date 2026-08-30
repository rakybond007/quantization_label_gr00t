# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "bridge"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = None
    # "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    use_multi_horizon_loss: bool = False
    """Enable multi-horizon auxiliary losses (compressed action targets).
    Pair this with --data-config single_panda_gripper_multi_horizon (or any
    config that fetches enough future timesteps)."""

    multi_horizon_factors: tuple[int, ...] = (2, 4)
    """Compression factors for auxiliary losses (32->16 = factor 2, 64->16 = factor 4)."""

    multi_horizon_loss_weights: tuple[float, ...] = (1.0, 1.0)
    """Per-factor loss weights, must match length of multi_horizon_factors."""

    multi_horizon_main_weight: float = 1.0
    """Weight on the original (factor=1) loss term."""

    aux_grad_scale_to_body: float = 0.1
    """Gradient scale applied to the gradient flowing FROM aux losses INTO the
    shared DiT body. 1.0 = no protection (full gradient), 0.0 = aux heads
    can only learn on top of frozen DiT representations. Typical: 0.1 ~ 0.5.
    Aux decoders themselves always receive full gradient."""

    aux_loss_warmup_steps: int = 5000
    """Linearly ramp aux loss WEIGHT from 0 to its configured value over this
    many forward steps. 0 disables warmup."""

    discrete_action_dims: tuple[int, ...] = ()
    """Action-dim indices to treat as DISCRETE during multi-horizon compression.
    For these dims, _compress_actions uses last-of-group (not sum). Use [] for
    none (legacy behavior). For single_panda_gripper data config:
    --discrete-action-dims 6 11  (gripper_close, control_mode)."""

    override_action_dim: int | None = None
    """If set, after loading the pretrained model, resize the action head to
    this new action_dim by reinitialising the action-dim-sensitive submodules
    (action encoder, output decoders). Used for dual-arm datasets whose action
    dim exceeds the base model's pretrained action_dim (e.g. dexjoco bimanual
    44 vs. base 32). Pretrained backbone + DiT body weights are preserved."""

    use_merged_8_head: bool = False
    """Enable the native 8-step 'merged_8' decoder head.
    Trained as a CO-MAIN (full grad to body, full weight, no warmup).
    Target = _compress_actions(actions_16, factor=2) -> (B, 8, D).
    At inference, use head='m8' to get a raw 8-step chunk."""

    merged_8_weight: float = 1.0
    """Loss weight for the merged_8 head (applied like main)."""

    use_ensemble_consistency_loss: bool = False
    """Enable shared-timestep cross-head velocity consistency loss.
    Requires --use-merged-8-head and --use-multi-horizon-loss.
    See docs/ENSEMBLE_DESIGN.md."""

    ensemble_consistency_weight: float = 0.1
    """Weight of the consistency loss after warmup."""

    ensemble_consistency_warmup_steps: int = 2000
    """Linear warmup for the consistency loss weight (0 -> configured)."""

    # === [Method 1] m8 distillation from f2 ===
    m8_distill_from_f2_weight: float = 0.0
    """Strong distillation: MSE(v_m8, v_f2[:8].detach()) * weight added to total loss.
    Used to fine-tune m8 head to match aux f2 quality. Requires use_merged_8_head and
    use_multi_horizon_loss. 0 = off."""

    # === [Method 2] m8 refinement decoder (post-hoc residual) ===
    use_m8_refinement: bool = False
    """Add a small residual MLP on top of m8 head output (action-space refinement).
    Backward compatible: head='m8' still works untouched; new head='m8_refined'
    applies the residual."""

    m8_refinement_weight: float = 1.0
    """Loss weight for the refinement MLP supervised loss."""

    m8_refinement_hidden: int = 512
    """Hidden width of the refinement MLP."""

    # === Additional decoder heads (for 4-expert MoE) ===
    use_merged_4_head: bool = False
    """Enable the m4 head (4-step output, target = sum-pair(GT[:8])).
    Used as one of the 4 MoE experts."""

    merged_4_weight: float = 1.0
    """Loss weight for the m4 head."""

    use_native_8_head: bool = False
    """Enable the n8 head (raw 8-step, target = GT[:8]).
    Used as one of the 4 MoE experts."""

    native_8_weight: float = 1.0
    """Loss weight for the n8 head."""

    # === MoE routing (4 experts: main + m8 + m4 + n8) ===
    use_moe_routing: bool = False
    """Enable MoE routing across 4 experts (requires --use-merged-8-head,
    --use-merged-4-head, --use-native-8-head). Soft mixture training,
    inference picks 1 expert via router."""

    moe_body_mode: str = "per_expert_h"
    """'shared_h16' (Option A: 1 body forward at h=16, cross-attn ExpertHeads,
    inference uses x_t reconstruction trick) or 'per_expert_h' (Option B:
    body forward at each expert's source horizon, inference is direct)."""

    moe_num_experts: int = 4
    """Number of experts (currently fixed at 4)."""

    moe_expert_n_layers: int = 2
    """Cross-attn layers per ExpertHead (only for shared_h16 mode)."""

    moe_router_temp: float = 0.5
    """Softmax temperature for router probs during training."""

    moe_target_temp: float = 0.3
    """Softmax temperature for the per-sample-loss-based router target (KL supervise)."""

    moe_balance_weight: float = 0.05
    """Weight on load-balance regularizer (push mean prob to uniform)."""

    moe_supervise_weight: float = 0.1
    """Weight on KL(router || softmax(-losses / τ_target))."""

    moe_compression_weight: float = 0.0
    """Compression-aware router prior. Negative bonus on Σ p_i · norm(1/horizon_i),
    pushing the router toward shorter (more compressed) experts. Default 0.
    Try 0.05~0.2."""

    moe_length_cost_weight: float = 0.0
    """Joint quality+length objective (option 1). Adds γ·horizon/H_max to each
    expert's per-sample loss BEFORE soft mix and supervise KL target — sample-
    specific quality-cost trade-off. Try 0.005~0.05."""

    moe_critic_weight: float = 0.0
    """Learnable compression-safety critic (option 2). Adds an MSE regression
    head that predicts per-expert losses; router KL target then uses the
    critic's per-sample prediction (chunk-specific quality prior). Try 0.1~1.0."""

    moe_critic_hidden: int = 256
    """Hidden dim of the compression-safety critic MLP."""

    moe_target_normalize: bool = False
    """Per-expert running-mean/var EMA z-score normalization of router supervise
    target. Removes systematic absolute-loss bias across experts so target signal
    becomes 'how unusually well/poorly did this expert do on THIS sample' (σ units).
    γ length cost becomes interpretable in σ units (try 0.2~0.5)."""

    moe_target_normalize_ema: float = 0.01
    """EMA decay for per-expert running stats (smaller = slower update, more smoothing)."""

    moe_routing_mode: str = "single_pick"
    """single_pick (legacy K-class softmax over experts) or
    per_quad_mask (4 quads × 3-class softmax over {main 4-raw, m8 2-comp, m4 1-deep})."""

    moe_per_quad_use_n8: bool = False
    """When per_quad_mask + use_native_8_head: add n8 as a 4th option (n8 4-raw)
    for quads 0,1 only (n8 covers env-steps 0..7). Quads 2,3 keep 3 options.
    Backward compatible: False reproduces original 4q×3o behavior."""

    moe_router_warmup_steps: int = 5000
    """Linear warmup over these many forward steps for balance + supervise weights."""

    moe_router_hidden: int = 256
    """Hidden width of the router MLP."""

    moe_inference_temp: float = 0.7
    """Softmax temperature at inference (for stochastic sampling)."""

    moe_inference_stochastic: bool = False
    """If True, sample expert by router probs at inference; else argmax."""

    moe_min_prob: float = 0.05
    """Anti-collapse: floor each expert's training prob (then renormalize).
    0.05 means every expert gets ≥5% gradient → prevents dead-expert collapse."""

    moe_uniform_warmup_steps: int = 0
    """Force uniform router probs for the first N forward steps. 0 disables.
    Use a few thousand steps to let all experts learn from real GT before the
    router commits."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""
    
    run_name: str = None

    load_action_head: bool = True
    """Whether to load the action head. If False, the action head will be randomly initialized"""

    backbone_model_type: Literal["eagle", "qwen2_5_vl", "paligemma", "qwen2_5_vl_7b"] = "eagle"

    backbone_select_layer: int = 12
    """Select layer for the backbone model. -1 means use all layers, 0 means use the first layer, 1 means use the second layer, etc."""

    backbone_path: str = None
    """Path to the backbone model. If None, the default backbone will be used."""

#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    if config.backbone_model_type != "eagle":
        transforms = data_config_cls.transform(config.backbone_model_type, backbone_path=config.backbone_path)
    else:
        transforms = data_config_cls.transform(config.backbone_model_type)

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    
    # ------------ step 2: load model ------------
    if config.base_model_path is not None:
        model = GR00T_N1_5.from_pretrained(
            pretrained_model_name_or_path=config.base_model_path,
            tune_llm=config.tune_llm,  # backbone's LLM
            tune_visual=config.tune_visual,  # backbone's vision tower
            tune_projector=config.tune_projector,  # action head's projector
            tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
            load_action_head=config.load_action_head,  # whether to load the action head
        )
    else:
        model_config = GR00T_N1_5_Config.from_pretrained(
            pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B"
        )
        model_config.tune_llm = config.tune_llm
        model_config.tune_visual = config.tune_visual
        model_config.tune_projector = config.tune_projector
        model_config.tune_diffusion_model = config.tune_diffusion_model
        # If no base model is provided, we initialize a new model
        if config.backbone_model_type == "eagle":
            print("Using Eagle backbone for GR00T-N1.5")
            # print(f"{model_config=}")
            model_config.backbone_model_type = config.backbone_model_type
            model_config.backbone_cfg["select_layer"] = config.backbone_select_layer

            if config.backbone_path is not None:
                model_config.backbone_cfg["eagle_path"] = config.backbone_path
        elif config.backbone_model_type == "qwen2_5_vl":
            print("Using Qwen2.5 backbone for GR00T-N1.5")
            # print(f"{model_config=}")
            model_config.backbone_model_type = config.backbone_model_type
            del model_config.backbone_cfg["eagle_path"]  # remove eagle_path, we will use Qwen2.5 backbone instead
            model_config.backbone_cfg["select_layer"] = config.backbone_select_layer

            if config.backbone_path is not None:
                model_config.backbone_cfg["qwen_path"] = config.backbone_path
        elif config.backbone_model_type == "paligemma":
            print("Using Paligemma backbone for GR00T-N1.5")
            # print(f"{model_config=}")
            model_config.backbone_model_type = config.backbone_model_type
            del model_config.backbone_cfg["eagle_path"]  # remove eagle_path, we will use Paligemma backbone instead
            model_config.backbone_cfg["select_layer"] = config.backbone_select_layer

            if config.backbone_path is not None:
                model_config.backbone_cfg["paligemma_path"] = config.backbone_path
        elif config.backbone_model_type == "qwen2_5_vl_7b":
            print("Using Qwen2.5 7B backbone for GR00T-N1.5")
            model_config.backbone_model_type = config.backbone_model_type
            del model_config.backbone_cfg["eagle_path"]
            model_config.backbone_cfg["select_layer"] = config.backbone_select_layer
            if config.backbone_path is not None:
                model_config.backbone_cfg["qwen_path"] = config.backbone_path
            else:
                model_config.backbone_cfg["qwen_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"

            model_config.action_head_cfg["diffusion_model_cfg"]['cross_attention_dim'] = 3584
            model_config.action_head_cfg["backbone_embedding_dim"] = 3584
            model_config.action_head_cfg["vl_self_attention_cfg"]["num_attention_heads"] = 56
            model_config.action_head_cfg["vl_self_attention_cfg"]["attention_head_dim"] = 64
        else:
            raise ValueError(
                f"Unsupported backbone model type: {config.backbone_model_type}. "
                "Supported types are: eagle, qwen2_5_vl, paligemma, qwen2_5_vl_7b"
            )
            
        model = GR00T_N1_5.from_config(
            config=model_config,
            tune_llm=config.tune_llm,  # backbone's LLM
            tune_visual=config.tune_visual,  # backbone's vision tower
            tune_projector=config.tune_projector,  # action head's projector
            tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
        )

    # ------------ optional: override action_dim (dual-arm) ------------
    if config.override_action_dim is not None:
        from gr00t.utils.action_dim_override import override_action_dim
        override_action_dim(model, int(config.override_action_dim))

    # ------------ multi-horizon loss: attach auxiliary decoders if enabled ------------
    if config.use_multi_horizon_loss:
        from gr00t.model.action_head.flow_matching_action_head import (
            CategorySpecificMLP,
        )
        import torch.nn as nn

        ah = model.action_head
        ah.use_multi_horizon_loss = True
        ah.multi_horizon_factors = list(config.multi_horizon_factors)
        ah.multi_horizon_loss_weights = list(config.multi_horizon_loss_weights)
        ah.multi_horizon_main_weight = float(config.multi_horizon_main_weight)
        ah.aux_grad_scale_to_body = float(config.aux_grad_scale_to_body)
        ah.aux_loss_warmup_steps = int(config.aux_loss_warmup_steps)
        ah.discrete_action_dims = list(config.discrete_action_dims)
        ah.use_merged_8_head = bool(config.use_merged_8_head)
        ah.merged_8_weight = float(config.merged_8_weight)
        if ah.use_merged_8_head and not hasattr(ah, "m8_action_decoder"):
            ah.m8_action_decoder = CategorySpecificMLP(
                num_categories=ah.config.max_num_embodiments,
                input_dim=ah.hidden_size,
                hidden_dim=ah.hidden_size,
                output_dim=ah.action_dim,
            )
        ah.use_ensemble_consistency_loss = bool(config.use_ensemble_consistency_loss)
        ah.ensemble_consistency_weight = float(config.ensemble_consistency_weight)
        ah.ensemble_consistency_warmup_steps = int(config.ensemble_consistency_warmup_steps)
        ah.aux_action_decoders = nn.ModuleDict({
            f"f{f}": CategorySpecificMLP(
                num_categories=ah.config.max_num_embodiments,
                input_dim=ah.hidden_size,
                hidden_dim=ah.hidden_size,
                output_dim=ah.action_dim,
            )
            for f in ah.multi_horizon_factors
        })
        # Initialize the forward step counter as a buffer (also a buffer in __init__,
        # but recreate here since we attached aux modules late)
        if not hasattr(ah, "_aux_forward_step"):
            ah.register_buffer("_aux_forward_step", torch.zeros((), dtype=torch.long), persistent=True)
        # Move new modules to the same device/dtype as the rest of the model
        ah.aux_action_decoders.to(device=model.device, dtype=ah.dtype)
        if ah.use_merged_8_head:
            ah.m8_action_decoder.to(device=model.device, dtype=ah.dtype)
        # Persist on config so subsequent loads recreate the heads
        ah.config.use_multi_horizon_loss = True
        ah.config.multi_horizon_factors = ah.multi_horizon_factors
        ah.config.multi_horizon_loss_weights = ah.multi_horizon_loss_weights
        ah.config.multi_horizon_main_weight = ah.multi_horizon_main_weight
        ah.config.aux_grad_scale_to_body = ah.aux_grad_scale_to_body
        ah.config.aux_loss_warmup_steps = ah.aux_loss_warmup_steps
        ah.config.discrete_action_dims = ah.discrete_action_dims
        ah.config.use_merged_8_head = ah.use_merged_8_head
        ah.config.merged_8_weight = ah.merged_8_weight
        ah.config.use_ensemble_consistency_loss = ah.use_ensemble_consistency_loss
        ah.config.ensemble_consistency_weight = ah.ensemble_consistency_weight
        ah.config.ensemble_consistency_warmup_steps = ah.ensemble_consistency_warmup_steps
        # Mirror onto top-level model config (so checkpoint will save it)
        model.config.action_head_cfg["use_multi_horizon_loss"] = True
        model.config.action_head_cfg["multi_horizon_factors"] = ah.multi_horizon_factors
        model.config.action_head_cfg["multi_horizon_loss_weights"] = ah.multi_horizon_loss_weights
        model.config.action_head_cfg["multi_horizon_main_weight"] = ah.multi_horizon_main_weight
        model.config.action_head_cfg["aux_grad_scale_to_body"] = ah.aux_grad_scale_to_body
        model.config.action_head_cfg["aux_loss_warmup_steps"] = ah.aux_loss_warmup_steps
        model.config.action_head_cfg["discrete_action_dims"] = ah.discrete_action_dims
        model.config.action_head_cfg["use_merged_8_head"] = ah.use_merged_8_head
        model.config.action_head_cfg["merged_8_weight"] = ah.merged_8_weight
        model.config.action_head_cfg["use_ensemble_consistency_loss"] = ah.use_ensemble_consistency_loss
        model.config.action_head_cfg["ensemble_consistency_weight"] = ah.ensemble_consistency_weight
        model.config.action_head_cfg["ensemble_consistency_warmup_steps"] = ah.ensemble_consistency_warmup_steps
        n_extra = sum(p.numel() for p in ah.aux_action_decoders.parameters())
        if ah.use_merged_8_head:
            n_extra += sum(p.numel() for p in ah.m8_action_decoder.parameters())
        print(f"[Multi-Horizon] Added {len(ah.multi_horizon_factors)} aux decoder heads "
              f"({n_extra:,} params): factors={ah.multi_horizon_factors}, "
              f"weights={ah.multi_horizon_loss_weights}")
        print(f"[Multi-Horizon] Protections: aux_grad_scale_to_body={ah.aux_grad_scale_to_body}, "
              f"aux_loss_warmup_steps={ah.aux_loss_warmup_steps}")
        if ah.use_merged_8_head:
            print(f"[Multi-Horizon] Added merged_8 head (native 8-step), weight={ah.merged_8_weight}")
        if ah.use_ensemble_consistency_loss:
            print(f"[Multi-Horizon] Ensemble-consistency loss ON: weight={ah.ensemble_consistency_weight}, "
                  f"warmup_steps={ah.ensemble_consistency_warmup_steps}")

        # === [Method 1] m8 distillation from f2 ===
        ah.m8_distill_from_f2_weight = float(getattr(config, "m8_distill_from_f2_weight", 0.0))
        ah.config.m8_distill_from_f2_weight = ah.m8_distill_from_f2_weight
        model.config.action_head_cfg["m8_distill_from_f2_weight"] = ah.m8_distill_from_f2_weight
        if ah.m8_distill_from_f2_weight > 0:
            print(f"[Method 1] m8 distillation from f2 ON: weight={ah.m8_distill_from_f2_weight}")

        # === [Method 2] m8 refinement decoder ===
        ah.use_m8_refinement = bool(getattr(config, "use_m8_refinement", False))
        ah.m8_refinement_weight = float(getattr(config, "m8_refinement_weight", 1.0))
        if ah.use_m8_refinement:
            assert ah.use_merged_8_head, "use_m8_refinement requires use_merged_8_head=True"
            if not hasattr(ah, "m8_refinement"):
                refine_hidden = int(getattr(config, "m8_refinement_hidden", 512))
                refine_in = ah.config.backbone_embedding_dim + 8 * ah.action_dim
                ah.m8_refinement = nn.Sequential(
                    nn.Linear(refine_in, refine_hidden), nn.GELU(),
                    nn.Linear(refine_hidden, refine_hidden), nn.GELU(),
                    nn.Linear(refine_hidden, 8 * ah.action_dim),
                )
                nn.init.zeros_(ah.m8_refinement[-1].weight)
                nn.init.zeros_(ah.m8_refinement[-1].bias)
                ah.m8_refinement.to(device=model.device, dtype=ah.dtype)
        ah.config.use_m8_refinement = ah.use_m8_refinement
        ah.config.m8_refinement_weight = ah.m8_refinement_weight
        ah.config.m8_refinement_hidden = int(getattr(config, "m8_refinement_hidden", 512))
        model.config.action_head_cfg["use_m8_refinement"] = ah.use_m8_refinement
        model.config.action_head_cfg["m8_refinement_weight"] = ah.m8_refinement_weight
        model.config.action_head_cfg["m8_refinement_hidden"] = ah.config.m8_refinement_hidden
        if ah.use_m8_refinement:
            n_refine = sum(p.numel() for p in ah.m8_refinement.parameters())
            print(f"[Method 2] m8_refinement MLP ON: hidden={ah.config.m8_refinement_hidden}, "
                  f"params={n_refine:,}, weight={ah.m8_refinement_weight}")

    # ------------ Additional decoder heads + MoE routing ------------
    if config.use_merged_4_head or config.use_native_8_head or config.use_moe_routing:
        from gr00t.model.action_head.flow_matching_action_head import (
            CategorySpecificMLP, ExpertHead,
        )
        import torch.nn as nn

        ah = model.action_head
        # Carry over discrete dims (used by _compress_actions)
        if not hasattr(ah, "discrete_action_dims") or len(ah.discrete_action_dims) == 0:
            ah.discrete_action_dims = list(config.discrete_action_dims)

        # m8 head: ensure built (MoE assumes use_merged_8_head=True)
        ah.use_merged_8_head = bool(config.use_merged_8_head) or bool(config.use_moe_routing)
        ah.merged_8_weight = float(config.merged_8_weight)
        if ah.use_merged_8_head and not hasattr(ah, "m8_action_decoder"):
            ah.m8_action_decoder = CategorySpecificMLP(
                num_categories=ah.config.max_num_embodiments,
                input_dim=ah.hidden_size,
                hidden_dim=ah.hidden_size,
                output_dim=ah.action_dim,
            )
            ah.m8_action_decoder.to(device=model.device, dtype=ah.dtype)

        # m4 head
        ah.use_merged_4_head = bool(config.use_merged_4_head)
        ah.merged_4_weight = float(config.merged_4_weight)
        if ah.use_merged_4_head and not hasattr(ah, "m4_action_decoder"):
            ah.m4_action_decoder = CategorySpecificMLP(
                num_categories=ah.config.max_num_embodiments,
                input_dim=ah.hidden_size,
                hidden_dim=ah.hidden_size,
                output_dim=ah.action_dim,
            )
            ah.m4_action_decoder.to(device=model.device, dtype=ah.dtype)

        # n8 head
        ah.use_native_8_head = bool(config.use_native_8_head)
        ah.native_8_weight = float(config.native_8_weight)
        if ah.use_native_8_head and not hasattr(ah, "n8_action_decoder"):
            ah.n8_action_decoder = CategorySpecificMLP(
                num_categories=ah.config.max_num_embodiments,
                input_dim=ah.hidden_size,
                hidden_dim=ah.hidden_size,
                output_dim=ah.action_dim,
            )
            ah.n8_action_decoder.to(device=model.device, dtype=ah.dtype)

        # MoE routing
        ah.use_moe_routing = bool(config.use_moe_routing)
        ah.moe_body_mode = str(config.moe_body_mode)
        ah.moe_num_experts = int(config.moe_num_experts)
        ah.moe_expert_n_layers = int(config.moe_expert_n_layers)
        ah.moe_router_temp = float(config.moe_router_temp)
        ah.moe_target_temp = float(config.moe_target_temp)
        ah.moe_balance_weight = float(config.moe_balance_weight)
        ah.moe_supervise_weight = float(config.moe_supervise_weight)
        ah.moe_compression_weight = float(config.moe_compression_weight)
        ah.moe_length_cost_weight = float(config.moe_length_cost_weight)
        ah.moe_critic_weight = float(config.moe_critic_weight)
        ah.moe_critic_hidden = int(config.moe_critic_hidden)
        ah.moe_target_normalize = bool(config.moe_target_normalize)
        ah.moe_target_normalize_ema = float(config.moe_target_normalize_ema)
        ah.moe_router_warmup_steps = int(config.moe_router_warmup_steps)
        ah.moe_router_hidden = int(config.moe_router_hidden)
        ah.moe_inference_temp = float(config.moe_inference_temp)
        ah.moe_inference_stochastic = bool(config.moe_inference_stochastic)

        if ah.use_moe_routing:
            assert ah.use_merged_8_head and ah.use_merged_4_head, \
                "MoE requires --use-merged-8-head and --use-merged-4-head"
            ah.moe_routing_mode = str(getattr(config, "moe_routing_mode", "single_pick"))
            ah.moe_per_quad_use_n8 = bool(getattr(config, "moe_per_quad_use_n8", False))
            assert ah.moe_routing_mode in ("single_pick", "per_quad_mask"), \
                f"Unknown moe_routing_mode: {ah.moe_routing_mode}"
            if ah.moe_routing_mode == "single_pick":
                assert ah.moe_num_experts in (3, 4), \
                    "Supported: --moe-num-experts=3 (main+m8+m4) or 4 (+n8)"
                if ah.moe_num_experts == 4:
                    assert ah.use_native_8_head, \
                        "--moe-num-experts=4 also requires --use-native-8-head"
                router_out = ah.moe_num_experts
            else:  # per_quad_mask: 4 quads × {3 or 4} options
                # Mirror FlowmatchingActionHead.__init__: when moe_per_quad_use_n8
                # AND use_native_8_head, add u8 (n8) as a 4th option for quads 0,1.
                ah._moe_num_quads = 4
                ah._per_quad_use_n8 = bool(ah.moe_per_quad_use_n8) and ah.use_native_8_head
                ah._moe_options_per_quad = 4 if ah._per_quad_use_n8 else 3
                router_out = ah._moe_num_quads * ah._moe_options_per_quad
                # Per-quad availability mask: (Q, C). u8 (c=3) only valid for quads 0,1.
                Q, C = ah._moe_num_quads, ah._moe_options_per_quad
                _pq_mask = torch.ones(Q, C, dtype=torch.bool)
                if ah._per_quad_use_n8:
                    _pq_mask[2:, 3] = False
                if not hasattr(ah, "_per_quad_option_mask"):
                    ah.register_buffer("_per_quad_option_mask",
                                       _pq_mask.to(device=model.device), persistent=False)
            if not hasattr(ah, "head_router"):
                router_in = ah.config.backbone_embedding_dim + ah.input_embedding_dim
                ah.head_router = nn.Sequential(
                    nn.Linear(router_in, ah.moe_router_hidden),
                    nn.GELU(),
                    nn.Linear(ah.moe_router_hidden, router_out),
                ).to(device=model.device, dtype=ah.dtype)
            if ah.moe_body_mode == "shared_h16" and not hasattr(ah, "moe_experts"):
                expert_specs = [
                    (ah.config.action_horizon, ah.config.action_horizon),
                    (ah.config.action_horizon // 2, ah.config.action_horizon),
                    (ah.config.action_horizon // 4, ah.config.action_horizon // 2),
                ]
                if ah.moe_num_experts == 4:
                    expert_specs.append(
                        (ah.config.action_horizon // 2, ah.config.action_horizon // 2)
                    )
                ah.moe_expert_specs = expert_specs
                ah.moe_experts = nn.ModuleList([
                    ExpertHead(
                        target_h=th,
                        body_dim=ah.hidden_size,
                        action_dim=ah.action_dim,
                        n_layers=ah.moe_expert_n_layers,
                    ) for (th, _) in expert_specs
                ]).to(device=model.device, dtype=ah.dtype)
            elif ah.moe_body_mode == "per_expert_h":
                ah.moe_expert_horizons = [
                    ah.config.action_horizon,
                    ah.config.action_horizon // 2,
                    ah.config.action_horizon // 4,
                ]
                if ah.moe_num_experts == 4:
                    ah.moe_expert_horizons.append(ah.config.action_horizon // 2)
            if not hasattr(ah, "_moe_forward_step"):
                ah.register_buffer("_moe_forward_step", torch.zeros((), dtype=torch.long), persistent=True)

        # Persist on action_head config + top-level model config so checkpoints save
        for attr in [
            "use_merged_8_head", "merged_8_weight",
            "use_merged_4_head", "merged_4_weight",
            "use_native_8_head", "native_8_weight",
            "use_moe_routing", "moe_body_mode", "moe_num_experts", "moe_expert_n_layers",
            "moe_router_temp", "moe_target_temp", "moe_balance_weight", "moe_supervise_weight",
            "moe_compression_weight", "moe_length_cost_weight",
            "moe_critic_weight", "moe_critic_hidden",
            "moe_target_normalize", "moe_target_normalize_ema",
            "moe_routing_mode", "moe_per_quad_use_n8",
            "moe_router_warmup_steps", "moe_router_hidden", "moe_inference_temp",
            "moe_inference_stochastic",
        ]:
            setattr(ah.config, attr, getattr(ah, attr))
            model.config.action_head_cfg[attr] = getattr(ah, attr)
        if ah.discrete_action_dims:
            ah.config.discrete_action_dims = list(ah.discrete_action_dims)
            model.config.action_head_cfg["discrete_action_dims"] = list(ah.discrete_action_dims)

        if ah.use_merged_4_head:
            n = sum(p.numel() for p in ah.m4_action_decoder.parameters())
            print(f"[MoE] m4 head added: weight={ah.merged_4_weight}, params={n:,}")
        if ah.use_native_8_head:
            n = sum(p.numel() for p in ah.n8_action_decoder.parameters())
            print(f"[MoE] n8 head added: weight={ah.native_8_weight}, params={n:,}")
        if ah.use_moe_routing:
            n_router = sum(p.numel() for p in ah.head_router.parameters())
            n_experts = (sum(p.numel() for p in ah.moe_experts.parameters())
                         if ah.moe_body_mode == "shared_h16" else 0)
            print(f"[MoE] routing ON: body_mode={ah.moe_body_mode}, num_experts={ah.moe_num_experts}")
            print(f"[MoE] router params: {n_router:,}, expert tower params: {n_experts:,}")
            print(f"[MoE] hyper: τ_router={ah.moe_router_temp}, τ_target={ah.moe_target_temp}, "
                  f"λ_balance={ah.moe_balance_weight}, λ_supervise={ah.moe_supervise_weight}, "
                  f"warmup={ah.moe_router_warmup_steps}")
            if ah.moe_length_cost_weight > 0 or ah.moe_critic_weight > 0:
                print(f"[MoE] length_cost_w={ah.moe_length_cost_weight}, "
                      f"critic_w={ah.moe_critic_weight}, critic_hidden={ah.moe_critic_hidden}")
                if ah.moe_critic_weight > 0 and hasattr(ah, "moe_critic_head"):
                    n_crit = sum(p.numel() for p in ah.moe_critic_head.parameters())
                    print(f"[MoE] critic head params: {n_crit:,}")
            if ah.moe_target_normalize:
                print(f"[MoE] target_normalize ON (per-expert EMA z-score, "
                      f"ema={ah.moe_target_normalize_ema}, γ in σ units)")

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=6,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        # MoE training adds a 1e-12 anti-unused-grad L2 on expert+decoder params so
        # DDP sees every param touched. find_unused_parameters can stay False.
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    if not os.path.exists(config.output_dir):
        config.resume = False

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # If the value is a list/tuple (e.g. dataset_path, multi_horizon_factors),
                    # add each element separately so tyro re-parses correctly downstream.
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
