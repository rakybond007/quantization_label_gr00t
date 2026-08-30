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

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


class GradScale(torch.autograd.Function):
    """Identity in forward, scales gradient by `scale` in backward.

    Used so that auxiliary losses can update their own decoder heads with full
    gradient, while the gradient flowing INTO the shared DiT body from those
    aux paths is dampened by `scale`. This protects pretrained DiT
    representations from being shifted too aggressively by newly added
    auxiliary tasks (cf. gradient reversal layer with positive scale).
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class _ExpertCABlock(nn.Module):
    """One cross-attn layer for ExpertHead: cross-attn(q->ctx) + self-attn(q) + FFN."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.ln_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x, ctx):
        h, _ = self.cross_attn(self.ln_ca(x), ctx, ctx, need_weights=False)
        x = x + h
        n = self.ln_sa(x)
        h, _ = self.self_attn(n, n, n, need_weights=False)
        x = x + h
        x = x + self.ffn(self.ln_ff(x))
        return x


class ExpertHead(nn.Module):
    """Maps body output (B, source_h, D) -> (B, target_h, action_dim) via
    target_h learned query tokens that cross-attend to body features.

    Used for the shared_h16 MoE body mode: body forward always at action_horizon
    (e.g. 16), and each expert has its own ExpertHead with its target_h queries
    + cross-attn layers + final action projection. Compression (e.g. 16→8) is
    fully learned (no fixed pool). For target_h == source_h experts, cross-attn
    still acts as a soft identity refinement.
    """

    def __init__(self, target_h: int, body_dim: int, action_dim: int,
                 n_layers: int = 2, num_heads: int = 8, init_query_std: float = 0.02):
        super().__init__()
        self.target_h = target_h
        self.query_tokens = nn.Parameter(torch.randn(target_h, body_dim) * init_query_std)
        self.layers = nn.ModuleList([_ExpertCABlock(body_dim, num_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(body_dim)
        self.proj = nn.Linear(body_dim, action_dim)

    def forward(self, body_out):  # body_out: (B, source_h, D)
        B = body_out.shape[0]
        x = self.query_tokens.unsqueeze(0).expand(B, -1, -1).to(body_out.dtype)
        for layer in self.layers:
            x = layer(x, body_out)
        return self.proj(self.norm(x))  # (B, target_h, action_dim)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadFairMoeV2Config(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)

    # === Multi-horizon (compressed) auxiliary loss ===
    # When enabled, the model has additional decoder heads that predict
    # 16-step actions where each step represents the SUM of multiple
    # adjacent original actions. For factor=2, model predicts 16 actions
    # where action_pred[i] ≈ sum of GT_actions[2i:2i+2]. This requires the
    # dataset to provide enough future timesteps (e.g. 64 for factor 4).
    # The model output structure (16-step) is preserved at inference; these
    # extra heads are training-only auxiliary losses by default.
    use_multi_horizon_loss: bool = field(default=False)
    multi_horizon_factors: list = field(default_factory=lambda: [2, 4],
        metadata={"help": "Compression factors for auxiliary losses (e.g. [2, 4] -> 32->16, 64->16)."})
    multi_horizon_loss_weights: list = field(default_factory=lambda: [1.0, 1.0],
        metadata={"help": "Per-factor weight for the auxiliary losses."})
    multi_horizon_main_weight: float = field(default=1.0,
        metadata={"help": "Weight on the original (factor=1) loss term."})

    # === Protective measures for the shared DiT body ===
    # Gradient flowing from aux losses INTO the shared DiT body is multiplied
    # by this scale. Aux decoders themselves still get full gradient.
    # 1.0 = no protection (gradient flows fully); 0.0 = aux heads only learn on
    # top of frozen DiT representations. Typical: 0.1 ~ 0.5.
    aux_grad_scale_to_body: float = field(default=0.1,
        metadata={"help": "Gradient scale for aux losses' gradient into the shared DiT body."})
    # Linear warmup of the aux loss WEIGHT over this many forward steps.
    # During warmup, weight = configured_weight * (step / warmup_steps).
    # 0 disables warmup (aux weights take effect immediately).
    aux_loss_warmup_steps: int = field(default=5000,
        metadata={"help": "Linearly ramp aux loss weight from 0 to configured value over this many forward steps."})

    # Indices of discrete action dimensions (e.g. gripper, control_mode).
    # When set:
    # - Training: _compress_actions uses LAST value of each group (instead of sum)
    #   for these dims. Sum makes no semantic sense for binary signals.
    # - Inference (ensemble_fix head): LS result is overwritten by main head's
    #   value for these dims, since LS averaging is meaningless for discrete signals.
    # Empty list = no special handling (legacy / pre-fix behavior).
    discrete_action_dims: list = field(default_factory=list)

    # === Merged-8 "second main" head (native 8-step decoder) ===
    # Predicts an 8-step action chunk where each step represents sum-of-2 env-step
    # deltas covering the same 16-env-step horizon as `action_decoder`. Unlike the
    # f2 aux head (which outputs 16 steps via compressed 32->16 target), this head
    # natively outputs horizon=8 and is treated as a CO-MAIN loss (full weight,
    # full gradient to the shared DiT body, no warmup). At inference, head="m8"
    # returns a raw 8-step chunk that the client can execute directly.
    use_merged_8_head: bool = field(default=False,
        metadata={"help": "Enable the native 8-step 'merged_8' decoder head."})
    merged_8_weight: float = field(default=1.0,
        metadata={"help": "Loss weight for the merged_8 head (applied like main)."})

    # === Ensemble-consistency loss (Section 3 of docs/ENSEMBLE_DESIGN.md) ===
    # Uses a SHARED timestep across main / f2 / m8 per forward so their predicted
    # velocities are directly comparable. Adds a cross-head MSE pulling
    # pair-summed main and m8 toward f2[:8] (which predicts the same quantity).
    # Requires use_merged_8_head=True and use_multi_horizon_loss=True.
    use_ensemble_consistency_loss: bool = field(default=False,
        metadata={"help": "Enable shared-t cross-head consistency loss on velocities."})
    ensemble_consistency_weight: float = field(default=0.1,
        metadata={"help": "Weight for the cross-head consistency loss (post-warmup)."})
    ensemble_consistency_warmup_steps: int = field(default=2000,
        metadata={"help": "Linear warmup of the consistency loss weight (0 -> configured)."})

    # === [Method 1] Distillation: m8 mimics aux f2 head's velocity ===
    # Adds MSE(v_m8, v_f2[:8].detach()) — m8 head learns to match the aux f2
    # decoder (already a target in consistency loss but unidirectional + stronger).
    # Useful for fine-tuning a trained model so m8-only inference catches up to
    # ensemble→8 quality.
    m8_distill_from_f2_weight: float = field(default=0.0,
        metadata={"help": "Weight for distillation loss MSE(v_m8, v_f2[:8].detach()). 0 = off."})

    # === [Method 2] m8 refinement decoder (post-hoc residual) ===
    # Tiny extra MLP module: refine(estimated_clean_action_m8, vl_pooled) -> delta.
    # At inference: final = m8_decode + refine(m8_decode, vl). At training: an
    # extra forward pass through m8_decoder with no_grad (or full grad) gets the
    # estimated clean action; the refine module learns to predict GT - estimate.
    use_m8_refinement: bool = field(default=False,
        metadata={"help": "Add a residual refinement module on top of m8 head output."})
    m8_refinement_weight: float = field(default=1.0,
        metadata={"help": "Loss weight for the refinement-output supervised loss."})
    m8_refinement_hidden: int = field(default=512,
        metadata={"help": "Hidden width of the refinement MLP (2 layers)."})

    # === Additional decoder heads for 4-expert MoE ===
    # m4: 4-step output, target = sum-pair(GT[:8]) (each output covers 2 sim steps,
    #     total 8 sim steps). Source horizon for body forward: 4 (per_expert_h) or 16 (shared_h16).
    # n8: 8-step output, target = GT[:8] (raw deltas, 8 sim steps).
    #     Source horizon: 8 (per_expert_h) or 16 (shared_h16).
    use_merged_4_head: bool = field(default=False,
        metadata={"help": "Enable the m4 head (4-step compressed from 8 GT actions)."})
    merged_4_weight: float = field(default=1.0,
        metadata={"help": "Loss weight for the m4 head."})
    use_native_8_head: bool = field(default=False,
        metadata={"help": "Enable the n8 head (raw 8-step from first 8 GT actions)."})
    native_8_weight: float = field(default=1.0,
        metadata={"help": "Loss weight for the n8 head."})

    # === MoE routing across 4 experts (main, m8, m4, n8) ===
    # When enabled, a small router on pooled state+VL features outputs probabilities
    # over the 4 experts. Training: all 4 expert losses combined as a soft mixture
    # weighted by router probs, plus load-balance regularization and a per-sample
    # KL supervision (router target = softmax(-loss / τ_target)).
    # Inference (head="moe"): router picks 1 expert (argmax or sampling); only that
    # expert is decoded.
    use_moe_routing: bool = field(default=False,
        metadata={"help": "Enable 4-expert MoE routing across main/m8/m4/n8."})
    moe_body_mode: str = field(default="per_expert_h",
        metadata={"help": "'shared_h16' (1 body forward at h=16, cross-attn experts) or "
                          "'per_expert_h' (body forward at each expert's source horizon)."})
    moe_num_experts: int = field(default=4,
        metadata={"help": "Number of experts (4 = main+m8+m4+n8)."})
    moe_expert_n_layers: int = field(default=2,
        metadata={"help": "Number of cross-attn layers per expert (only for shared_h16 mode)."})
    moe_router_temp: float = field(default=0.5,
        metadata={"help": "Softmax temperature for router probs during training."})
    moe_target_temp: float = field(default=0.3,
        metadata={"help": "Softmax temperature for the per-sample-loss-based router target."})
    moe_balance_weight: float = field(default=0.05,
        metadata={"help": "Weight on the load-balance regularizer (push mean prob to uniform)."})
    moe_supervise_weight: float = field(default=0.1,
        metadata={"help": "Weight on KL(router || softmax(-losses / τ_target))."})
    moe_compression_weight: float = field(default=0.0,
        metadata={"help": "Weight on compression-aware router prior (negative bonus on "
                          "Σ p_i · norm(1/horizon_i)). Pushes router toward shorter experts. "
                          "Default 0 (off). Try 0.05~0.2."})
    moe_length_cost_weight: float = field(default=0.0,
        metadata={"help": "Joint quality+length objective (option 1). Adds γ·horizon/H_max "
                          "to each expert's per-sample loss BEFORE soft mix and KL target. "
                          "Sample-specific quality-cost trade-off. Try 0.005~0.05."})
    moe_critic_weight: float = field(default=0.0,
        metadata={"help": "Learnable compression-safety critic (option 2). Adds an MSE "
                          "regression head that predicts per-expert losses; router KL "
                          "target uses critic's prediction (chunk-specific). Default 0 (off). "
                          "Try 0.1~1.0."})
    moe_critic_hidden: int = field(default=256,
        metadata={"help": "Hidden dim of the compression-safety critic MLP."})
    moe_target_normalize: bool = field(default=False,
        metadata={"help": "Per-expert running-mean/var EMA over losses_b4. When ON, the "
                          "router supervise target uses z-scored loss signal (bias-free across "
                          "experts) and γ length cost is in σ units. soft-mixture uses raw losses."})
    moe_target_normalize_ema: float = field(default=0.01,
        metadata={"help": "EMA decay for running mean/var (smaller = slower, more smoothing)."})
    moe_router_warmup_steps: int = field(default=5000,
        metadata={"help": "Linear warmup over these many forward steps for balance+supervise weights."})
    moe_router_hidden: int = field(default=256,
        metadata={"help": "Hidden width of the router MLP."})
    moe_routing_mode: str = field(default="single_pick",
        metadata={"help": "single_pick (legacy K-class softmax over experts) or "
                          "per_quad_mask (4 quad × 3-class softmax over {main 4-raw, m8 2-comp, m4 1-deep})."})
    moe_per_quad_use_n8: bool = field(default=False,
        metadata={"help": "When per_quad_mask: add n8 as 4th raw option for quads 0,1 (env-steps "
                          "0..7 covered by n8). Quads 2,3 keep 3 options (no n8 there). Backward "
                          "compatible: False reproduces original 4q×3o behavior."})
    moe_inference_temp: float = field(default=0.7,
        metadata={"help": "Softmax temperature at inference (used for stochastic sampling)."})
    moe_inference_stochastic: bool = field(default=False,
        metadata={"help": "If True, sample expert by router probs at inference; else argmax."})

    # Anti-collapse: clip per-expert prob to at least this value (then renormalize).
    # Ensures every expert always receives some gradient → prevents the dead-expert
    # problem where router commits to one expert and others drift / regress.
    moe_min_prob: float = field(default=0.05,
        metadata={"help": "Min prob per expert during training (0 = disabled)."})
    # Force uniform router probs for the first N forward steps so all experts learn
    # before the router differentiates. After this many steps, router output is used.
    moe_uniform_warmup_steps: int = field(default=0,
        metadata={"help": "Force uniform router probs for first N steps (0 = disabled)."})

    # ===== jimin-dev loss knobs (back-ported from adf55c4) =====
    # Defaults match jimin's emaKL.sh: only [B] is even available, [A] and [C]
    # remain off. Turn each on via CLI flag when needed.

    # [A] shared timestep across experts in per_expert_h MoE
    moe_shared_t: bool = field(default=False,
        metadata={"help": "[A] Share one timestep across all experts in per_expert_h MoE."})

    # [B] EMA-normalize per-expert losses before KL target softmax
    moe_normalize_loss_for_kl: bool = field(default=False,
        metadata={"help": "[B] EMA-normalize per-expert losses before KL target softmax."})
    moe_loss_ema_momentum: float = field(default=0.99,
        metadata={"help": "[B] EMA momentum for per-expert mean-loss tracker."})

    # [C] Balance loss type
    moe_balance_type: str = field(default="mean_sq",
        metadata={"help": "[C] 'mean_sq' (Σ (mean_p - 1/K)^2) or 'switch' (K · Σ f_i · p_i)."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHeadFairMoeV2(nn.Module):
    config_class = FlowmatchingActionHeadFairMoeV2Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadFairMoeV2Config,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # === Auxiliary decoder heads for multi-horizon loss ===
        # One extra head per compression factor, sharing input from the same DiT body.
        # Each head predicts a 16-step trajectory for a coarser granularity target.
        self.use_multi_horizon_loss = getattr(config, "use_multi_horizon_loss", False)
        if self.use_multi_horizon_loss:
            self.multi_horizon_factors = list(config.multi_horizon_factors)
            self.multi_horizon_loss_weights = list(config.multi_horizon_loss_weights)
            self.multi_horizon_main_weight = config.multi_horizon_main_weight
            assert len(self.multi_horizon_factors) == len(self.multi_horizon_loss_weights), (
                "multi_horizon_factors and multi_horizon_loss_weights must have same length"
            )
            self.aux_action_decoders = nn.ModuleDict({
                f"f{f}": CategorySpecificMLP(
                    num_categories=config.max_num_embodiments,
                    input_dim=self.hidden_size,
                    hidden_dim=self.hidden_size,
                    output_dim=self.action_dim,
                )
                for f in self.multi_horizon_factors
            })
        else:
            self.multi_horizon_factors = []
            self.multi_horizon_loss_weights = []
            self.multi_horizon_main_weight = 1.0
            self.aux_action_decoders = nn.ModuleDict()

        # Protective measures (gradient scaling + warmup) for aux losses
        self.aux_grad_scale_to_body = float(getattr(config, "aux_grad_scale_to_body", 0.1))
        self.aux_loss_warmup_steps = int(getattr(config, "aux_loss_warmup_steps", 5000))
        # Internal forward counter (auto-incremented). Saved as a buffer so it
        # persists across checkpoint save/load. Approximates the global step;
        # exact alignment isn't required for warmup.
        self.register_buffer(
            "_aux_forward_step", torch.zeros((), dtype=torch.long), persistent=True
        )

        # Discrete action dim indices (e.g. gripper, control_mode).
        # Mutable attribute (not buffer); can be overridden after load via
        # `head.discrete_action_dims = [6, 11]`.
        self.discrete_action_dims = list(getattr(config, "discrete_action_dims", []) or [])

        # Merged-8 head: native 8-step decoder (see config docstring).
        self.use_merged_8_head = bool(getattr(config, "use_merged_8_head", False))
        self.merged_8_weight = float(getattr(config, "merged_8_weight", 1.0))
        if self.use_merged_8_head:
            self.m8_action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )

        # Merged-4 head: 4-step compressed (sum-pair of GT[:8]).
        self.use_merged_4_head = bool(getattr(config, "use_merged_4_head", False))
        self.merged_4_weight = float(getattr(config, "merged_4_weight", 1.0))
        if self.use_merged_4_head:
            self.m4_action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )

        # Native-8 head: raw 8-step (= GT[:8], no compression).
        self.use_native_8_head = bool(getattr(config, "use_native_8_head", False))
        self.native_8_weight = float(getattr(config, "native_8_weight", 1.0))
        if self.use_native_8_head:
            self.n8_action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )

        # === MoE routing across 4 experts ===
        self.use_moe_routing = bool(getattr(config, "use_moe_routing", False))
        self.moe_body_mode = str(getattr(config, "moe_body_mode", "per_expert_h"))
        self.moe_router_temp = float(getattr(config, "moe_router_temp", 0.5))
        self.moe_target_temp = float(getattr(config, "moe_target_temp", 0.3))
        self.moe_balance_weight = float(getattr(config, "moe_balance_weight", 0.05))
        self.moe_supervise_weight = float(getattr(config, "moe_supervise_weight", 0.1))
        self.moe_compression_weight = float(getattr(config, "moe_compression_weight", 0.0))
        self.moe_length_cost_weight = float(getattr(config, "moe_length_cost_weight", 0.0))
        self.moe_critic_weight = float(getattr(config, "moe_critic_weight", 0.0))
        self.moe_critic_hidden = int(getattr(config, "moe_critic_hidden", 256))
        self.moe_target_normalize = bool(getattr(config, "moe_target_normalize", False))
        self.moe_target_normalize_ema = float(getattr(config, "moe_target_normalize_ema", 0.01))
        self.moe_router_warmup_steps = int(getattr(config, "moe_router_warmup_steps", 5000))
        self.moe_inference_temp = float(getattr(config, "moe_inference_temp", 0.7))
        self.moe_inference_stochastic = bool(getattr(config, "moe_inference_stochastic", False))
        self.moe_min_prob = float(getattr(config, "moe_min_prob", 0.05))
        self.moe_uniform_warmup_steps = int(getattr(config, "moe_uniform_warmup_steps", 0))
        # ===== jimin-dev knobs =====
        self.moe_shared_t = bool(getattr(config, "moe_shared_t", False))
        self.moe_normalize_loss_for_kl = bool(getattr(config, "moe_normalize_loss_for_kl", False))
        self.moe_loss_ema_momentum = float(getattr(config, "moe_loss_ema_momentum", 0.99))
        self.moe_balance_type = str(getattr(config, "moe_balance_type", "mean_sq"))
        self.moe_expert_n_layers = int(getattr(config, "moe_expert_n_layers", 2))
        self.moe_router_hidden = int(getattr(config, "moe_router_hidden", 256))
        self.moe_num_experts = int(getattr(config, "moe_num_experts", 4))
        self.moe_routing_mode = str(getattr(config, "moe_routing_mode", "single_pick"))
        if self.use_moe_routing:
            assert self.use_merged_8_head, "use_moe_routing requires use_merged_8_head=True"
            assert self.use_merged_4_head, "use_moe_routing requires use_merged_4_head=True"
            assert self.moe_routing_mode in ("single_pick", "per_quad_mask"), \
                f"Unknown moe_routing_mode: {self.moe_routing_mode}"
            if self.moe_routing_mode == "single_pick":
                assert self.moe_num_experts in (3, 4), \
                    "Supported: 3 (main+m8+m4) or 4 (+n8) experts"
                if self.moe_num_experts == 4:
                    assert self.use_native_8_head, \
                        "moe_num_experts=4 requires use_native_8_head=True (n8 expert)"
                router_out = self.moe_num_experts
            else:  # per_quad_mask: 4 quads × {3 or 4 options}
                # Option 0=main 4-raw, 1=m8 2-comp, 2=m4 1-deep. When
                # `moe_per_quad_use_n8` is True AND `use_native_8_head` is True,
                # add option 3=n8 4-raw (only valid for quads 0,1 since n8 only
                # covers env-steps 0..7). Quads 2,3 keep 3 options (n8 masked).
                self._moe_num_quads = 4
                self._per_quad_use_n8 = bool(getattr(config, "moe_per_quad_use_n8", False)) \
                                          and self.use_native_8_head
                self._moe_options_per_quad = 4 if self._per_quad_use_n8 else 3
                router_out = self._moe_num_quads * self._moe_options_per_quad
                # Per-quad availability mask: (Q, max_C). True = option valid for that quad.
                Q = self._moe_num_quads
                C = self._moe_options_per_quad
                mask = torch.ones(Q, C, dtype=torch.bool)
                if self._per_quad_use_n8:
                    # n8 covers env-steps 0..7 → only quads 0,1 (env-steps 0..3 and 4..7)
                    mask[2:, 3] = False
                self.register_buffer("_per_quad_option_mask", mask, persistent=False)
            # Router: pooled state + pooled vl -> router_out logits.
            router_in = config.backbone_embedding_dim + self.input_embedding_dim
            self.head_router = nn.Sequential(
                nn.Linear(router_in, self.moe_router_hidden),
                nn.GELU(),
                nn.Linear(self.moe_router_hidden, router_out),
            )
            # Compression-safety critic head (option 2). Predicts per-expert
            # loss from the router input features; trained to regress to the
            # actual measured losses_b4.detach() so it learns a chunk-specific
            # quality prior. Only built for single_pick (per_quad_mask uses a
            # different decision surface and is not yet supported by critic).
            if self.moe_critic_weight > 0 and self.moe_routing_mode == "single_pick":
                self.moe_critic_head = nn.Sequential(
                    nn.Linear(router_in, self.moe_critic_hidden),
                    nn.GELU(),
                    nn.Linear(self.moe_critic_hidden, self.moe_num_experts),
                )
            # Cross-attn ExpertHeads only needed for shared_h16 mode (where body
            # output length is fixed at action_horizon and experts must compress).
            if self.moe_body_mode == "shared_h16":
                # Expert specs: (target_h, source_slice_end). source_slice_end indicates
                # how many positions of the body output the expert reads (16 = full,
                # 8 = first 8 only). For per_expert_h mode this is unused.
                expert_specs = [
                    (config.action_horizon, config.action_horizon),                   # 1: main 16
                    (config.action_horizon // 2, config.action_horizon),              # 2: m8 from 16
                    (config.action_horizon // 4, config.action_horizon // 2),         # 3: m4 from 8
                ]
                if self.moe_num_experts == 4:
                    expert_specs.append(
                        (config.action_horizon // 2, config.action_horizon // 2)      # 4: n8 from 8
                    )
                self.moe_expert_specs = expert_specs
                self.moe_experts = nn.ModuleList([
                    ExpertHead(
                        target_h=th,
                        body_dim=self.hidden_size,
                        action_dim=self.action_dim,
                        n_layers=self.moe_expert_n_layers,
                    ) for (th, _) in expert_specs
                ])
            elif self.moe_body_mode == "per_expert_h":
                # source_h per expert (for body forward). target_h equals source_h
                # because each expert's body forward is at its own horizon.
                # Order (3-expert): [main_16, m8_8, m4_4]
                # Order (4-expert): [main_16, m8_8, m4_4, n8_8]
                self.moe_expert_horizons = [
                    config.action_horizon,        # main 16
                    config.action_horizon // 2,   # m8 8 (compressed sum-pair from 16)
                    config.action_horizon // 4,   # m4 4 (compressed sum-pair from 8)
                ]
                if self.moe_num_experts == 4:
                    self.moe_expert_horizons.append(config.action_horizon // 2)  # n8 raw 8
                # Decoder per expert (each is CategorySpecificMLP). For per_expert_h,
                # we reuse existing main/m8/m4/n8 decoders.
                # (Built above.)
            else:
                raise ValueError(f"Unknown moe_body_mode: {self.moe_body_mode}")
            # Router warmup counter (separate from aux warmup)
            self.register_buffer(
                "_moe_forward_step", torch.zeros((), dtype=torch.long), persistent=True
            )
            # [B] Per-expert EMA of mean loss for KL target scale normalization.
            # Init to 1.0 so the first step's normalized loss equals the raw loss.
            self.register_buffer(
                "_moe_loss_ema",
                torch.ones(self.moe_num_experts, dtype=torch.float32),
                persistent=True,
            )
            # Per-expert running mean/var of losses_b4 (for z-score target normalization).
            # Buffers are registered when moe_target_normalize is true; if False, the
            # forward path skips them entirely so existing ckpts still load.
            if self.moe_target_normalize and self.moe_routing_mode == "single_pick":
                K = self.moe_num_experts
                self.register_buffer(
                    "_moe_loss_running_mean", torch.zeros(K), persistent=True
                )
                self.register_buffer(
                    "_moe_loss_running_var", torch.ones(K), persistent=True
                )
                self.register_buffer(
                    "_moe_norm_inited", torch.zeros((), dtype=torch.long), persistent=True
                )
        # Ensemble-consistency loss (see docs/ENSEMBLE_DESIGN.md)
        self.use_ensemble_consistency_loss = bool(getattr(config, "use_ensemble_consistency_loss", False))
        self.ensemble_consistency_weight = float(getattr(config, "ensemble_consistency_weight", 0.1))
        self.ensemble_consistency_warmup_steps = int(getattr(config, "ensemble_consistency_warmup_steps", 2000))

        # [Method 1] Distillation: m8 mimics aux f2 (frozen target).
        self.m8_distill_from_f2_weight = float(getattr(config, "m8_distill_from_f2_weight", 0.0))

        # [Method 2] m8 refinement: tiny residual decoder on top of m8 output.
        # Input  = pooled VL features (D_in) ⊕ flat m8 prediction (8 * action_dim).
        # Output = 8 * action_dim residual added to m8 prediction.
        # Backward compat: only built when use_m8_refinement=True; inference path
        # only uses it when explicitly requested.
        self.use_m8_refinement = bool(getattr(config, "use_m8_refinement", False))
        self.m8_refinement_weight = float(getattr(config, "m8_refinement_weight", 1.0))
        if self.use_m8_refinement:
            assert self.use_merged_8_head, "use_m8_refinement requires use_merged_8_head=True"
            refine_hidden = int(getattr(config, "m8_refinement_hidden", 512))
            refine_in = config.backbone_embedding_dim + 8 * self.action_dim
            self.m8_refinement = nn.Sequential(
                nn.Linear(refine_in, refine_hidden), nn.GELU(),
                nn.Linear(refine_hidden, refine_hidden), nn.GELU(),
                nn.Linear(refine_hidden, 8 * self.action_dim),
            )
            # Zero-init last layer so refine starts as identity (delta=0).
            nn.init.zeros_(self.m8_refinement[-1].weight)
            nn.init.zeros_(self.m8_refinement[-1].bias)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # print(f"TY: In init: config.noise_beta_alpha: {config.noise_beta_alpha.dtype}")
        # print(f"TY: In init: config.noise_beta_beta: {config.noise_beta_beta.dtype}")
        # self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.beta_dist = Beta(
            torch.tensor(config.noise_beta_alpha, dtype=torch.float32),
            torch.tensor(config.noise_beta_beta,  dtype=torch.float32),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        # if self.freeze_decode_layer:
        #     self.decode_layer.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        # with torch.cuda.amp.autocast(enabled=False):
        #     # NOTE: self.beta_dist was built once on fp32 scalars, so it will
        #     # stay fp32 inside this block
        #     sample_fp32 = self.beta_dist.sample([batch_size]).to(device=device, dtype=torch.float32)

        # # Now cast to the requested dtype (bfloat16 when AMP is on)
        # sample = sample_fp32.to(dtype)
        # print(f"### In sample time, sample: {sample.dtype}")
        # # sample = sample.to(device, dtype=dtype)
        # print(f"### In sample time, sample: {sample.dtype}")
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _compress_actions(self, actions_extended, action_mask_extended, factor):
        """Compress every `factor` adjacent actions to produce 16-step target.

        Continuous dims: SUM within each group (delta accumulation).
        Discrete dims (self.discrete_action_dims): LAST value of each group
            (sum is meaningless for binary signals).
        """
        B, T_ext, D = actions_extended.shape
        H = T_ext // factor
        actions_extended = actions_extended[:, : H * factor]
        action_mask_extended = action_mask_extended[:, : H * factor]
        grouped = actions_extended.reshape(B, H, factor, D)         # (B, H, f, D)
        compressed = grouped.sum(dim=2)                              # default: sum
        if self.discrete_action_dims:
            disc = self.discrete_action_dims
            # Take last value of each group for discrete dims
            compressed[..., disc] = grouped[:, :, -1, :][..., disc]
        mask_grouped = action_mask_extended.reshape(B, H, factor, D)
        compressed_mask = mask_grouped.all(dim=2)
        return compressed, compressed_mask

    def _per_sample_flow_matching_loss(self, vl_embs, vl_attn_mask, state_features, embodiment_id,
                                       clean_actions, action_mask, decoder, device, t_shared=None):
        """Like _flow_matching_loss but returns per-sample loss (B,) not scalar.

        For MoE soft-mixture training. Body forward is at clean_actions's horizon.
        """
        noise = torch.randn(clean_actions.shape, device=clean_actions.device, dtype=clean_actions.dtype)
        if t_shared is not None:
            t = t_shared.to(dtype=clean_actions.dtype)
        else:
            t = self.sample_time(clean_actions.shape[0], device=clean_actions.device, dtype=clean_actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * clean_actions
        velocity = clean_actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = decoder(model_output, embodiment_id)
        pred_actions = pred[:, -clean_actions.shape[1]:]
        per_dim_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask    # (B, h, A)
        # Reduce over (h, A) but keep batch dim for soft-mixture weighting.
        denom_per_sample = action_mask.reshape(action_mask.shape[0], -1).sum(-1).clamp(min=1.0)
        per_sample_loss = per_dim_loss.reshape(per_dim_loss.shape[0], -1).sum(-1) / denom_per_sample
        return per_sample_loss

    def _flow_matching_loss(self, vl_embs, vl_attn_mask, state_features, embodiment_id,
                            clean_actions, action_mask, decoder, device,
                            grad_scale_to_body: float = 1.0,
                            t_shared=None, return_pred: bool = False,
                            return_noisy_t: bool = False):
        """Run a single flow-matching forward pass and compute MSE loss.

        Args:
            grad_scale_to_body: If < 1.0, the gradient flowing from this loss
                INTO the shared DiT body (via model_output) is scaled by this
                value. The decoder itself receives full gradient. Used to
                protect pretrained DiT representations from auxiliary losses.
            t_shared: Optional (B,) tensor of timesteps to reuse across heads
                (enables ensemble-consistency loss). If None, samples fresh.
            return_pred: If True, also return the predicted velocity
                (B, horizon, D) — used by ensemble-consistency loss.

        Returns: scalar loss (and pred velocity if return_pred=True).
        """
        noise = torch.randn(clean_actions.shape, device=clean_actions.device, dtype=clean_actions.dtype)
        if t_shared is not None:
            t = t_shared.to(dtype=clean_actions.dtype)
        else:
            t = self.sample_time(clean_actions.shape[0], device=clean_actions.device, dtype=clean_actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * clean_actions
        velocity = clean_actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )

        # Dampen gradient flowing into the shared DiT body (and shared encoders)
        # while preserving full gradient through the decoder itself.
        if grad_scale_to_body < 1.0 and grad_scale_to_body >= 0.0:
            model_output_for_decoder = GradScale.apply(model_output, grad_scale_to_body)
        else:
            model_output_for_decoder = model_output

        pred = decoder(model_output_for_decoder, embodiment_id)
        pred_actions = pred[:, -clean_actions.shape[1]:]

        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        denom = action_mask.sum().clamp(min=1.0)
        loss = loss.sum() / denom
        if return_noisy_t:
            # Always include pred too; t is the (B, 1, 1) tensor used internally.
            return loss, pred_actions, noisy_trajectory, t
        if return_pred:
            return loss, pred_actions
        return loss

    def _moe_compute_router_inputs(self, vl_embeds, state_features):
        """Pool VL + state to a single feature vector per sample for the router."""
        vl_pooled = vl_embeds.mean(dim=1)               # (B, D_vl)
        state_pooled = state_features.mean(dim=1)       # (B, D_s)
        return torch.cat([vl_pooled, state_pooled], dim=-1)  # (B, D_vl + D_s)

    def _moe_expert_targets(self, actions, action_mask):
        """Build the per-expert (clean, mask) supervision targets.

        3-expert order:
            0: main 16-raw  -> actions[:, :H], action_mask[:, :H]
            1: m8 (sum-pair from 16) -> _compress_actions(actions[:, :H], 2)
            2: m4 (sum-pair from 8)  -> _compress_actions(actions[:, :H/2], 2)
        4-expert adds:
            3: n8 raw 8     -> actions[:, :H/2], action_mask[:, :H/2]
        H = self.config.action_horizon (e.g. 16 for robocasa).
        """
        H = self.config.action_horizon
        H8 = H // 2
        c1, m1 = actions[:, :H], action_mask[:, :H]
        c2, m2 = self._compress_actions(actions[:, :H], action_mask[:, :H], factor=2)
        c3, m3 = self._compress_actions(actions[:, :H8], action_mask[:, :H8], factor=2)
        out = [(c1, m1), (c2, m2), (c3, m3)]
        if self.moe_num_experts == 4:
            c4, m4 = actions[:, :H8], action_mask[:, :H8]
            out.append((c4, m4))
        return out

    def _moe_forward(self, vl_embs, vl_attn_mask, state_features, embodiment_id,
                     actions, action_mask, device):
        """K-expert MoE soft-mixture training forward (K = self.moe_num_experts).

        Steps:
          1. Build per-expert supervision (c_i, m_i) targets.
          2. Run experts (per_expert_h: K separate body forwards; shared_h16: 1
             body forward at h=H, then cross-attn ExpertHead per expert).
          3. Compute per-sample MSE loss for each expert -> (B, K).
          4. Run router -> (B, K) probs.
          5. Soft mixture:    total = (probs * losses).sum(-1).mean()
                + load balance: λ_b * (mean_probs - 1/K)^2
                + supervised KL: λ_s * KL(router || softmax(-losses / τ_target))
        """
        B = actions.shape[0]
        targets = self._moe_expert_targets(actions, action_mask)  # list of K (clean, mask)
        K = self.moe_num_experts

        # --------- compute per-expert per-sample loss ---------
        per_sample = []  # list of (B,) tensors
        if self.moe_body_mode == "per_expert_h":
            decoders = [
                self.action_decoder,
                self.m8_action_decoder,
                self.m4_action_decoder,
            ]
            if K == 4:
                decoders.append(self.n8_action_decoder)
            # [A] Optional shared timestep across experts so per-sample loss is
            # comparable across experts (KL target then reflects expert quality
            # rather than per-expert t-noise variance).  Noise is NOT shared.
            t_shared = self.sample_time(B, device=device, dtype=actions.dtype) \
                if self.moe_shared_t else None
            for (clean_i, mask_i), decoder_i in zip(targets, decoders):
                # Each expert: body forward at clean_i.shape[1] (per_expert_h).
                loss_i = self._per_sample_flow_matching_loss(
                    vl_embs, vl_attn_mask, state_features, embodiment_id,
                    clean_i, mask_i.to(clean_i.dtype), decoder_i, device,
                    t_shared=t_shared,
                )
                per_sample.append(loss_i)
        elif self.moe_body_mode == "shared_h16":
            # 1 body forward at h=H. All experts share noise+t derived from this.
            H = self.config.action_horizon
            clean_full = actions[:, :H]
            mask_full = action_mask[:, :H]
            # Sample one (noise, t) shared across experts.
            noise_full = torch.randn(clean_full.shape, device=device, dtype=clean_full.dtype)
            t_shared = self.sample_time(B, device=device, dtype=clean_full.dtype)
            t = t_shared[:, None, None]
            noisy_full = (1 - t) * noise_full + t * clean_full
            t_disc = (t[:, 0, 0] * self.num_timestep_buckets).long()
            action_features = self.action_encoder(noisy_full, t_disc, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_disc,
                return_all_hidden_states=False,
            )
            body_out_full = model_output[:, -H:]   # (B, H, D_body)
            body_out_first8 = body_out_full[:, : H // 2]   # (B, H/2, D_body)

            # Per-expert: cross-attn ExpertHead → target_h velocity. Loss vs (clean - noise)
            # at expert's target_h, where noise is derived from noise_full to keep flow
            # matching theory consistent (e.g. compressed noise = sum-pair of noise_full).
            for i, ((clean_i, mask_i), expert_i) in enumerate(zip(targets, self.moe_experts)):
                target_h = clean_i.shape[1]
                source_slice_end = self.moe_expert_specs[i][1]
                ctx = body_out_full if source_slice_end == H else body_out_first8
                pred_v = expert_i(ctx)                  # (B, target_h, A)
                # Build per-expert noise (consistent with noise_full).
                if i == 0:                               # main 16-raw: noise unchanged
                    noise_i = noise_full
                elif i == 1:                             # m8 from 16: sum-pair noise_full
                    noise_i_sum, _ = self._compress_actions(noise_full, mask_full, factor=2)
                    noise_i = noise_i_sum
                elif i == 2:                             # m4 from 8: sum-pair first-8 of noise_full
                    noise_first8 = noise_full[:, : H // 2]
                    mask_first8 = mask_full[:, : H // 2]
                    noise_i_sum, _ = self._compress_actions(noise_first8, mask_first8, factor=2)
                    noise_i = noise_i_sum
                else:                                    # n8 raw 8 (4-expert only): first 8 of noise_full
                    noise_i = noise_full[:, : H // 2]
                v_target = clean_i - noise_i             # (B, target_h, A)
                per_dim = F.mse_loss(pred_v, v_target, reduction="none") * mask_i.to(clean_i.dtype)
                denom = mask_i.reshape(B, -1).sum(-1).clamp(min=1.0)
                per_sample_i = per_dim.reshape(B, -1).sum(-1) / denom
                per_sample.append(per_sample_i)
        else:
            raise ValueError(f"Unknown moe_body_mode: {self.moe_body_mode}")

        losses_b4 = torch.stack(per_sample, dim=-1)     # (B, K)

        # --------- router ---------
        router_in = self._moe_compute_router_inputs(vl_embs, state_features)   # (B, D)
        router_logits = self.head_router(router_in.float())                    # (B, 4)
        # Single derivation chain: log_softmax once, softmax = exp(log_softmax).
        # Avoids separate softmax / log_softmax graph nodes that cause DDP's
        # "marked ready twice" complaint when find_unused_parameters is True.
        scaled_logits = router_logits / max(self.moe_router_temp, 1e-3)
        log_router = F.log_softmax(scaled_logits, dim=-1)
        router_probs = log_router.exp()

        # Anti-collapse: clip min prob, then renormalize. This guarantees every
        # expert receives ≥ moe_min_prob × loss_i gradient so dead experts can't
        # drift from lack of supervision. Standard "noisy-top-k" variant.
        if self.moe_min_prob > 0:
            router_probs = torch.clamp(router_probs, min=self.moe_min_prob)
            router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)

        # Pre-warmup uniform: for the first N steps, force prob = 1/K. This lets
        # all experts learn from real GT supervision before the router commits.
        if self.training and self.moe_uniform_warmup_steps > 0 and \
                int(self._moe_forward_step.item()) < self.moe_uniform_warmup_steps:
            uniform = torch.full_like(router_probs, 1.0 / self.moe_num_experts)
            router_probs = uniform

        # --------- per-expert running mean/var update (z-score normalization) ---------
        # When moe_target_normalize is ON, maintain EMA of per-expert loss stats
        # over training so target signal can be z-scored (bias-free across experts).
        # Buffers are only registered/used in single_pick mode.
        normalize_on = getattr(self, "moe_target_normalize", False) and \
                       self.moe_routing_mode == "single_pick" and \
                       hasattr(self, "_moe_loss_running_mean")
        if normalize_on and self.training:
            with torch.no_grad():
                cur_mean = losses_b4.detach().float().mean(dim=0)             # (K,)
                cur_var  = losses_b4.detach().float().var(dim=0, unbiased=False)
                alpha = float(self.moe_target_normalize_ema)
                if int(self._moe_norm_inited.item()) == 0:
                    # First step: init buffers to current batch stats (avoid 0/1 init bias)
                    self._moe_loss_running_mean.copy_(cur_mean)
                    self._moe_loss_running_var.copy_(cur_var + 1e-6)
                    self._moe_norm_inited.fill_(1)
                else:
                    self._moe_loss_running_mean.mul_(1.0 - alpha).add_(cur_mean * alpha)
                    self._moe_loss_running_var.mul_(1.0 - alpha).add_((cur_var + 1e-6) * alpha)

        # --------- joint quality+length adjusted losses (option 1) ---------
        # If moe_length_cost_weight > 0, add a per-expert length-cost term.
        # - normalize OFF: applied to BOTH soft-mixture and target in RAW loss units
        # - normalize ON : applied only to TARGET in z-score (σ) units; soft-mixture
        #                  stays on raw losses (purest expert training signal)
        if getattr(self, "moe_length_cost_weight", 0.0) > 0 and not normalize_on:
            horizons = torch.tensor(self.moe_expert_horizons, device=device,
                                    dtype=losses_b4.dtype)
            length_cost = horizons / horizons.max()                    # (K,)
            losses_adj = losses_b4 + self.moe_length_cost_weight * length_cost
        else:
            losses_adj = losses_b4

        # --------- soft mixture loss (weighted per-sample) ---------
        # Detach losses_adj inside router_probs path? No — we want gradient to flow
        # so router learns to favor low-loss experts. But `losses_b4` is already a
        # function of all expert/body params; multiplying by router_probs lets router
        # gradient through. This is standard soft-MoE.
        soft_mixture = (router_probs * losses_adj.to(router_probs.dtype)).sum(-1).mean()

        # --------- load balance regularizer ---------
        mean_p = router_probs.mean(dim=0)               # (4,)
        uniform = 1.0 / self.moe_num_experts
        if self.moe_balance_type == "switch":
            # [C] Switch-Transformer balance: K · Σ f_i.detach() · p_i
            # where f_i is the argmax-assignment fraction over the batch.
            with torch.no_grad():
                hard_pick = router_probs.argmax(dim=-1)
                f_i = F.one_hot(hard_pick, num_classes=self.moe_num_experts).to(
                    router_probs.dtype
                ).mean(dim=0)
            balance = self.moe_num_experts * (f_i.detach() * mean_p).sum()
        else:
            balance = ((mean_p - uniform) ** 2).sum()

        # --------- learnable compression-safety critic (option 2) ---------
        # Critic predicts each expert's per-sample loss from the router input
        # features. Trained to regress to actual `losses_b4.detach()`. When
        # active, replaces actual losses in the router's KL target — router
        # then aligns to a CHUNK-SPECIFIC quality prediction that's already
        # combined with the length cost.
        critic_loss = torch.zeros((), device=device, dtype=losses_b4.dtype)
        if getattr(self, "moe_critic_weight", 0.0) > 0 and hasattr(self, "moe_critic_head"):
            critic_pred = self.moe_critic_head(router_in.float())               # (B, K)
            critic_loss = F.mse_loss(critic_pred, losses_b4.detach().float())
            target_basis = critic_pred.detach()
        else:
            target_basis = losses_b4.detach().float()

        # --------- supervised KL on router (per-sample) ---------
        # [B] EMA-normalize per-expert losses (mean-only) before KL target if
        # moe_normalize_loss_for_kl is set; this is jimin-dev's emaKL trick.
        # Falls through to legacy z-score (mean+var) path if normalize_on is
        # active; otherwise raw target_basis.
        if self.moe_normalize_loss_for_kl:
            with torch.no_grad():
                if self.training:
                    m = self.moe_loss_ema_momentum
                    batch_mean = losses_b4.detach().float().mean(dim=0).clamp(min=1e-8)
                    self._moe_loss_ema.mul_(m).add_(batch_mean * (1.0 - m))
                ema = self._moe_loss_ema.detach().clamp(min=1e-8).to(target_basis.dtype)
            target_signal = target_basis / ema.unsqueeze(0)
        elif normalize_on:
            mean_buf = self._moe_loss_running_mean.to(target_basis.dtype)
            std_buf = (self._moe_loss_running_var + 1e-6).sqrt().to(target_basis.dtype)
            target_signal = (target_basis - mean_buf) / std_buf
        else:
            target_signal = target_basis
        if getattr(self, "moe_length_cost_weight", 0.0) > 0:
            horizons = torch.tensor(self.moe_expert_horizons, device=device,
                                    dtype=target_signal.dtype)
            target_signal = target_signal + self.moe_length_cost_weight * (horizons / horizons.max())
        target_dist = F.softmax(-target_signal / max(self.moe_target_temp, 1e-3), dim=-1)
        # Reuse log_router computed above (single graph node from router_logits).
        kl_supervise = F.kl_div(log_router, target_dist, reduction="batchmean")

        # --------- warmup for balance + supervise ---------
        if self.moe_router_warmup_steps > 0:
            warmup = min(1.0, float(self._moe_forward_step.item()) / self.moe_router_warmup_steps)
        else:
            warmup = 1.0

        total = soft_mixture + (warmup * self.moe_balance_weight) * balance \
                             + (warmup * self.moe_supervise_weight) * kl_supervise \
                             + getattr(self, "moe_critic_weight", 0.0) * critic_loss

        # --------- compression-aware router prior ---------
        # Negative bonus pushing router probability toward shorter (more
        # compressed) experts. Weight ~ 1/horizon, normalized so a uniform
        # router gets a fixed (subtractable) baseline. This nudges the router
        # to favor m8/m4 over main when their losses are close, without
        # changing the per-expert supervision (preserves specialization).
        if getattr(self, "moe_compression_weight", 0.0) > 0:
            horizons = torch.tensor(
                self.moe_expert_horizons, device=device, dtype=mean_p.dtype
            )
            inv = 1.0 / horizons
            inv = inv / inv.sum()                         # normalize → sums to 1
            compression_bonus = -(self.moe_compression_weight) * (mean_p * inv).sum()
            total = total + warmup * compression_bonus

        # Anti-unused-grad term: tiny L2 on every expert + decoder param so DDP
        # reducer always sees gradient on every parameter. In shared_h16 mode the
        # legacy CategorySpecificMLP decoders (action_decoder, m8/m4/n8) are also
        # unused (replaced by cross-attn ExpertHeads) — include them too. Magnitude
        # is so small it doesn't affect learning, but it touches every param's
        # autograd path so DDP sees gradient.
        moe_param_modules = [self.action_decoder]
        if hasattr(self, "moe_experts"):
            moe_param_modules.append(self.moe_experts)
        for attr in ("m8_action_decoder", "m4_action_decoder", "n8_action_decoder",
                     "aux_action_decoders", "m8_refinement"):
            if hasattr(self, attr):
                moe_param_modules.append(getattr(self, attr))
        anti_unused = sum(
            (p.float().pow(2).sum() for m in moe_param_modules for p in m.parameters()),
            start=torch.zeros((), device=device, dtype=torch.float32),
        )
        total = total + 1e-12 * anti_unused

        # --------- bookkeeping ---------
        if self.training:
            with torch.no_grad():
                self._moe_forward_step += 1

        # Per-expert mean loss + router mean prob for logging.
        out = {
            "loss": total,
            "loss_main": losses_b4[:, 0].mean(),
            "loss_m8":   losses_b4[:, 1].mean(),
            "loss_m4":   losses_b4[:, 2].mean(),
            "loss_soft_mixture": soft_mixture.detach(),
            "loss_balance":      balance.detach(),
            "loss_kl_supervise": kl_supervise.detach(),
            "router_p_main":     mean_p[0].detach(),
            "router_p_m8":       mean_p[1].detach(),
            "router_p_m4":       mean_p[2].detach(),
            "moe_warmup":        torch.tensor(warmup, device=device),
        }
        if self.moe_num_experts == 4:
            out["loss_n8"] = losses_b4[:, 3].mean()
            out["router_p_n8"] = mean_p[3].detach()
        return BatchFeature(data=out)

    def _moe_forward_per_quad_mask(self, vl_embs, vl_attn_mask, state_features,
                                   embodiment_id, actions, action_mask, device):
        """Per-quad mask routing: 4 quads (k=0..3, each spans 4 env-steps of the
        16-step horizon) × {3 or 4} options:
            c=0: main 4-raw  (main_pred[4k:4k+4] vs GT[4k:4k+4])
            c=1: m8 2-comp   (m8_pred[2k:2k+2]  vs sum-pair(GT[4k:4k+4]))
            c=2: m4 1-deep   (m4_pred[k]        vs sum-pair-of-pair(GT[4k:4k+4]))
            c=3: n8 4-raw    (n8_pred[4k:4k+4] vs GT[4k:4k+4])   — only quads 0,1

        n8 covers env-steps 0..7 only, so c=3 is masked out for quads 2,3 when
        active. When `moe_per_quad_use_n8`=False (default), C=3 (legacy 4q3o).
        Loss = Σ_k Σ_c p_{b,k,c} · MSE_{b,k,c}, masked soft-mixture. Per-quad
        load-balance pushes each quad's mean prob toward 1/n_avail_k.
        """
        B = actions.shape[0]
        H = self.config.action_horizon            # 16
        H8 = H // 2                                # 8
        Q = self._moe_num_quads                    # 4
        C = self._moe_options_per_quad             # 3 or 4
        use_n8 = getattr(self, "_per_quad_use_n8", False)
        # (Q, C) bool mask, available options per quad. For non-n8 case full True.
        option_mask = self._per_quad_option_mask if hasattr(self, "_per_quad_option_mask") \
            else torch.ones(Q, C, dtype=torch.bool, device=device)
        option_mask = option_mask.to(device)

        # === Per-expert (main, m8, m4 [, n8]) per-position loss ===
        c_main, m_main = actions[:, :H], action_mask[:, :H]
        c_m8,   m_m8   = self._compress_actions(actions[:, :H], action_mask[:, :H], factor=2)
        c_m4,   m_m4   = self._compress_actions(actions[:, :H8], action_mask[:, :H8], factor=2)
        # n8 sees raw 8-step (first half of the chunk).
        if use_n8:
            c_n8, m_n8 = actions[:, :H8], action_mask[:, :H8]

        def _denoise_keep_per_pos(clean_i, mask_i, decoder_i):
            """Returns (B, T_i, D) per-position squared error against velocity target."""
            mask_f = mask_i.to(clean_i.dtype)
            return self._per_position_flow_matching_loss(
                vl_embs, vl_attn_mask, state_features, embodiment_id,
                clean_i, mask_f, decoder_i, device,
            )
        per_pos_main = _denoise_keep_per_pos(c_main, m_main, self.action_decoder)    # (B, 16, D)
        per_pos_m8   = _denoise_keep_per_pos(c_m8,   m_m8,   self.m8_action_decoder)  # (B, 8, D)
        per_pos_m4   = _denoise_keep_per_pos(c_m4,   m_m4,   self.m4_action_decoder)  # (B, 4, D)
        if use_n8:
            per_pos_n8 = _denoise_keep_per_pos(c_n8, m_n8, self.n8_action_decoder)    # (B, 8, D)

        # === Slice per quad: each quad covers 4 env-steps ===
        losses_BQC = torch.zeros((B, Q, C), device=device, dtype=per_pos_main.dtype)
        for k in range(Q):
            # mean over T and D so loss magnitudes are comparable across options.
            losses_BQC[:, k, 0] = per_pos_main[:, 4 * k:4 * k + 4].reshape(B, -1).mean(-1)
            losses_BQC[:, k, 1] = per_pos_m8[:,   2 * k:2 * k + 2].reshape(B, -1).mean(-1)
            losses_BQC[:, k, 2] = per_pos_m4[:,   k:k + 1].reshape(B, -1).mean(-1)
            if use_n8 and option_mask[k, 3].item():
                # n8 has 8 raw actions covering env-steps 0..7, so quad k ∈ {0,1}.
                losses_BQC[:, k, 3] = per_pos_n8[:, 4 * k:4 * k + 4].reshape(B, -1).mean(-1)

        # === Router: (B, Q*C) -> (B, Q, C) softmax over valid options ===
        router_in = self._moe_compute_router_inputs(vl_embs, state_features)
        router_logits = self.head_router(router_in.float()).reshape(B, Q, C)
        scaled = router_logits / max(self.moe_router_temp, 1e-3)
        # Mask invalid options with large negative logits BEFORE softmax so they
        # get prob ~0 (and excluded from KL/balance via masking later).
        if use_n8:
            neg_inf = torch.full_like(scaled, -1e9)
            mask_expand = option_mask[None].expand_as(scaled)               # (B, Q, C)
            scaled = torch.where(mask_expand, scaled, neg_inf)
        log_router = F.log_softmax(scaled, dim=-1)
        router_probs = log_router.exp()

        # n_avail_k: number of available options per quad (used for clamp & balance).
        n_avail = option_mask.sum(dim=-1).to(router_probs.dtype)             # (Q,)
        if self.moe_min_prob > 0:
            # Clamp only over VALID options. Invalid options stay at 0.
            mask_expand_f = option_mask[None].expand_as(router_probs).to(router_probs.dtype)
            router_probs = torch.where(option_mask[None].expand_as(router_probs),
                                       torch.clamp(router_probs, min=self.moe_min_prob),
                                       router_probs)
            row_sum = router_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            router_probs = router_probs / row_sum
        if self.training and self.moe_uniform_warmup_steps > 0 and \
                int(self._moe_forward_step.item()) < self.moe_uniform_warmup_steps:
            # Uniform over available options per quad.
            mask_expand_f = option_mask[None].expand_as(router_probs).to(router_probs.dtype)
            uniform = mask_expand_f / n_avail[None, :, None].clamp_min(1.0)
            router_probs = uniform

        # === Soft mixture loss (invalid options contribute 0 since prob=0) ===
        soft_mixture = (router_probs * losses_BQC.to(router_probs.dtype)).sum(-1).mean()

        # Per-quad load balance: push each quad's mean prob to uniform (1/n_avail_k).
        mean_p_QC = router_probs.mean(dim=0)               # (Q, C)
        target_uniform = option_mask.to(mean_p_QC.dtype) / n_avail[:, None].clamp_min(1.0)
        balance = ((mean_p_QC - target_uniform) ** 2 * option_mask.to(mean_p_QC.dtype)).sum()

        # Supervised KL per (b, k): target = softmax(-loss/τ) over valid options.
        target_logits = -losses_BQC.detach().float() / max(self.moe_target_temp, 1e-3)
        if use_n8:
            target_logits = torch.where(option_mask[None].expand_as(target_logits),
                                        target_logits, torch.full_like(target_logits, -1e9))
        target_dist = F.softmax(target_logits, dim=-1)
        kl_supervise = F.kl_div(log_router, target_dist, reduction="batchmean")

        if self.moe_router_warmup_steps > 0:
            warmup = min(1.0, float(self._moe_forward_step.item()) / self.moe_router_warmup_steps)
        else:
            warmup = 1.0

        total = soft_mixture + (warmup * self.moe_balance_weight) * balance \
                             + (warmup * self.moe_supervise_weight) * kl_supervise

        # Compression-aware bonus: prefer c=2 (m4, 1 action) > c=1 (m8) > c=0 (main).
        # n8 (c=3) has same env-coverage and action count as main → same bonus.
        if getattr(self, "moe_compression_weight", 0.0) > 0:
            base = [1.0 / 4, 1.0 / 2, 1.0 / 1]
            if use_n8:
                base.append(1.0 / 4)
            inv_actions = torch.tensor(base, device=device, dtype=mean_p_QC.dtype)
            inv_actions = inv_actions / inv_actions.sum()
            # Apply mask so invalid quad/option pairs don't contribute (their prob
            # is 0 anyway, but this keeps the bonus comparable across configs).
            comp_bonus = -(self.moe_compression_weight) * \
                (mean_p_QC * inv_actions[None, :] * option_mask.to(mean_p_QC.dtype)).sum()
            total = total + warmup * comp_bonus

        # Anti-unused gradient term (covers all expert decoders).
        moe_param_modules = [self.action_decoder, self.m8_action_decoder, self.m4_action_decoder]
        if hasattr(self, "n8_action_decoder"):
            moe_param_modules.append(self.n8_action_decoder)
        anti = sum(
            (p.float().pow(2).sum() for m in moe_param_modules for p in m.parameters()),
            start=torch.zeros((), device=device, dtype=torch.float32),
        )
        total = total + 1e-12 * anti

        if self.training:
            with torch.no_grad():
                self._moe_forward_step += 1

        out = {
            "loss": total,
            "loss_main": losses_BQC[:, :, 0].mean(),
            "loss_m8":   losses_BQC[:, :, 1].mean(),
            "loss_m4":   losses_BQC[:, :, 2].mean(),
            "loss_soft_mixture": soft_mixture.detach(),
            "loss_balance":      balance.detach(),
            "loss_kl_supervise": kl_supervise.detach(),
            "router_p_main_mean": mean_p_QC[:, 0].mean().detach(),
            "router_p_m8_mean":   mean_p_QC[:, 1].mean().detach(),
            "router_p_m4_mean":   mean_p_QC[:, 2].mean().detach(),
            "moe_warmup":         torch.tensor(warmup, device=device),
        }
        return BatchFeature(data=out)

    def _per_position_flow_matching_loss(self, vl_embs, vl_attn_mask, state_features,
                                         embodiment_id, clean_actions, action_mask,
                                         decoder, device):
        """Same as `_per_sample_flow_matching_loss` but returns (B, T, D) per-position
        squared error (no summing). Used by per_quad_mask routing to slice loss
        per quad. Matches the noise sampling and timestep schedule of the
        standard per-sample helper."""
        B, T, D = clean_actions.shape
        noise = torch.randn(clean_actions.shape, device=device, dtype=clean_actions.dtype)
        t_full = self.sample_time(B, device=device, dtype=clean_actions.dtype)
        t = t_full[:, None, None]
        noisy = (1 - t) * noise + t * clean_actions
        velocity = clean_actions - noise
        t_disc = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy, t_disc, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_disc,
            return_all_hidden_states=False,
        )
        pred = decoder(model_output, embodiment_id)
        pred_actions = pred[:, -clean_actions.shape[1]:]
        per_pos = (pred_actions - velocity) ** 2 * action_mask
        return per_pos

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        vl_attn_mask = backbone_output.backbone_attention_mask

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # === Main loss: original 16-step action ===
        actions = action_input.action
        action_mask = action_input.action_mask

        # ============================================================
        # MoE forward branch (when use_moe_routing=True). Skips the standard
        # multi-horizon / m8 / consistency / refinement code path entirely.
        # ============================================================
        if self.use_moe_routing:
            if self.moe_routing_mode == "per_quad_mask":
                return self._moe_forward_per_quad_mask(
                    vl_embeds, vl_attn_mask, state_features, embodiment_id,
                    actions, action_mask, device,
                )
            return self._moe_forward(
                vl_embeds, vl_attn_mask, state_features, embodiment_id,
                actions, action_mask, device,
            )

        # === Shared-t for ensemble-consistency loss ===
        # When enabled, main / m8 / f2 all use the same (B,) timestep tensor,
        # so their predicted velocities can be compared directly.
        consist_on = (self.use_ensemble_consistency_loss and self.use_merged_8_head
                      and self.use_multi_horizon_loss)
        if consist_on:
            t_shared = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)
        else:
            t_shared = None

        ret_main = self._flow_matching_loss(
            vl_embeds, vl_attn_mask, state_features, embodiment_id,
            actions, action_mask, self.action_decoder, device,
            t_shared=t_shared, return_pred=consist_on,
        )
        if consist_on:
            main_loss, v_main = ret_main
        else:
            main_loss = ret_main
            v_main = None

        # === Auxiliary multi-horizon losses ===
        aux_losses = {}
        total_loss = self.multi_horizon_main_weight * main_loss

        # === Co-main loss: merged_8 (native 8-step) ===
        # Target = _compress_actions(actions_16, factor=2) -> (B, 8, D). Full grad
        # to body, no warmup — treated as a second main.
        m8_loss = None
        v_m8 = None
        noisy_m8 = None     # captured for refinement path (1-step Euler clean est)
        t_m8 = None
        if self.use_merged_8_head:
            clean_m8, mask_m8 = self._compress_actions(actions, action_mask, factor=2)
            if mask_m8.sum() > 0:
                # When refinement is on we need (noisy, t) too — request 4-tuple.
                want_full = bool(self.use_m8_refinement)
                ret_m8 = self._flow_matching_loss(
                    vl_embeds, vl_attn_mask, state_features, embodiment_id,
                    clean_m8, mask_m8.to(clean_m8.dtype), self.m8_action_decoder, device,
                    t_shared=t_shared,
                    return_pred=(consist_on or want_full),
                    return_noisy_t=want_full,
                )
                if want_full:
                    m8_loss, v_m8, noisy_m8, t_m8 = ret_m8
                elif consist_on:
                    m8_loss, v_m8 = ret_m8
                else:
                    m8_loss = ret_m8
            else:
                m8_loss = main_loss.new_zeros(())
            total_loss = total_loss + self.merged_8_weight * m8_loss

        # Warmup factor for aux loss WEIGHTS: linearly ramp 0 → 1 over warmup_steps.
        # Use the model's internal forward counter (auto-increments per call).
        if self.use_multi_horizon_loss and self.aux_loss_warmup_steps > 0:
            warmup = min(1.0, float(self._aux_forward_step.item()) / self.aux_loss_warmup_steps)
        else:
            warmup = 1.0

        v_f2 = None
        if self.use_multi_horizon_loss and "action_extended" in action_input:
            actions_ext = action_input.action_extended
            action_mask_ext = action_input.action_extended_mask
            for f, w in zip(self.multi_horizon_factors, self.multi_horizon_loss_weights):
                clean_f, mask_f = self._compress_actions(actions_ext, action_mask_ext, factor=f)
                if mask_f.sum() == 0:
                    aux_losses[f"loss_f{f}"] = main_loss.new_zeros(())
                    continue
                decoder_f = self.aux_action_decoders[f"f{f}"]
                # Share t only for f=2 (used in consistency). f=4 keeps its own t to
                # preserve independent sampling — we don't use v_f4 in consistency.
                use_shared_t = consist_on and (f == 2)
                ret_f = self._flow_matching_loss(
                    vl_embeds, vl_attn_mask, state_features, embodiment_id,
                    clean_f, mask_f.to(clean_f.dtype), decoder_f, device,
                    grad_scale_to_body=self.aux_grad_scale_to_body,
                    t_shared=t_shared if use_shared_t else None,
                    return_pred=use_shared_t,
                )
                if use_shared_t:
                    loss_f, v_f2 = ret_f
                else:
                    loss_f = ret_f
                aux_losses[f"loss_f{f}"] = loss_f
                total_loss = total_loss + (w * warmup) * loss_f

        # === Ensemble-consistency loss (shared-t cross-head velocity MSE) ===
        consist_loss = None
        if consist_on and v_main is not None and v_m8 is not None and v_f2 is not None:
            # pair-sum main velocity to 8-step semantics
            v_main_8 = v_main[:, 0::2, :] + v_main[:, 1::2, :]
            v_f2_8 = v_f2[:, :8, :]
            # Treat f2[:8] as the anchor: pull main-pooled and m8 toward it.
            consist_loss = F.mse_loss(v_main_8, v_f2_8) + F.mse_loss(v_m8, v_f2_8)
            if self.ensemble_consistency_warmup_steps > 0:
                c_warmup = min(1.0, float(self._aux_forward_step.item())
                               / self.ensemble_consistency_warmup_steps)
            else:
                c_warmup = 1.0
            total_loss = total_loss + (self.ensemble_consistency_weight * c_warmup) * consist_loss

        # === [Method 1] Distillation: m8 ← f2 (frozen target) ===
        # One-directional, stronger than consistency: f2 is treated as a fixed
        # teacher and m8 must match it. Useful when fine-tuning a trained model
        # so m8-only inference quality approaches ensemble→8.
        m8_distill_loss = None
        if (self.m8_distill_from_f2_weight > 0
                and v_m8 is not None and v_f2 is not None):
            m8_distill_loss = F.mse_loss(v_m8, v_f2[:, :8, :].detach())
            total_loss = total_loss + self.m8_distill_from_f2_weight * m8_distill_loss

        # === [Method 2] Refinement decoder supervised loss ===
        # Use the *actual* m8 inference output as refine input: run the same
        # 4-step flow-matching denoising loop the inference path uses, with
        # no_grad to decouple refine training from m8 head training. Refine
        # learns to predict the residual from this realistic m8 output to GT.
        # Cost: extra `num_inference_timesteps` (default 4) DiT body forwards
        # per batch, all under no_grad.
        m8_refinement_loss = None
        if self.use_m8_refinement and self.use_merged_8_head:
            mask_m8_3d = mask_m8 if mask_m8.dim() == 3 else mask_m8.unsqueeze(-1).expand_as(clean_m8)
            with torch.no_grad():
                a_m8_inf = self._denoise_with_decoder(
                    vl_embeds, state_features, embodiment_id,
                    self.m8_action_decoder, horizon=8,
                ).detach()
                vl_pooled = vl_embeds.mean(dim=1)  # (B, D_vl)
            refine_in = torch.cat([vl_pooled.float(),
                                   a_m8_inf.flatten(1).float()], dim=-1)
            delta = self.m8_refinement(refine_in).view_as(a_m8_inf)
            a_refined = a_m8_inf + delta
            err = (a_refined - clean_m8) ** 2 * mask_m8_3d
            m8_refinement_loss = err.sum() / (mask_m8_3d.sum() + 1e-9)
            total_loss = total_loss + self.m8_refinement_weight * m8_refinement_loss

        # Increment internal forward counter (for warmup tracking)
        if self.use_multi_horizon_loss and self.training:
            with torch.no_grad():
                self._aux_forward_step += 1

        output_dict = {
            "loss": total_loss,
            "loss_main": main_loss,
        }
        # Expose the current warmup factor for logging/debugging
        if self.use_multi_horizon_loss:
            output_dict["aux_warmup"] = main_loss.new_tensor(warmup)
        if m8_loss is not None:
            output_dict["loss_m8"] = m8_loss
        if consist_loss is not None:
            output_dict["loss_consist"] = consist_loss
        output_dict.update(aux_losses)
        return BatchFeature(data=output_dict)

    def _denoise_with_decoder(self, vl_embeds, state_features, embodiment_id, decoder,
                              horizon=None):
        """Run a full flow-matching denoising loop using `decoder` to predict velocity.

        Args:
            horizon: action-chunk length. Defaults to `self.action_horizon` (16).
                Pass 8 for the merged_8 head.

        Returns: actions tensor of shape (B, horizon, action_dim).
        """
        if horizon is None:
            horizon = self.config.action_horizon
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )
        num_steps = self.num_inference_timesteps
        # Optional: Beta-sampled timestep schedule at inference. Shared across the
        # batch (per-call sample) — per-sample diversity comes from different
        # initial noise. Enabled at runtime by setting
        # `action_head._beta_sample_t_at_inference = True` on the policy.
        use_beta = bool(getattr(self, "_beta_sample_t_at_inference", False))
        if use_beta and num_steps > 1:
            inner = self.beta_dist.sample([num_steps - 1]).to(device=device, dtype=torch.float32)
            inner_sorted, _ = torch.sort(inner)
            t_schedule = torch.cat([
                torch.zeros(1, dtype=torch.float32, device=device),
                inner_sorted,
                torch.ones(1, dtype=torch.float32, device=device),
            ])
            t_evals = t_schedule[:-1].tolist()
            dts = (t_schedule[1:] - t_schedule[:-1]).tolist()
        else:
            t_evals = [k / float(num_steps) for k in range(num_steps)]
            dts = [1.0 / num_steps] * num_steps
        for step_idx in range(num_steps):
            t_cont = t_evals[step_idx]
            dt = dts[step_idx]
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
            pred = decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -horizon:]
            actions = actions + dt * pred_velocity
        return actions

    def _moe_shared_h16_denoise(self, vl_embeds, state_features, embodiment_id, picked: int):
        """Inference for MoE with shared_h16 body mode.

        x_t evolves in the picked expert's target_h space; body forward is at h=H
        always. Body input is reconstructed from x_t at each Euler step:
          - For target_h == H: x_t is body input directly.
          - For target_h < H (compression factor f = H // target_h): each x_t step
            is split f-ways (each = x_t[k] / f) along sequence so that the f-way
            sum equals x_t[k]. This preserves the "sum-of-pair" semantic the
            ExpertHead was trained against (target = compress(GT, f)).
          - For experts whose target_h is "raw first 8 of body" (n8): x_t fills
            the first target_h positions, the rest is fresh per-step noise (so
            body sees the same N(0,I) marginal distribution as training).
        """
        H = self.config.action_horizon
        target_h = self.moe_expert_specs[picked][0]
        source_slice_end = self.moe_expert_specs[picked][1]
        action_dim = self.config.action_dim
        bsize = vl_embeds.shape[0]
        device = vl_embeds.device

        x_t = torch.randn(
            (bsize, target_h, action_dim), dtype=vl_embeds.dtype, device=device,
        )
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full((bsize,), t_disc, device=device)

            # Reconstruct body input (length H) from x_t (length target_h).
            if target_h == H:
                body_in = x_t
            elif source_slice_end == H:
                # Compressed expert (m8 from 16, m4 from 16 — but m4 is from 8 so
                # source=8 not 16; falls into elif below).
                # x_t covers H sim steps via target_h compressed actions. Split each
                # into f equal halves so f-way sum equals x_t[k].
                f = H // target_h
                body_in = (x_t / float(f)).repeat_interleave(f, dim=1)   # (B, H, A)
            else:
                # source_slice_end < H (e.g. m4 from 8, n8 from 8). x_t covers first
                # source_slice_end positions of body; rest is fresh per-step noise.
                f = source_slice_end // target_h           # 1 (n8) or 2 (m4)
                if f > 1:
                    first_part = (x_t / float(f)).repeat_interleave(f, dim=1)   # (B, source_slice_end, A)
                else:
                    first_part = x_t                                              # (B, target_h=source_slice_end, A)
                tail_noise = torch.randn(
                    (bsize, H - source_slice_end, action_dim),
                    dtype=vl_embeds.dtype, device=device,
                )
                body_in = torch.cat([first_part, tail_noise], dim=1)             # (B, H, A)

            action_features = self.action_encoder(body_in, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
            body_out_full = model_output[:, -H:]
            ctx = body_out_full if source_slice_end == H else body_out_full[:, :source_slice_end]
            v_t = self.moe_experts[picked](ctx)              # (B, target_h, A)
            x_t = x_t + dt * v_t
        return x_t

    def _build_ensemble_ls_matrix(self, factors, H, device, dtype):
        """Build the constraint matrix A and inverse-variance weight vector w
        for least-squares ensemble across heads of different time scales.

        For each compression factor f in `factors` (e.g. [2, 4]), we have
        16-step macro predictions where macro[i] = sum(env_step_actions[fi:fi+f]).
        Only the first H/f macros cover the first H env steps.

        Equations:
          a[i] = main[i]                              for i in [0, H)        (H rows, weight 1)
          sum(a[fi..fi+f]) = aux_f[i]                 for i in [0, H/f)      (H/f rows per f, weight 1/f)

        A: (H + sum(H//f), H), w: (H + sum(H//f),)
        """
        rows = [torch.eye(H, device=device, dtype=dtype)]
        weights = [torch.ones(H, device=device, dtype=dtype)]
        for f in factors:
            n_macros = H // f
            R = torch.zeros(n_macros, H, device=device, dtype=dtype)
            for i in range(n_macros):
                R[i, f * i : f * i + f] = 1.0
            rows.append(R)
            # Weight: 1/f because each macro is sum of f independent estimates,
            # so its variance is f * sigma^2 if individual actions have variance sigma^2.
            # WLS uses weight = 1/variance => 1/f.
            weights.append(torch.full((n_macros,), 1.0 / f, device=device, dtype=dtype))
        A = torch.cat(rows, dim=0)              # (M, H)
        w = torch.cat(weights, dim=0)           # (M,)
        return A, w

    def _ensemble_least_squares(self, main, aux_dict):
        """Combine multi-horizon predictions via weighted least squares.

        Args:
            main: (B, H, D) prediction from main head (each step = 1 env step delta).
            aux_dict: dict of {factor: pred_tensor} where each pred is (B, H, D)
                      and pred[i] represents sum of `factor` adjacent env-step deltas.

        Returns:
            (B, H, D) reconciled action sequence that best satisfies all heads
            in a weighted least-squares sense.
        """
        B, H, D = main.shape
        device = main.device
        in_dtype = main.dtype  # remember to cast back

        factors = sorted(aux_dict.keys())
        # Solve in float32 (torch.linalg.inv does not support bf16/fp16).
        # Disable autocast inside this block: otherwise the surrounding
        # bf16 autocast context will auto-downcast our fp32 matmuls back to bf16.
        ls_dtype = torch.float32
        with torch.amp.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                enabled=False):
            A, w = self._build_ensemble_ls_matrix(factors, H, device, ls_dtype)

            # Build target b: (B, M, D), cast to fp32
            b_parts = [main.to(ls_dtype)]
            for f in factors:
                n = H // f
                b_parts.append(aux_dict[f][:, :n, :].to(ls_dtype))
            b = torch.cat(b_parts, dim=1)  # (B, M, D)

            # Weighted least squares: a* = (A^T W A)^-1 A^T W b
            W = torch.diag(w)                          # (M, M)
            AtWA = (A.T @ W @ A).float()                # (H, H), force fp32
            AtWb = torch.einsum('ij,jk,bkd->bid', A.T, W, b).float()  # (B, H, D)
            AtWA_inv = torch.linalg.inv(AtWA)           # (H, H)
            a = torch.einsum('ij,bjd->bid', AtWA_inv, AtWb)   # (B, H, D)
            # Discrete-dim fix: LS averaging is meaningless for binary signals.
            # Overwrite those dims with main head's prediction (already valid binary).
            if getattr(self, "_ensemble_fix_discrete", False) and self.discrete_action_dims:
                disc = self.discrete_action_dims
                a[..., disc] = main.float()[..., disc]
        return a.to(in_dtype)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature,
                   head: str = "main") -> BatchFeature:
        """Run flow-matching denoising and return the predicted 16-step action.

        Args:
            head:
                - "main" (default): standard 16-step action_decoder.
                - "f<N>": auxiliary aux_action_decoders[f"f{N}"] (e.g. "f2", "f4").
                  Each step represents N original env-steps' worth of delta.
                  Only available when the model was trained with multi-horizon loss.
                - "ensemble": run main + all aux heads, then combine via weighted
                  least squares using the trained sum-equivalence between heads.

        Returns:
            BatchFeature with key "action_pred" of shape (B, 16, action_dim).
        """
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # === MoE inference: route to a single expert ===
        if head == "moe":
            assert self.use_moe_routing, "head='moe' requires use_moe_routing=True at training."

            # Per-quad mask routing (per_quad_mask): run main / m8 / m4 [+ n8]
            # forwards, router emits a (Q, C) per-quad pick (masked), chunk =
            # mix of slices. C ∈ {3, 4} depending on `moe_per_quad_use_n8`.
            if getattr(self, "moe_routing_mode", "single_pick") == "per_quad_mask":
                Q, C = self._moe_num_quads, self._moe_options_per_quad
                use_n8 = getattr(self, "_per_quad_use_n8", False)
                option_mask = getattr(self, "_per_quad_option_mask", None)
                router_in = self._moe_compute_router_inputs(vl_embeds, state_features)
                router_logits = self.head_router(router_in.float()).reshape(-1, Q, C)
                scaled = router_logits / max(self.moe_inference_temp, 1e-3)
                if use_n8 and option_mask is not None:
                    mask_dev = option_mask.to(scaled.device)[None].expand_as(scaled)
                    scaled = torch.where(mask_dev, scaled, torch.full_like(scaled, -1e9))
                probs = F.softmax(scaled, dim=-1)
                assert probs.shape[0] == 1, "per_quad_mask inference expects B=1."
                if self.moe_inference_stochastic:
                    flat = probs[0].reshape(Q, C)
                    picks = torch.stack([torch.multinomial(flat[k], 1).squeeze(-1) for k in range(Q)])
                else:
                    picks = probs[0].argmax(dim=-1)                                # (Q,)
                # Run main / m8 / m4 forwards. n8 only if active.
                main_act = self._denoise_with_decoder(vl_embeds, state_features, embodiment_id,
                                                     self.action_decoder, horizon=16)
                m8_act = self._denoise_with_decoder(vl_embeds, state_features, embodiment_id,
                                                    self.m8_action_decoder, horizon=8)
                m4_act = self._denoise_with_decoder(vl_embeds, state_features, embodiment_id,
                                                    self.m4_action_decoder, horizon=4)
                n8_act = None
                if use_n8:
                    n8_act = self._denoise_with_decoder(vl_embeds, state_features, embodiment_id,
                                                       self.n8_action_decoder, horizon=8)
                pieces = []
                for k in range(Q):
                    c = int(picks[k].item())
                    if c == 0:    # main 4 raw actions
                        pieces.append(main_act[:, 4 * k:4 * k + 4])
                    elif c == 1:  # m8 2 compressed actions
                        pieces.append(m8_act[:, 2 * k:2 * k + 2])
                    elif c == 2:  # m4 1 deep compressed action
                        pieces.append(m4_act[:, k:k + 1])
                    else:         # c == 3: n8 4 raw actions (k < 2 only by mask)
                        assert use_n8 and n8_act is not None and k < 2, \
                            f"n8 pick at invalid quad {k} (use_n8={use_n8})"
                        pieces.append(n8_act[:, 4 * k:4 * k + 4])
                actions = torch.cat(pieces, dim=1)                                 # (1, 4..16, D)
                return BatchFeature(data={
                    "action_pred": actions,
                    "moe_picked": picks.detach(),                                  # (Q,)
                    "moe_probs": probs[0].detach(),                                # (Q, C)
                })

            router_in = self._moe_compute_router_inputs(vl_embeds, state_features)
            router_logits = self.head_router(router_in.float())
            probs = F.softmax(router_logits / max(self.moe_inference_temp, 1e-3), dim=-1)
            if self.moe_inference_stochastic:
                pick = torch.multinomial(probs, num_samples=1).squeeze(-1)        # (B,)
            else:
                pick = probs.argmax(dim=-1)                                        # (B,)
            # Currently support B=1 inference (typical websocket request).
            assert pick.shape[0] == 1, "MoE inference path expects batch size 1 (websocket)."
            picked = int(pick[0].item())

            if self.moe_body_mode == "per_expert_h":
                decoders = [self.action_decoder, self.m8_action_decoder, self.m4_action_decoder]
                if self.moe_num_experts == 4:
                    decoders.append(self.n8_action_decoder)
                horizons = self.moe_expert_horizons
                actions = self._denoise_with_decoder(
                    vl_embeds, state_features, embodiment_id,
                    decoders[picked], horizon=horizons[picked],
                )
                return BatchFeature(data={
                    "action_pred": actions,
                    "moe_picked": torch.tensor([picked], device=actions.device),
                    "moe_probs": probs[0].detach(),
                })
            elif self.moe_body_mode == "shared_h16":
                # Body forward at h=H once. Picked expert decodes via cross-attn.
                # Flow matching denoise loop: x_t evolves in target_h space; body
                # input is reconstructed from x_t each step (pi0.5 m8 trick).
                actions = self._moe_shared_h16_denoise(
                    vl_embeds, state_features, embodiment_id, picked,
                )
                return BatchFeature(data={
                    "action_pred": actions,
                    "moe_picked": torch.tensor([picked], device=actions.device),
                    "moe_probs": probs[0].detach(),
                })
            else:
                raise ValueError(f"Unknown moe_body_mode: {self.moe_body_mode}")

        if head in ("ensemble", "ensemble_fix"):
            assert hasattr(self, "aux_action_decoders") and len(self.aux_action_decoders) > 0, (
                "Ensemble head requires aux_action_decoders to be populated."
            )
            # Run main + each aux head separately. Each gets its own random noise
            # / denoising trajectory (independent estimators).
            main = self._denoise_with_decoder(vl_embeds, state_features, embodiment_id,
                                              self.action_decoder)
            aux_dict = {}
            for key, dec in self.aux_action_decoders.items():
                factor = int(key[1:])
                aux_dict[factor] = self._denoise_with_decoder(
                    vl_embeds, state_features, embodiment_id, dec
                )
            # ensemble_fix: overwrite discrete dims with main's prediction
            self._ensemble_fix_discrete = (head == "ensemble_fix")
            actions = self._ensemble_least_squares(main, aux_dict)
            return BatchFeature(data={"action_pred": actions})

        # Single-head path
        if head == "m8":
            assert hasattr(self, "m8_action_decoder"), (
                "head='m8' requires use_merged_8_head=True at training time."
            )
            actions = self._denoise_with_decoder(
                vl_embeds, state_features, embodiment_id, self.m8_action_decoder, horizon=8,
            )
            return BatchFeature(data={"action_pred": actions})

        if head == "main_and_m8":
            # Returns BOTH main (16-step) and m8 (8-step) for client-side
            # dynamic head selection. Output dict keys:
            #   "action_pred"     -> main 16-step (default key, kept for the
            #                        validator + downstream interface)
            #   "action_pred_m8"  -> m8 8-step (extra key carried through unapply_transforms)
            assert hasattr(self, "m8_action_decoder"), (
                "head='main_and_m8' requires use_merged_8_head=True at training time."
            )
            main_act = self._denoise_with_decoder(
                vl_embeds, state_features, embodiment_id, self.action_decoder,
            )
            m8_act = self._denoise_with_decoder(
                vl_embeds, state_features, embodiment_id, self.m8_action_decoder, horizon=8,
            )
            return BatchFeature(data={"action_pred": main_act, "action_pred_m8": m8_act})

        if head == "moe_selective":
            # MoE router-gated selective: run router first; if main picked,
            # return N main samples + m8 (for AAC compression). If non-main
            # expert picked, return that expert's chunk as-is (no compression).
            assert self.use_moe_routing, "head='moe_selective' requires MoE training."
            assert hasattr(self, "m8_action_decoder"), "moe_selective also needs m8 head."
            assert vl_embeds.shape[0] == 1, "moe_selective expects B=1."
            router_in = self._moe_compute_router_inputs(vl_embeds, state_features)
            router_logits = self.head_router(router_in.float())
            probs = F.softmax(router_logits / max(self.moe_inference_temp, 1e-3), dim=-1)
            pick = probs.argmax(dim=-1)            # (1,)
            picked = int(pick[0].item())
            if self.moe_num_experts == 4 and picked == 3:
                # n8 picked: ALSO compute m4 (compressed sum-pair over same 8 env
                # steps) so client can do n8↔m4 self_agree compression decision.
                # Piggyback m4 on the action_pred_m8 key (client uses moe_picked
                # to know it's actually m4).
                n8_act = self._denoise_with_decoder(
                    vl_embeds, state_features, embodiment_id,
                    self.n8_action_decoder, horizon=8,
                )
                m4_act = self._denoise_with_decoder(
                    vl_embeds, state_features, embodiment_id,
                    self.m4_action_decoder, horizon=4,
                )
                return BatchFeature(data={
                    "action_pred": n8_act,
                    "action_pred_m8": m4_act,        # piggyback (client knows from picked)
                    "moe_picked": pick,
                })
            if picked != 0:
                # m8 (1) or m4 (2): already compressed, no further compression.
                if self.moe_body_mode == "per_expert_h":
                    decoders = [self.action_decoder, self.m8_action_decoder, self.m4_action_decoder]
                    if self.moe_num_experts == 4:
                        decoders.append(self.n8_action_decoder)
                    horizons = self.moe_expert_horizons
                    expert_chunk = self._denoise_with_decoder(
                        vl_embeds, state_features, embodiment_id,
                        decoders[picked], horizon=horizons[picked],
                    )
                elif self.moe_body_mode == "shared_h16":
                    expert_chunk = self._moe_shared_h16_denoise(
                        vl_embeds, state_features, embodiment_id, picked,
                    )
                else:
                    raise ValueError(f"Unknown moe_body_mode: {self.moe_body_mode}")
                return BatchFeature(data={
                    "action_pred": expert_chunk,
                    "moe_picked": pick,
                })
            # Main picked: produce N main samples + m8 (selective compression inputs).
            N = int(getattr(self, "_n_samples_at_inference", 1))
            if N <= 1:
                main_act = self._denoise_with_decoder(
                    vl_embeds, state_features, embodiment_id, self.action_decoder,
                )
            else:
                vl_rep = vl_embeds.repeat(N, 1, 1)
                state_rep = state_features.repeat(N, 1, 1)
                emb_rep = embodiment_id.repeat(N) if embodiment_id is not None else None
                main_act = self._denoise_with_decoder(
                    vl_rep, state_rep, emb_rep, self.action_decoder,
                )
            m8_act = self._denoise_with_decoder(
                vl_embeds, state_features, embodiment_id, self.m8_action_decoder, horizon=8,
            )
            return BatchFeature(data={
                "action_pred": main_act,
                "action_pred_m8": m8_act,
                "moe_picked": pick,
            })

        if head == "selective":
            # Returns N main 16-step samples (different noise) + 1 m8 8-step.
            # N is read from `self._n_samples_at_inference` (set by inference_service
            # at server startup; default 1 → equivalent to main_and_m8).
            # Shapes:
            #   action_pred     -> (N, 16, action_dim)  — sample[0] used for execution,
            #                      all N used for per-pair variance (entropy score)
            #   action_pred_m8  -> (1, 8, action_dim)
            # Note: B=1 inference assumed (single websocket request); we replicate
            # input N times along batch dim before the batched denoise.
            assert hasattr(self, "m8_action_decoder"), (
                "head='selective' requires use_merged_8_head=True at training time."
            )
            assert vl_embeds.shape[0] == 1, "head=selective expects B=1 inference."
            N = int(getattr(self, "_n_samples_at_inference", 1))
            if N <= 1:
                # Fast path: equivalent to main_and_m8.
                main_act = self._denoise_with_decoder(
                    vl_embeds, state_features, embodiment_id, self.action_decoder,
                )
            else:
                vl_rep = vl_embeds.repeat(N, 1, 1)
                state_rep = state_features.repeat(N, 1, 1)
                emb_rep = embodiment_id.repeat(N) if embodiment_id is not None else None
                main_act = self._denoise_with_decoder(
                    vl_rep, state_rep, emb_rep, self.action_decoder,
                )
            m8_act = self._denoise_with_decoder(
                vl_embeds, state_features, embodiment_id, self.m8_action_decoder, horizon=8,
            )
            return BatchFeature(data={"action_pred": main_act, "action_pred_m8": m8_act})

        if head == "m8_refined":
            assert hasattr(self, "m8_action_decoder") and hasattr(self, "m8_refinement"), (
                "head='m8_refined' requires use_merged_8_head=True AND use_m8_refinement=True."
            )
            actions = self._denoise_with_decoder(
                vl_embeds, state_features, embodiment_id, self.m8_action_decoder, horizon=8,
            )
            # Apply residual refinement: delta = MLP([vl_pooled, flat_action])
            vl_pooled = vl_embeds.mean(dim=1)
            refine_in = torch.cat([vl_pooled.float(),
                                   actions.flatten(1).float()], dim=-1)
            delta = self.m8_refinement(refine_in).view_as(actions)
            actions = actions + delta
            return BatchFeature(data={"action_pred": actions})

        if head != "main":
            assert hasattr(self, "aux_action_decoders") and head in self.aux_action_decoders, (
                f"Aux head '{head}' not found. Available: "
                f"{list(self.aux_action_decoders.keys()) if hasattr(self, 'aux_action_decoders') else []}"
            )
            decoder = self.aux_action_decoders[head]
        else:
            decoder = self.action_decoder

        actions = self._denoise_with_decoder(vl_embeds, state_features, embodiment_id, decoder)
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
