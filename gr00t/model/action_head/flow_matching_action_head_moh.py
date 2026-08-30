# SPDX-License-Identifier: Apache-2.0
"""Mixture of Horizons (MoH) action head for GR00T.


Key idea:
    A single (weight-shared) flow-matching DiT body is run on H different action
    horizon views in parallel by stacking them along the batch dimension. A small
    gating network produces per-(step, horizon) weights to fuse the H velocity
    predictions; the gate is trained with auxiliary + load-balance losses. At
    inference, the fused velocity drives the Euler denoising loop; optionally,
    a dynamic-replanning rule decides how many of the predicted steps to
    actually execute based on L1 disagreement across horizons.

This file is intentionally isolated from any other auxiliary-loss method that
lives in the sibling `flow_matching_action_head.py` (multi-horizon
compression, m8/m4/n8 heads, MoE routing, ensemble consistency, ...). Only the
GR00T base building blocks (`CategorySpecificMLP`, `CategorySpecificLinear`,
`MultiEmbodimentActionEncoder`, `DiT`, `SelfAttentionTransformer`) are reused.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


@dataclass
class FlowmatchingActionHeadMoHConfig(PretrainedConfig):
    """Config for the MoH action head.

    Base GR00T fields mirror `FlowmatchingActionHeadConfig`; MoH-specific fields
    are at the bottom.
    """

    # --- GR00T base fields (copied verbatim) ---
    add_pos_embed: bool = field(default=True)
    model_dtype: str = field(default="float32")
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=1536)
    backbone_embedding_dim: int = field(default=1536)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    max_state_dim: int = field(default=64)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=None)
    max_num_embodiments: int = field(default=32)
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)
    use_vlln: bool = field(default=True)
    vl_self_attention_cfg: dict = field(default=None)

    # --- MoH-specific fields ---
    horizons: list = field(default_factory=lambda: [4, 8, 12, 16])
    aux_weight: float = field(default=1.0)
    balance_weight: float = field(default=0.001)
    use_gate_noise: bool = field(default=True)
    mean_fusion: bool = field(default=False)
    use_dynamic_replanning: bool = field(default=False)
    scale_ratio: float = field(default=1.0)
    min_replan_steps: int = field(default=5)
    min_active_horizons: int = field(default=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class FlowmatchingActionHeadMoH(nn.Module):
    """MoH action head.

    Shape conventions:
        B   : batch size
        H   : number of horizon views (=len(horizons))
        max_h: action_horizon (e.g. 16)
        h_i : i-th horizon (e.g. 4, 8, 12, 16)
        S   : number of state tokens (typically 1)
        A   : action_dim
        D   : hidden_size
        T   : S + max_h (full token sequence into DiT)

    Batched-H ordering is always **interleave**: row `b * H + i` of any batched
    tensor corresponds to (sample b, horizon view i). This lets us reshape
    `(B*H, ...)` outputs to `(B, H, ...)` by a contiguous view.
    """

    config_class = FlowmatchingActionHeadMoHConfig
    supports_gradient_checkpointing = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def __init__(self, config: FlowmatchingActionHeadMoHConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        horizons = list(config.horizons)
        assert len(horizons) > 0, "horizons must be non-empty"
        assert sorted(horizons) == horizons, "horizons must be sorted ascending"
        assert max(horizons) <= config.action_horizon, (
            f"max horizon {max(horizons)} exceeds action_horizon {config.action_horizon}"
        )
        self.horizons = horizons
        self.num_horizons = len(horizons)

        # MoH gating knobs (mutable; can be overridden post-construction).
        self.aux_weight = float(config.aux_weight)
        self.balance_weight = float(config.balance_weight)
        self.use_gate_noise = bool(config.use_gate_noise)
        self.mean_fusion = bool(config.mean_fusion)

        # --- Core flow-matching modules (GR00T base) ---
        self.model = DiT(**config.diffusion_model_cfg)

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
            output_dim=config.action_dim,
        )

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

        self.beta_dist = Beta(
            torch.tensor(config.noise_beta_alpha, dtype=torch.float32),
            torch.tensor(config.noise_beta_beta, dtype=torch.float32),
        )
        self.num_timestep_buckets = config.num_timestep_buckets

        # --- MoH-specific modules ---
        self.gate_out_proj = nn.Linear(self.hidden_size, 1)
        nn.init.zeros_(self.gate_out_proj.weight)
        nn.init.zeros_(self.gate_out_proj.bias)
        if self.use_gate_noise:
            self.gate_noise_layer = nn.Linear(self.hidden_size, 1)
            nn.init.zeros_(self.gate_noise_layer.weight)
            nn.init.zeros_(self.gate_noise_layer.bias)
        self.softplus = nn.Softplus()

        # Cached masks (rebuilt lazily on the right device).
        self._valid_heads_mask_cache: Optional[torch.Tensor] = None

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
            self.gate_out_proj.requires_grad_(False)
            if self.use_gate_noise:
                self.gate_noise_layer.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune MoH action head projector: {self.tune_projector}")
        print(f"Tune MoH action head diffusion model: {self.tune_diffusion_model}")

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                self.gate_out_proj.eval()
                if self.use_gate_noise:
                    self.gate_noise_layer.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    # ------------------------------------------------------------------
    # Sampling utils (GR00T conventions)
    # ------------------------------------------------------------------
    def sample_time(self, batch_size, device, dtype):
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

    # ------------------------------------------------------------------
    # MoH helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cv_squared(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Coefficient-of-variation squared: var / (mean^2 + eps).

        Used in the load-balance loss to push the per-expert average gate
        probability within a segment toward uniform.
        """
        if x.numel() <= 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x_f = x.float()
        return x_f.var() / (x_f.mean() ** 2 + eps)

    def _build_valid_heads_mask(self, device) -> torch.Tensor:
        """(max_h, num_h) bool: True where step < horizons[i] (head i is valid at step)."""
        if (
            self._valid_heads_mask_cache is not None
            and self._valid_heads_mask_cache.device == device
        ):
            return self._valid_heads_mask_cache
        steps = torch.arange(self.action_horizon, device=device)[:, None]  # (max_h, 1)
        hs = torch.tensor(self.horizons, device=device)[None, :]            # (1, num_h)
        mask = steps < hs                                                    # (max_h, num_h)
        self._valid_heads_mask_cache = mask
        return mask

    def _compute_load_balance_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Encourages even gate usage across experts in each horizon-segment.

        gate_weights: (B, max_h, num_h) probabilities.
        For each segment [boundaries[i], boundaries[i+1]) where boundaries are
        the sorted union of {0} and self.horizons, compute the average gate
        probability over active experts (experts whose horizon > segment start)
        and apply cv_squared. The final loss is the mean over segments with >1
        active expert.
        """
        boundaries = sorted(set([0] + list(self.horizons)))
        components = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            active = [j for j, h in enumerate(self.horizons) if h > start]
            if len(active) <= 1:
                continue
            seg = gate_weights[:, start:end, :]              # (B, seg_len, num_h)
            seg_active = seg[:, :, active]                   # (B, seg_len, n_active)
            avg_prob = seg_active.mean(dim=(0, 1))           # (n_active,)
            components.append(self._cv_squared(avg_prob))
        if not components:
            return gate_weights.new_zeros(())
        return torch.stack(components).mean()

    def _build_self_attn_mask(
        self, B: int, state_len: int, max_h: int, device
    ) -> torch.Tensor:
        """Build a key-side self-attention mask for the batched-H DiT call.

        Returns a 2D mask of shape (B * num_h, T) where T = state_len + max_h.
        Diffusers' Attention treats this as a key padding mask: True positions
        are kept, False positions are masked out from the attention softmax.

        For each (sample b, horizon view i):
          - state tokens (positions [0, state_len)) are always valid keys.
          - action tokens (positions [state_len, state_len + h_i)) are valid.
          - action tokens (positions [state_len + h_i, state_len + max_h))
            are padded zeros and must be masked out.
        """
        T = state_len + max_h
        num_h = self.num_horizons
        mask = torch.zeros((B * num_h, T), dtype=torch.bool, device=device)
        mask[:, :state_len] = True
        for i, h_i in enumerate(self.horizons):
            rows = torch.arange(i, B * num_h, num_h, device=device)
            cols = state_len + torch.arange(h_i, device=device)
            mask[rows[:, None], cols[None, :]] = True
        return mask

    # ------------------------------------------------------------------
    # DiT forward (reimplemented to thread a self-attention mask)
    # ------------------------------------------------------------------
    def _dit_forward(
        self,
        hidden_states: torch.Tensor,             # (B*H, T, D)
        encoder_hidden_states: torch.Tensor,     # (B*H, n_vl, D_vl)
        timestep: torch.Tensor,                  # (B*H,)
        self_attn_key_mask: Optional[torch.Tensor] = None,  # (B*H, T) bool, True=keep
    ) -> torch.Tensor:
        """Run the underlying DiT but pass a self-attention mask in the odd
        (self-attention) blocks. Mirrors `DiT.forward` line-for-line except for
        the `attention_mask` arg in the self-attn branch.

        The diffusers `Attention` module accepts a key-padding mask of shape
        (batch, key_len); we convert bool → additive float (-large at False).
        """
        dit = self.model
        temb = dit.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        if self_attn_key_mask is not None:
            # Diffusers Attention.prepare_attention_mask expects additive mask.
            sa_add = torch.where(
                self_attn_key_mask,
                torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device),
                torch.full((), -1e9, dtype=hidden_states.dtype, device=hidden_states.device),
            )
        else:
            sa_add = None

        for idx, block in enumerate(dit.transformer_blocks):
            if idx % 2 == 1 and dit.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=sa_add,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )

        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = dit.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return dit.proj_out_2(hidden_states)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embeds = backbone_output.backbone_features                 # (B, n_vl, D_vl)
        vl_attn_mask = backbone_output.backbone_attention_mask        # (B, n_vl)
        embodiment_id = action_input.embodiment_id                    # (B,)
        state_features = self.state_encoder(action_input.state, embodiment_id)  # (B, S, D)
        actions = action_input.action                                  # (B, max_h, A)
        action_mask = action_input.action_mask                         # (B, max_h, A)
        device = vl_embeds.device

        B = actions.shape[0]
        max_h = self.action_horizon
        num_h = self.num_horizons
        A = self.action_dim
        S = state_features.shape[1]

        # --- Shared flow-matching noise & time (one sample shared across H views) ---
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)  # (B, max_h, A)
        t = self.sample_time(B, device, actions.dtype)                          # (B,) in [0, 1)
        t_b11 = t[:, None, None]
        noisy_full = (1 - t_b11) * noise + t_b11 * actions                      # GR00T convention
        velocity_target = actions - noise                                        # clean - noise
        t_disc = (t * self.num_timestep_buckets).long()                         # (B,)

        # --- Batched-H expansion (interleave order) ---
        batched_state_feat = state_features.repeat_interleave(num_h, dim=0)     # (B*H, S, D)
        batched_vl = vl_embeds.repeat_interleave(num_h, dim=0)                  # (B*H, n_vl, D_vl)
        # batched_vl_mask currently unused (DiT ignores encoder_attention_mask),
        # but we keep the repeat for future-proofing / parity with π₀-MoH.
        _batched_vl_mask = vl_attn_mask.repeat_interleave(num_h, dim=0)
        batched_emb_id = embodiment_id.repeat_interleave(num_h, dim=0)          # (B*H,)
        batched_t_disc = t_disc.repeat_interleave(num_h, dim=0)                 # (B*H,)

        # Per-horizon padded noisy actions: [0, h_i) carries real values; rest zero.
        padded = [
            F.pad(noisy_full[:, :h_i, :], (0, 0, 0, max_h - h_i)) for h_i in self.horizons
        ]
        batched_x_t = torch.stack(padded, dim=1).reshape(B * num_h, max_h, A)    # interleave

        # --- Encode action features ---
        action_features = self.action_encoder(batched_x_t, batched_t_disc, batched_emb_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(max_h, dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat([batched_state_feat, action_features], dim=1)        # (B*H, S+max_h, D)
        # Ensure the input dtype matches the DiT body's compute dtype.
        sa_embs = sa_embs.to(dtype=batched_vl.dtype)

        # --- DiT forward with self-attention mask hiding pad action keys ---
        self_attn_key_mask = self._build_self_attn_mask(B, S, max_h, device)
        model_output = self._dit_forward(sa_embs, batched_vl, batched_t_disc, self_attn_key_mask)
        # (B*H, S+max_h, hidden_size)
        action_out = model_output[:, -max_h:]                                    # (B*H, max_h, D)

        # --- Per-horizon velocity predictions ---
        pred_full = self.action_decoder(action_out, batched_emb_id)              # (B*H, max_h, A)
        all_v = pred_full.view(B, num_h, max_h, A)

        # --- Individual loss (GR00T-style mask MSE) ---
        per_head_losses = []
        for i, h_i in enumerate(self.horizons):
            pred_i = all_v[:, i, :h_i, :]                                        # (B, h_i, A)
            target_i = velocity_target[:, :h_i, :]
            mask_i = action_mask[:, :h_i, :].to(pred_i.dtype)
            per_dim = F.mse_loss(pred_i, target_i, reduction="none") * mask_i
            denom = mask_i.sum().clamp(min=1.0)
            per_head_losses.append(per_dim.sum() / denom)
        individual_loss = torch.stack(per_head_losses).sum()

        # --- Gate logits (with optional learned noise during training) ---
        if self.mean_fusion:
            gate_logits_raw = action_out.new_ones((B, max_h, num_h))
        else:
            logits = self.gate_out_proj(action_out.float())                      # (B*H, max_h, 1)
            if self.training and self.use_gate_noise:
                raw_std = self.gate_noise_layer(action_out.float())
                noise_std = self.softplus(raw_std) + 1e-2
                logits = logits + torch.randn_like(logits) * noise_std
            gate_logits_raw = logits.reshape(B, num_h, max_h).permute(0, 2, 1)   # (B, max_h, num_h)

        valid_heads_mask = self._build_valid_heads_mask(device).unsqueeze(0)     # (1, max_h, num_h)
        gate_logits = torch.where(
            valid_heads_mask,
            gate_logits_raw,
            torch.full_like(gate_logits_raw, torch.finfo(gate_logits_raw.dtype).min),
        )
        gate_weights = F.softmax(gate_logits, dim=-1)                            # (B, max_h, num_h)

        # --- Auxiliary loss: fused velocity vs full target ---
        v_fused = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v.float()).sum(dim=1)
        # action_mask in original dtype; cast to float for the aux loss accumulation
        per_dim_aux = F.mse_loss(v_fused, velocity_target.float(), reduction="none") * action_mask.float()
        denom_aux = action_mask.float().sum().clamp(min=1.0)
        auxiliary_loss = per_dim_aux.sum() / denom_aux

        # --- Load-balance loss ---
        load_balance_loss = self._compute_load_balance_loss(gate_weights)

        total_loss = (
            individual_loss
            + self.aux_weight * auxiliary_loss
            + self.balance_weight * load_balance_loss
        )

        return BatchFeature(data={
            "loss": total_loss,
            "individual_loss": individual_loss.detach(),
            "aux_loss": auxiliary_loss.detach(),
            "balance_loss": load_balance_loss.detach(),
        })

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        use_dynamic_replanning: Optional[bool] = None,
        scale_ratio: Optional[float] = None,
        min_replan_steps: Optional[int] = None,
        min_active_horizons: Optional[int] = None,
        ret_weights: bool = False,
    ) -> BatchFeature:
        use_dyn = self.config.use_dynamic_replanning if use_dynamic_replanning is None else use_dynamic_replanning
        scl = self.config.scale_ratio if scale_ratio is None else scale_ratio
        min_rep = self.config.min_replan_steps if min_replan_steps is None else min_replan_steps
        min_act = self.config.min_active_horizons if min_active_horizons is None else min_active_horizons

        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)

        B = vl_embeds.shape[0]
        device = vl_embeds.device
        max_h = self.action_horizon
        num_h = self.num_horizons
        A = self.action_dim
        S = state_features.shape[1]

        if use_dyn:
            assert B == 1, "Dynamic replanning supports batch size 1 only."

        # Pre-batch (constant across denoising steps).
        batched_state_feat = state_features.repeat_interleave(num_h, dim=0)
        batched_vl = vl_embeds.repeat_interleave(num_h, dim=0)
        _batched_vl_mask = vl_attn_mask.repeat_interleave(num_h, dim=0)
        batched_emb_id = embodiment_id.repeat_interleave(num_h, dim=0)
        self_attn_key_mask = self._build_self_attn_mask(B, S, max_h, device)

        # Start from N(0, I) noise; integrate t: 0 → 1 with dt = +1/num_steps.
        x_t = torch.randn((B, max_h, A), device=device, dtype=vl_embeds.dtype)
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        l1_disagreement_sum = (
            torch.zeros((B, max_h), device=device, dtype=torch.float32) if use_dyn else None
        )
        gate_log = [] if ret_weights else None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_disc_scalar = int(t_cont * self.num_timestep_buckets)
            t_disc = torch.full((B * num_h,), t_disc_scalar, device=device, dtype=torch.long)

            padded = [F.pad(x_t[:, :h, :], (0, 0, 0, max_h - h)) for h in self.horizons]
            batched_x_t = torch.stack(padded, dim=1).reshape(B * num_h, max_h, A)

            action_features = self.action_encoder(batched_x_t, t_disc, batched_emb_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(max_h, dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat([batched_state_feat, action_features], dim=1).to(dtype=batched_vl.dtype)

            model_output = self._dit_forward(sa_embs, batched_vl, t_disc, self_attn_key_mask)
            action_out = model_output[:, -max_h:]

            pred_full = self.action_decoder(action_out, batched_emb_id)
            all_v = pred_full.view(B, num_h, max_h, A)

            if self.mean_fusion:
                gate_logits_raw = action_out.new_ones((B, max_h, num_h))
            else:
                logits = self.gate_out_proj(action_out.float())
                gate_logits_raw = logits.reshape(B, num_h, max_h).permute(0, 2, 1)

            valid_heads_mask = self._build_valid_heads_mask(device).unsqueeze(0)
            gate_logits = torch.where(
                valid_heads_mask,
                gate_logits_raw,
                torch.full_like(gate_logits_raw, torch.finfo(gate_logits_raw.dtype).min),
            )
            gate_weights = F.softmax(gate_logits, dim=-1)                        # (B, max_h, num_h)

            if ret_weights:
                gate_log.append(torch.round(gate_weights, decimals=3))

            v_fused = (gate_weights.permute(0, 2, 1).unsqueeze(-1) * all_v.float()).sum(dim=1)

            if use_dyn:
                fused_iter = x_t.float() + dt * v_fused                           # (B, max_h, A)
                indiv_iter = x_t.float().unsqueeze(1) + dt * all_v.float()        # (B, num_h, max_h, A)
                per_iter_l1 = (indiv_iter - fused_iter.unsqueeze(1)).abs().sum(-1)  # (B, num_h, max_h)
                per_iter_l1 = (per_iter_l1 * gate_weights.permute(0, 2, 1)).sum(dim=1)
                l1_disagreement_sum += per_iter_l1

            # Euler step (GR00T convention: x_{t+dt} = x_t + dt * velocity).
            x_t = (x_t.float() + dt * v_fused).to(x_t.dtype)

        result = {"action_pred": x_t}
        if ret_weights:
            result["gate_weights"] = torch.stack(gate_log, dim=1).detach().cpu()

        if use_dyn:
            dyn_steps = min_rep
            threshold = l1_disagreement_sum[0, :min_rep].mean() * scl
            for s in range(min_rep, max_h):
                active = sum(1 for h in self.horizons if s < h)
                if active <= min_act:
                    break
                if l1_disagreement_sum[0, s] < threshold:
                    dyn_steps += 1
                else:
                    break
            final_steps = max(min_rep, dyn_steps)
            result["action_pred"] = x_t[:, :final_steps, :]
            result["replan_steps"] = torch.tensor(final_steps, device=device)

        return BatchFeature(data=result)

    # ------------------------------------------------------------------
    # Public inference entry (matches GR00T base API)
    # ------------------------------------------------------------------
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        head: str = "main",
    ) -> BatchFeature:
        if head != "main":
            raise ValueError(
                f"FlowmatchingActionHeadMoH only supports head='main', got head={head!r}"
            )
        return self.sample_actions(backbone_output, action_input)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
