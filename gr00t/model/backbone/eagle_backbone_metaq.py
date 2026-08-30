# Eagle backbone with meta-query token injection. Forks EagleBackbone but
# replicates Eagle2_5_VLForConditionalGeneration.forward so we can splice
# learnable meta-query tokens into the language-model input. The forward
# returns both the original VL features and the meta-query features
# (both projected through eagle_linear); downstream modules (action head)
# use meta_q_features for the MoE router and vl_features for cross-attn.

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .eagle_backbone import EagleBackbone


class EagleBackboneMetaQ(EagleBackbone):
    """Eagle backbone with N learnable meta-query tokens appended to the LM
    input.  Meta-query positions are processed jointly with vision/text by
    the language model, so their output features are conditioned on the full
    multimodal context. The action head consumes them directly for the MoE
    router input (replacing the mean-pool over vl_embeds).
    """

    def __init__(self, *args, n_meta_q: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_meta_q = int(n_meta_q)
        # Meta-query embeddings live in the LM hidden dim (pre eagle_linear)
        # so they get projected together with the rest of the LM output.
        lm_hidden = self.eagle_model.language_model.config.hidden_size
        # Small init (~Transformer embedding init scale).
        self.meta_q_emb = nn.Parameter(torch.randn(self.n_meta_q, lm_hidden) * 0.02)

    def forward_eagle(self, vl_input: BatchFeature):
        """Replicate Eagle2_5_VLForConditionalGeneration.forward up through
        the language model call but inject `self.n_meta_q` learnable tokens
        at the end of the embed sequence. Returns
        ``(vl_features, vl_attn_mask, meta_q_features)`` where vl_features
        and meta_q_features are both projected by ``self.eagle_linear``.
        """
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        em = self.eagle_model
        input_ids = eagle_input["input_ids"]
        pixel_values = eagle_input["pixel_values"]
        attn_mask = eagle_input["attention_mask"]
        position_ids = eagle_input.get("position_ids", None)
        image_flags = eagle_input.get("image_flags", None)

        # ---- Replicate the input-embed construction from Eagle's forward ----
        input_embeds = em.language_model.get_input_embeddings()(input_ids)
        vit_embeds = em.extract_feature(pixel_values)
        if image_flags is not None:
            image_flags = image_flags.view(-1)
            vit_embeds = vit_embeds[image_flags == 1]
        B, T, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(B * T, D)
        flat_ids = input_ids.reshape(B * T)
        selected = flat_ids == em.image_token_index
        try:
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, D)
        except Exception as e:
            vit_flat = vit_embeds.reshape(-1, D)
            n_token = int(selected.sum().item())
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_flat[:n_token]
        input_embeds = flat_embeds.reshape(B, T, D)

        # ---- Inject meta-query tokens at the end ----
        # Be explicit about dtype + device — autocast may swap dtype around;
        # nn.Parameter device follows the model but be safe.
        meta_q = self.meta_q_emb[None].expand(B, -1, -1).to(
            dtype=input_embeds.dtype, device=input_embeds.device,
        )
        # Print first-call sanity stats once (helps debug NaN issues without
        # needing a debugger run); guarded so it doesn't spam training logs.
        if not getattr(self, "_meta_q_printed", False):
            with torch.no_grad():
                ie_max = float(input_embeds.detach().abs().max().item())
                mq_max = float(meta_q.detach().abs().max().item())
                ie_nan = bool(torch.isnan(input_embeds).any().item())
                mq_nan = bool(torch.isnan(meta_q).any().item())
                print(f"[metaq sanity] input_embeds: |max|={ie_max:.4g} nan={ie_nan}; "
                      f"meta_q: |max|={mq_max:.4g} nan={mq_nan}; "
                      f"shapes ie={tuple(input_embeds.shape)} mq={tuple(meta_q.shape)}")
            self._meta_q_printed = True
        full_embeds = torch.cat([input_embeds, meta_q], dim=1)               # (B, T+N, D_lang)
        meta_q_mask = torch.ones(B, self.n_meta_q,
                                 dtype=attn_mask.dtype, device=attn_mask.device)
        full_attn_mask = torch.cat([attn_mask, meta_q_mask], dim=1)
        if position_ids is not None:
            # Continue the sequence; safe even with attention padding because
            # padded positions also have attention_mask=0 and don't influence
            # the output.
            last_pos = position_ids[:, -1:]
            offsets = torch.arange(1, self.n_meta_q + 1, device=position_ids.device)
            new_pos = last_pos + offsets[None].expand(B, -1)
            full_position_ids = torch.cat([position_ids, new_pos], dim=1)
        else:
            full_position_ids = None

        # ---- Language model forward ----
        lm_out = em.language_model(
            inputs_embeds=full_embeds,
            attention_mask=full_attn_mask,
            position_ids=full_position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = lm_out.hidden_states[self.select_layer]                    # (B, T+N, D_lang)
        projected = self.eagle_linear(hidden)                                # (B, T+N, D_vl)
        vl_features = projected[:, :T]                                       # (B, T, D_vl)
        meta_q_features = projected[:, T:]                                   # (B, N, D_vl)
        return vl_features, attn_mask, meta_q_features

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        vl_features, vl_mask, meta_q_features = self.forward_eagle(vl_input)
        return BatchFeature(data={
            "backbone_features": vl_features,
            "backbone_attention_mask": vl_mask,
            "meta_q_features": meta_q_features,
        })
