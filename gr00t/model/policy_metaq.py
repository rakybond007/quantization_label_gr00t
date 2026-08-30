# Meta-query fork of Gr00tPolicy.  Inherits everything except _load_model,
# which loads GR00T_N1_5_MetaQ (with EagleBackboneMetaQ + FlowmatchingActionHeadMetaQ)
# instead of the legacy GR00T_N1_5.

from gr00t.model.policy import Gr00tPolicy, COMPUTE_DTYPE
from gr00t.model.gr00t_n1_metaq import GR00T_N1_5_MetaQ


class Gr00tPolicyMetaQ(Gr00tPolicy):
    """Drop-in replacement for Gr00tPolicy that loads the meta-query variant
    of GR00T-N1.5.  The training/inference contract is identical (websocket
    server expects the same input/output schema).  The only difference is
    that the underlying model is GR00T_N1_5_MetaQ which appends N learnable
    meta-query tokens to the VLM input and routes the MoE on their LM
    output features instead of vl_embeds.mean()."""

    def _load_model(self, model_path):
        model = GR00T_N1_5_MetaQ.from_pretrained(
            model_path, torch_dtype=COMPUTE_DTYPE,
            backbone_model_type=self.backbone_model_type,
        )
        model.eval()
        model.to(device=self.device)
        self.model = model
