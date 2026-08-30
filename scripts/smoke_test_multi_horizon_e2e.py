"""End-to-end smoke test: load real data + run model forward with multi-horizon loss enabled.

This loads the actual GR00T-N1.5-3B pretrained checkpoint, attaches the
auxiliary multi-horizon decoders, and runs a forward pass on a real batch.
"""

import os
import torch
import torch.nn as nn

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.action_head.flow_matching_action_head import CategorySpecificMLP
from gr00t.model.transforms import DefaultDataCollator


DATA_PATH = "/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
MODEL_ID = "nvidia/GR00T-N1.5-3B"


# Use the project's actual data collator (handles eagle_content with PIL images, etc.)
collate = DefaultDataCollator()


def main():
    print("Loading dataset...")
    cfg = DATA_CONFIG_MAP["single_panda_gripper_multi_horizon"]
    dataset = LeRobotSingleDataset(
        dataset_path=DATA_PATH,
        modality_configs=cfg.modality_config(),
        transforms=cfg.transform(),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    samples = [dataset[i] for i in [100, 200]]
    batch = collate(samples)

    # Wrap eagle_content properly: it's a list of dicts → keep as-is for now
    print(f"Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}  {v.dtype}")

    print("\nLoading model...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        tune_llm=False, tune_visual=False,
        tune_projector=True, tune_diffusion_model=True,
        load_action_head=True,
        torch_dtype=torch.bfloat16,
    )
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(cuda_device)

    # Attach multi-horizon decoders (mirrors finetune script logic)
    ah = model.action_head
    ah.use_multi_horizon_loss = True
    ah.multi_horizon_factors = [2, 4]
    ah.multi_horizon_loss_weights = [1.0, 1.0]
    ah.multi_horizon_main_weight = 1.0
    ah.aux_action_decoders = nn.ModuleDict({
        f"f{f}": CategorySpecificMLP(
            num_categories=ah.config.max_num_embodiments,
            input_dim=ah.hidden_size,
            hidden_dim=ah.hidden_size,
            output_dim=ah.action_dim,
        )
        for f in ah.multi_horizon_factors
    })
    device = next(model.parameters()).device
    ah.aux_action_decoders.to(device=device, dtype=ah.dtype)
    print(f"Added {len(ah.multi_horizon_factors)} aux heads")

    model = model.to(device)
    print(f"Model on device: {device}")

    print("\nRunning forward...")
    model.train()
    out = model(batch)
    print(f"  loss = {out['loss'].item():.4f}")
    print(f"  loss_main = {out['loss_main'].item():.4f}")
    for k in out.keys():
        if k.startswith("loss_f"):
            print(f"  {k} = {out[k].item():.4f}")

    print("\nBackward...")
    out["loss"].backward()
    n_grad_aux = sum(p.grad.abs().sum().item() for p in ah.aux_action_decoders.parameters() if p.grad is not None)
    print(f"  aux decoders grad magnitude: {n_grad_aux:.2f}")
    assert n_grad_aux > 0, "Aux decoders should have gradient flow"

    print("\n✓ End-to-end multi-horizon forward+backward OK.")


if __name__ == "__main__":
    main()
