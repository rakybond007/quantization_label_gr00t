"""End-to-end smoke test: load real RoboCasa dataset with the multi-horizon
DataConfig, fetch a sample, and verify the transformed batch has both
`action` (16 steps) and `action_extended` (64 steps) keys.
"""

import os
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP


DATA_PATH = "/sjw_alinlab2/home/myungkyu/shared/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"


def main():
    cfg = DATA_CONFIG_MAP["single_panda_gripper_multi_horizon"]
    modality_configs = cfg.modality_config()
    transforms = cfg.transform()

    dataset = LeRobotSingleDataset(
        dataset_path=DATA_PATH,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[100]
    print(f"\nSample keys ({len(sample)}):")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k:35s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:35s}  type={type(v).__name__}")

    # Sanity checks
    assert "action" in sample, "Missing 'action' key"
    assert "action_extended" in sample, "Missing 'action_extended' key (multi-horizon)"
    assert sample["action"].shape[0] == 16, f"action should be 16-step, got {sample['action'].shape}"
    assert sample["action_extended"].shape[0] == 64, (
        f"action_extended should be 64-step, got {sample['action_extended'].shape}"
    )
    # First 16 steps of extended must match standard action (channel-padded)
    a16 = sample["action"]
    a64 = sample["action_extended"]
    assert torch.allclose(torch.as_tensor(a16), torch.as_tensor(a64[:16])), (
        "First 16 steps of action_extended should equal action"
    )
    print("\n✓ All checks passed.")


if __name__ == "__main__":
    main()
