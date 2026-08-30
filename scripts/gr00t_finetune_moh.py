# SPDX-License-Identifier: Apache-2.0
"""GR00T-N1.5 fine-tuning with Mixture-of-Horizons (MoH) action head.

This script is the MoH counterpart of `scripts/gr00t_finetune.py`. It strips
every flag related to other auxiliary-loss / MoE methods (multi-horizon
compression, m8/m4/n8 heads, MoE routing, ensemble consistency, ...) and only
exposes MoH-specific knobs.

The model loaded is `GR00T_N1_5_MoH` from `gr00t.model.gr00t_n1_moh`. By default
the NVIDIA pretrained action head is transferred — only the MoH gate modules
(`gate_out_proj`, `gate_noise_layer`) start from scratch.

Note on installation: the `gr00t` Python package must point at THIS repo
(`GR00T-action-quantization-moh`). If you reuse a conda env that was previously
`pip install -e .`'d against a different gr00t checkout, re-run the editable
install inside this repo or pass PYTHONPATH=. when launching.
"""

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
from gr00t.model.gr00t_n1_moh import GR00T_N1_5_MoH, GR00T_N1_5_MoH_Config
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T-MoH fine-tuning."""

    # ---- Dataset ----
    dataset_path: List[str]
    """Path to the dataset directory (or directories for a mixture)."""

    output_dir: str = "/tmp/gr00t_moh"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "single_panda_gripper"
    """Data configuration name from DATA_CONFIG_MAP. Use a config that yields
    exactly `action_horizon` future steps — MoH does not need the extended
    horizon used by multi-horizon compression methods."""

    # ---- Training ----
    batch_size: int = 8
    """Per-GPU batch size. MoH expands batch by num_horizons (=4 by default)
    inside the action head, so memory cost is ~4x the base model's. Set
    accordingly; the default here is intentionally lower than base."""

    max_steps: int = 10000
    """Maximum number of optimizer steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training (uses torchrun for >1)."""

    save_steps: int = 1000
    """Number of steps between checkpoints."""

    # ---- Model ----
    base_model_path: str = None
    """Path or HF model ID for the base checkpoint (e.g. nvidia/GR00T-N1.5-3B).
    If None, a new MoH model is built from the default base config (rare; you
    almost always want to point this at the NVIDIA checkpoint)."""

    tune_llm: bool = False
    """Fine-tune the backbone LLM."""

    tune_visual: bool = False
    """Fine-tune the backbone vision tower."""

    tune_projector: bool = True
    """Fine-tune the action head's projector (state_encoder, action_encoder,
    action_decoder, position_embedding, gate modules)."""

    tune_diffusion_model: bool = True
    """Fine-tune the action head's DiT body."""

    load_action_head: bool = True
    """Whether to transfer the pretrained action-head weights from the
    checkpoint. Default True — only the MoH-specific gate modules are
    initialized from scratch. Set False to train the entire action head
    from scratch."""

    # ---- MoH-specific ----
    horizons: tuple[int, ...] = (4, 8, 12, 16)
    """Horizon set used by MoH. Each entry must be <= action_horizon (=16 by
    default). Sorted ascending."""

    aux_weight: float = 1.0
    """Weight on the auxiliary fused-velocity loss."""

    balance_weight: float = 0.001
    """Weight on the load-balance loss (cv² over per-expert avg gate prob)."""

    use_gate_noise: bool = True
    """If True, add learned-stddev Gaussian noise to gate logits during
    training (standard noisy-topk gating; off during eval)."""

    mean_fusion: bool = False
    """If True, force the gate to a uniform distribution at every (step,
    horizon). Useful as a sanity / ablation baseline."""

    use_dynamic_replanning: bool = False
    """Enable L1-disagreement-driven early termination at inference. Cannot
    be used during training; this flag is only persisted into the saved
    config so `get_action` picks it up after loading the checkpoint."""

    scale_ratio: float = 1.0
    """Threshold multiplier (vs mean disagreement over the first
    `min_replan_steps`) for dynamic replanning."""

    min_replan_steps: int = 5
    """Always commit at least this many steps before considering early stop."""

    min_active_horizons: int = 1
    """Stop extending dynamic execution once <= this many horizons remain
    active. Default 1 means we keep going as long as ANY horizon still
    covers the next step."""

    # ---- Optim ----
    learning_rate: float = 1e-4
    """Learning rate for AdamW."""

    weight_decay: float = 1e-5
    """AdamW weight decay."""

    warmup_ratio: float = 0.05
    """Warmup fraction of total training steps."""

    save_total_limit: int = 6
    """Max number of checkpoints to keep on disk."""

    resume: bool = False
    """Resume from checkpoint at output_dir."""

    # ---- LoRA ----
    lora_rank: int = 0
    """LoRA rank (0 disables LoRA)."""

    lora_alpha: int = 16
    """LoRA alpha."""

    lora_dropout: float = 0.1
    """LoRA dropout."""

    lora_full_model: bool = False
    """If True, LoRA wraps the whole model; else only the action head."""

    # ---- Data loading ----
    dataloader_num_workers: int = 8
    """Workers per dataloader."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag (e.g. 'new_embodiment', 'gr1')."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend."""

    balance_dataset_weights: bool = True
    """Mixture-dataset: weight by trajectory count."""

    balance_trajectory_weights: bool = True
    """Mixture-dataset: weight trajectories within a dataset by length."""

    # ---- Logging ----
    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """Where to report metrics."""

    run_name: str = None
    """Run name for the logger."""

    # ---- Backbone ----
    backbone_model_type: Literal["eagle", "qwen2_5_vl", "paligemma", "qwen2_5_vl_7b"] = "eagle"
    """Backbone family."""

    backbone_select_layer: int = 12
    """Hidden-state layer index pulled out of the backbone."""

    backbone_path: str = None
    """Backbone weights path (None uses the family's default)."""


#####################################################################################
# Main
#####################################################################################


def main(config: ArgsConfig):
    # -------- Step 1: dataset --------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    if config.backbone_model_type != "eagle":
        transforms = data_config_cls.transform(
            config.backbone_model_type, backbone_path=config.backbone_path
        )
    else:
        transforms = data_config_cls.transform(config.backbone_model_type)

    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            single_datasets.append(LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            ))
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in single_datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        print(f"[MoH] Loaded {len(single_datasets)} datasets from {config.dataset_path}")

    # -------- Step 2: model --------
    # Validate horizons up front (clearer error than waiting for the head)
    horizons = list(config.horizons)
    assert sorted(horizons) == horizons, "horizons must be sorted ascending"
    assert len(horizons) > 0, "horizons must be non-empty"

    moh_action_head_kwargs = dict(
        horizons=horizons,
        aux_weight=config.aux_weight,
        balance_weight=config.balance_weight,
        use_gate_noise=config.use_gate_noise,
        mean_fusion=config.mean_fusion,
        use_dynamic_replanning=config.use_dynamic_replanning,
        scale_ratio=config.scale_ratio,
        min_replan_steps=config.min_replan_steps,
        min_active_horizons=config.min_active_horizons,
    )

    if config.base_model_path is not None:
        model = GR00T_N1_5_MoH.from_pretrained(
            pretrained_model_name_or_path=config.base_model_path,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            load_action_head=config.load_action_head,
            action_head_kwargs=moh_action_head_kwargs,
        )
        # Make sure all horizons are persisted on top-level config (so the
        # next from_pretrained from a saved checkpoint roundtrips).
        for k, v in moh_action_head_kwargs.items():
            model.config.action_head_cfg[k] = v
    else:
        # Build from the NVIDIA base config but swap into MoH config space.
        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained("nvidia/GR00T-N1.5-3B")
        cfg_dict = base_cfg.to_dict()
        cfg_dict.pop("model_type", None)
        cfg_dict.pop("architectures", None)
        ah_cfg = dict(cfg_dict.get("action_head_cfg", {}))
        ah_cfg.update(moh_action_head_kwargs)
        cfg_dict["action_head_cfg"] = ah_cfg

        # Backbone family wiring (mirrors base script)
        if config.backbone_model_type == "eagle":
            print("[MoH] Using Eagle backbone")
            cfg_dict["backbone_model_type"] = config.backbone_model_type
            cfg_dict["backbone_cfg"]["select_layer"] = config.backbone_select_layer
            if config.backbone_path is not None:
                cfg_dict["backbone_cfg"]["eagle_path"] = config.backbone_path
        elif config.backbone_model_type == "qwen2_5_vl":
            print("[MoH] Using Qwen2.5-VL backbone")
            cfg_dict["backbone_model_type"] = config.backbone_model_type
            cfg_dict["backbone_cfg"].pop("eagle_path", None)
            cfg_dict["backbone_cfg"]["select_layer"] = config.backbone_select_layer
            if config.backbone_path is not None:
                cfg_dict["backbone_cfg"]["qwen_path"] = config.backbone_path
        elif config.backbone_model_type == "paligemma":
            print("[MoH] Using Paligemma backbone")
            cfg_dict["backbone_model_type"] = config.backbone_model_type
            cfg_dict["backbone_cfg"].pop("eagle_path", None)
            cfg_dict["backbone_cfg"]["select_layer"] = config.backbone_select_layer
            if config.backbone_path is not None:
                cfg_dict["backbone_cfg"]["paligemma_path"] = config.backbone_path
        elif config.backbone_model_type == "qwen2_5_vl_7b":
            print("[MoH] Using Qwen2.5-VL-7B backbone")
            cfg_dict["backbone_model_type"] = config.backbone_model_type
            cfg_dict["backbone_cfg"].pop("eagle_path", None)
            cfg_dict["backbone_cfg"]["select_layer"] = config.backbone_select_layer
            cfg_dict["backbone_cfg"]["qwen_path"] = (
                config.backbone_path or "Qwen/Qwen2.5-VL-7B-Instruct"
            )
            cfg_dict["action_head_cfg"]["diffusion_model_cfg"]["cross_attention_dim"] = 3584
            cfg_dict["action_head_cfg"]["backbone_embedding_dim"] = 3584
            cfg_dict["action_head_cfg"]["vl_self_attention_cfg"]["num_attention_heads"] = 56
            cfg_dict["action_head_cfg"]["vl_self_attention_cfg"]["attention_head_dim"] = 64

        moh_config = GR00T_N1_5_MoH_Config(**cfg_dict)
        moh_config.tune_llm = config.tune_llm
        moh_config.tune_visual = config.tune_visual
        moh_config.tune_projector = config.tune_projector
        moh_config.tune_diffusion_model = config.tune_diffusion_model

        model = GR00T_N1_5_MoH.from_config(
            config=moh_config,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )

    # bf16 compute
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # Mirror the MoH knobs onto the live action_head as well (config is the
    # source of truth, but having the attrs on the head is convenient for
    # inference-time overrides).
    ah = model.action_head
    ah.aux_weight = float(config.aux_weight)
    ah.balance_weight = float(config.balance_weight)
    ah.use_gate_noise = bool(config.use_gate_noise)
    ah.mean_fusion = bool(config.mean_fusion)

    n_gate = sum(p.numel() for p in ah.gate_out_proj.parameters())
    n_gate_noise = (
        sum(p.numel() for p in ah.gate_noise_layer.parameters()) if ah.use_gate_noise else 0
    )
    print(
        f"[MoH] horizons={horizons}  aux_weight={config.aux_weight}  "
        f"balance_weight={config.balance_weight}  use_gate_noise={config.use_gate_noise}  "
        f"mean_fusion={config.mean_fusion}"
    )
    print(
        f"[MoH] Gate modules from-scratch: gate_out_proj={n_gate:,} params, "
        f"gate_noise_layer={n_gate_noise:,} params"
    )
    if config.use_dynamic_replanning:
        print(
            f"[MoH] Dynamic replanning ON at inference: scale_ratio={config.scale_ratio}, "
            f"min_replan_steps={config.min_replan_steps}, min_active_horizons={config.min_active_horizons}"
        )

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # -------- Step 3: TrainingArguments + run --------
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
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    if not os.path.exists(config.output_dir):
        config.resume = False

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.train()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("GR00T-MoH FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    )
    assert config.num_gpus > 0, "num_gpus must be > 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
            ]
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    cmd.append(f"--{key.replace('_', '-')}" if value else f"--no-{key.replace('_', '-')}")
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    if isinstance(value, (list, tuple)):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command:", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
