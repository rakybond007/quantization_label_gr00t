"""CPU-only regression coverage for weights-only Trainer RNG restore."""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.training_args import ParallelMode

from gr00t.experiment.trainer import DualBrainTrainer, _RNG_SAFE_GLOBALS


def _capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])


def _assert_rng_equal(actual, expected):
    assert actual["python"] == expected["python"]
    assert actual["numpy"][0] == expected["numpy"][0]
    assert np.array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]
    assert torch.equal(actual["cpu"], expected["cpu"])


def _trainer_for_rank(rank: int):
    trainer = object.__new__(DualBrainTrainer)
    trainer.args = SimpleNamespace(world_size=2, process_index=rank, parallel_mode=ParallelMode.NOT_DISTRIBUTED)
    return trainer


def test_weights_only_rng_restore_is_scoped_and_restores_cpu_rng(tmp_path):
    caller = _capture_rng()
    try:
        random.seed(11); np.random.seed(12); torch.manual_seed(13)
        saved = _capture_rng()
        torch.save(saved, tmp_path / "rng_state_0.pth")
        random.seed(21); np.random.seed(22); torch.manual_seed(23)
        _trainer_for_rank(0)._load_rng_state(str(tmp_path))
        _assert_rng_equal(_capture_rng(), saved)
        # The allowlist is lexical: a normal weights-only load still rejects
        # the NumPy reconstruction object after the trainer method returns.
        with pytest.raises(pickle.UnpicklingError):
            torch.load(tmp_path / "rng_state_0.pth", weights_only=True)
    finally:
        _restore_rng(caller)


@pytest.mark.parametrize("rank", [0, 1])
def test_existing_libero_rank_rng_files_restore_without_mutation(rank):
    checkpoint = Path("/ckpt/hojin/libero/isr_20260907_v1/checkpoint-500")
    if not checkpoint.exists():
        pytest.skip("cluster checkpoint unavailable")
    caller = _capture_rng()
    try:
        with torch.serialization.safe_globals(_RNG_SAFE_GLOBALS):
            expected = torch.load(checkpoint / f"rng_state_{rank}.pth", weights_only=True)
        _trainer_for_rank(rank)._load_rng_state(str(checkpoint))
        _assert_rng_equal(_capture_rng(), expected)
    finally:
        _restore_rng(caller)
