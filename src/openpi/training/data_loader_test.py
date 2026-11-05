import dataclasses

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_source_mix_sampler():
    sources = {
        "molmo": {"first_frame": 0, "end_frame": 100},
        "abc": {"first_frame": 100, "end_frame": 400},
        "oopsie_A": {"first_frame": 400, "end_frame": 410},
        "oopsie_B": {"first_frame": 410, "end_frame": 1410},
        "absim": {"first_frame": 1410, "end_frame": 1500},
    }
    mix = {"abc+molmo": 0.35, "oopsie_*": 0.35, "absim": 0.30}
    s = _data_loader.SourceMixSampler(sources, mix, num_samples=200_000, seed=0)
    idx = np.fromiter(iter(s), np.int64)
    assert len(idx) == 200_000
    assert idx.min() >= 0
    assert idx.max() < 1500
    frac = lambda a, b: np.mean((idx >= a) & (idx < b))
    assert abs(frac(0, 400) - 0.35) < 0.01
    assert abs(frac(1410, 1500) - 0.30) < 0.01
    assert abs(frac(400, 410) - 0.175) < 0.01  # labs equal
    assert abs(frac(410, 1410) - 0.175) < 0.01
    assert abs(frac(0, 100) / frac(0, 400) - 0.25) < 0.02  # frame-proportional inside the abc+molmo pool
    with pytest.raises(ValueError, match="not covered"):
        _data_loader.SourceMixSampler(sources, {"abc+molmo": 1.0}, num_samples=1)
