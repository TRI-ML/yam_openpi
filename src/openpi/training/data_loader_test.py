import dataclasses
import types

import jax
import numpy as np
import pytest

from openpi import transforms as _transforms
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


def test_source_mix_sampler_s2v2():
    sources = {
        "molmo": {"first_frame": 0, "end_frame": 100},
        "abc": {"first_frame": 100, "end_frame": 400},
        "oopsie_A": {"first_frame": 400, "end_frame": 410},
        "oopsie_B": {"first_frame": 410, "end_frame": 1410},
        "absim": {"first_frame": 1410, "end_frame": 1500},
        "absim_teleop": {"first_frame": 1500, "end_frame": 5000},
    }
    mix = _config.get_config("pi05_yam_s2v2_meta").data.base_config.source_mix
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    s = _data_loader.SourceMixSampler(sources, mix, num_samples=200_000, seed=0)
    idx = np.fromiter(iter(s), np.int64)

    def frac(a, b):
        return np.mean((idx >= a) & (idx < b))

    assert abs(frac(0, 400) - 0.35) < 0.01
    assert abs(frac(400, 1410) - 0.30) < 0.01
    assert abs(frac(1410, 1500) - 0.15) < 0.01  # exact key "absim" does not swallow absim_teleop
    assert abs(frac(1500, 5000) - 0.20) < 0.01
    assert _transforms.SOURCE_GROUPS[_transforms.source_group("absim_teleop")] == "absim_teleop"


class _FlakyDataset:
    def __init__(self, bad):
        self.bad = bad
        self.episode_data_index = {"from": np.array([0, 5]), "to": np.array([5, 10])}

    def __getitem__(self, i):
        if i in self.bad:
            raise AssertionError("query timestamps violate the tolerance")
        return {"i": i, "source": "abc"}

    def __len__(self):
        return 10


def test_skip_bad_samples():
    ds = _data_loader.SkipBadSamples(_FlakyDataset({3}), count_key="data_skipped")
    assert ds[2] == {"i": 2, "source": "abc", "data_skipped": 0}
    item = ds[3]
    assert item["i"] != 3
    assert item["data_skipped"] >= 1
    with pytest.raises(AssertionError):
        _data_loader.SkipBadSamples(_FlakyDataset(set(range(10))), max_retries=8)[0]
    assert "data_skipped" not in _data_loader.SkipBadSamples(_FlakyDataset(set()))[0]


def test_video_tolerance_passed(monkeypatch):
    seen = {}

    meta = types.SimpleNamespace(fps=30, tasks={}, features={"source": {}})
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda repo_id: meta)
    monkeypatch.setattr(_data_loader, "make_lerobot_dataset", lambda repo_id, **kw: seen.update(kw) or _FlakyDataset(set()))
    for name in ("pi05_yam_s2_plain", "pi05_yam_s2_meta", "pi05_yam_s2v2_plain", "pi05_yam_s2v2_meta"):
        dc = dataclasses.replace(_config.get_config(name).data.base_config, repo_id="local/x", prompt_from_task=False)
        assert dc.video_tolerance_s == 0.05
        ds = _data_loader.create_torch_dataset(dc, 4, pi0_config.Pi0Config())
        assert seen["tolerance_s"] == 0.05
        assert ds[0]["data_skipped"] == 0


def test_stratified_mix_sampler():
    # frames: real_success abc 900 + oopsie_A 100; real_failure oopsie_A 100 + oopsie_B 1600; sim_success tele 2500; sim_failure absim 400
    eps = []
    f0 = 0
    for src, q, d, lens in (("abc", 1, 0, [300, 600]), ("oopsie_A", 1, 0, [100]), ("oopsie_A", 0, 0, [50, 50]),
                            ("oopsie_B", 0, 0, [1600]), ("absim_teleop", 1, 1, [2500]), ("absim", 0, 1, [400])):
        for n in lens:
            eps.append({"episode_index": len(eps), "source": src, "quality": q, "domain": d, "first_frame": f0, "length": n})
            f0 += n
    mix = {"real_success": 0.30, "real_failure": 0.20, "sim_success": 0.30, "sim_failure": 0.20}
    s = _data_loader.StratifiedMixSampler(eps, mix, num_samples=400_000, alpha=0.5, seed=0)
    idx = np.fromiter(iter(s), np.int64)
    assert idx.min() >= 0 and idx.max() < f0 and len(idx) == 400_000

    def frac(a, b):
        return np.mean((idx >= a) & (idx < b))

    assert abs(frac(0, 1000) - 0.30) < 0.01  # real_success
    assert abs(frac(1000, 2700) - 0.20) < 0.01  # real_failure
    assert abs(frac(2700, 5200) - 0.30) < 0.01  # sim_success
    assert abs(frac(5200, 5600) - 0.20) < 0.01  # sim_failure
    assert abs(frac(0, 900) / frac(0, 1000) - 30 / 40) < 0.02  # sqrt(900):sqrt(100) = 3:1 inside real_success
    assert abs(frac(1000, 1100) / frac(1000, 2700) - 10 / 50) < 0.02  # sqrt(100):sqrt(1600) = 1:4 inside real_failure
    assert abs(frac(1000, 1050) - frac(1050, 1100)) < 0.005  # uniform over a pool's frames
    # a stratum absent from the repo is dropped and the rest renormalised; an unlisted present stratum is an error
    s2 = _data_loader.StratifiedMixSampler([e for e in eps if e["domain"] == 0], mix, num_samples=100_000, seed=0)
    idx2 = np.fromiter(iter(s2), np.int64)
    assert abs(np.mean(idx2 < 1000) - 0.6) < 0.01
    with pytest.raises(ValueError, match="not in strata_mix"):
        _data_loader.StratifiedMixSampler(eps, {"real_success": 1.0}, num_samples=1)


def test_quality_domain_prefix_prompt_field_dropout():
    tasks = {0: "sim pouring beads", 1: "fold the towel"}
    remap = {"sim pouring beads": "pouring beads"}
    t = _transforms.QualityDomainPrefixPrompt(tasks, dropout_p=0.3, field_dropout=True, prompt_remap=remap)
    outs = [t({"task_index": 0, "index": i, "quality": 0, "domain": 1}) for i in range(4000)]
    q = np.mean([o["quality_dropped"] for o in outs])
    d = np.mean([o["domain_dropped"] for o in outs])
    assert abs(q - 0.3) < 0.03 and abs(d - 0.3) < 0.03
    assert abs(np.mean([o["prefix_dropped"] for o in outs]) - (1 - 0.7 * 0.7)) < 0.03  # independent fields
    assert all(o["prompt"].endswith("pouring beads") and "sim pouring" not in o["prompt"] for o in outs)
    full = next(o for o in outs if not o["prefix_dropped"])
    assert full["prompt"] == "<quality=failure> <domain=sim> pouring beads"
    only_d = next(o for o in outs if o["quality_dropped"] and not o["domain_dropped"])
    assert only_d["prompt"] == "<domain=sim> pouring beads"
    # v2 behaviour unchanged: joint drop, no remap
    t2 = _transforms.QualityDomainPrefixPrompt(tasks, dropout_p=0.3)
    o = [t2({"task_index": 1, "index": i, "quality": 1, "domain": 0}) for i in range(2000)]
    assert all(x["quality_dropped"] == x["domain_dropped"] for x in o)
    assert abs(np.mean([x["prefix_dropped"] for x in o]) - 0.3) < 0.03
    assert any(x["prompt"] == "<quality=success> <domain=real> fold the towel" for x in o)


def test_s3_config():
    c = _config.get_config("pi05_yam_s3_meta")
    b = c.data.base_config
    assert abs(sum(b.strata_mix.values()) - 1.0) < 1e-9 and b.meta_prefix_field_dropout and b.source_mix is None
    import json
    remap = json.load(open(b.prompt_remap_json))
    assert len(remap) == 1431 and set(remap.values()) <= {
        "grab clutter", "put relative", "pouring beads", "hang the mug on the mug rack", "turn the mug right side up",
        "throw plastic bottles in bin", "sweep away paper scraps from the table", "load the plates into the dish rack"}
    assert c.lr_schedule.decay_steps == c.num_train_steps == 120_000


def test_stratified_mix_sampler_sim_only():
    eps = []
    f0 = 0
    for src, q, d, n in (("abc", 1, 0, 1000), ("oopsie_A", 0, 0, 100), ("absim_teleop", 1, 1, 3000), ("absim", 0, 1, 1000)):
        eps.append({"episode_index": len(eps), "source": src, "quality": q, "domain": d, "first_frame": f0, "length": n})
        f0 += n
    mix = {"real_success": 0.0, "real_failure": 0.0, "sim_success": 0.5, "sim_failure": 0.5}
    s = _data_loader.StratifiedMixSampler(eps, mix, num_samples=200_000, seed=0)
    idx = np.fromiter(iter(s), np.int64)
    assert idx.min() >= 1100  # real pools never drawn
    assert abs(np.mean((idx >= 1100) & (idx < 4100)) - 0.5) < 0.01 and abs(np.mean(idx >= 4100) - 0.5) < 0.01
    assert s._names == ["sim_failure/absim", "sim_success/absim_teleop"]
    # a positive-share stratum that is absent is dropped (warning) and the rest renormalised
    s2 = _data_loader.StratifiedMixSampler([e for e in eps if e["source"] != "absim"], mix, num_samples=50_000, seed=0)
    idx2 = np.fromiter(iter(s2), np.int64)
    assert idx2.min() >= 1100 and idx2.max() < 4100
    with pytest.raises(ValueError, match="no stratum"):
        _data_loader.StratifiedMixSampler([e for e in eps if e["domain"] == 0], mix, num_samples=1)


def test_s3sim_config():
    c = _config.get_config("pi05_yam_s3sim_meta")
    b = c.data.base_config
    assert b.strata_mix == {"real_success": 0.0, "real_failure": 0.0, "sim_success": 0.5, "sim_failure": 0.5}
    assert b.meta_prefix_field_dropout and b.prompt_remap_json and c.data.repo_id == "local/yam_s2v2_train_v21"
    assert c.lr_schedule.decay_steps == c.num_train_steps == 120_000 and c.save_interval == 5_000
