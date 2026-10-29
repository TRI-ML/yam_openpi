"""compute_norm_stats for a config given as tyro CLI overrides (the same argv the training run uses), so the
stats match a dataset chosen at launch time (`--data.repo-id`, `--data.assets.asset-id`) rather than a config
baked into config.py. Usage: yam_norm_stats.py [--max-frames N] -- <config> [tyro overrides...]"""
import argparse
import sys

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

sys.path.insert(0, "scripts")
from compute_norm_stats import create_torch_dataloader  # noqa: E402  (the reference implementation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None)
    ns, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.argv = [sys.argv[0], *rest]
    config = _config.cli()
    data_config = config.data.create(config.assets_dirs, config.model)
    max_frames = ns.max_frames if ns.max_frames and ns.max_frames > 0 else None
    loader, num_batches = create_torch_dataloader(data_config, config.model.action_horizon, config.batch_size,
                                                  config.model, config.num_workers, max_frames)
    keys = ["state", "actions"]
    stats = {k: normalize.RunningStats() for k in keys}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing stats"):
        for k in keys:
            stats[k].update(np.asarray(batch[k]))
    out = config.assets_dirs / data_config.repo_id
    normalize.save(out, {k: s.get_statistics() for k, s in stats.items()})
    print(f"norm stats -> {out} ({num_batches} batches x {config.batch_size})")


if __name__ == "__main__":
    main()
