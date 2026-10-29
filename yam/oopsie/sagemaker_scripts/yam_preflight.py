"""Preflight for a YAM pi05 launch: resolve the TrainConfig from the exact training argv, open the LeRobot
dataset, run two samples through the data transforms and print the shapes the model will see. Exits non-zero
on any problem — meant to run on CPU (JAX_PLATFORMS=cpu) before a GPU job is submitted.
Usage: yam_preflight.py -- <config> [tyro overrides...]"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    sys.argv = [sys.argv[0], *argv]
    config = _config.cli()
    data_config = config.data.create(config.assets_dirs, config.model)
    if not data_config.repo_id:
        raise SystemExit("data config has no repo_id")
    from lerobot.common.datasets import lerobot_dataset
    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    report = {"config": config.name, "exp_name": config.exp_name, "repo_id": data_config.repo_id,
              "episodes": meta.total_episodes, "frames": meta.total_frames, "fps": meta.fps,
              "video_keys": list(meta.video_keys), "tasks": list(meta.tasks.values())[:5],
              "action_horizon": config.model.action_horizon, "batch_size": config.batch_size,
              "num_train_steps": config.num_train_steps, "save_interval": config.save_interval,
              "weight_loader": repr(config.weight_loader), "assets_dirs": str(config.assets_dirs),
              "checkpoint_dir": str(config.checkpoint_dir)}
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(dataset, [*data_config.repack_transforms.inputs,
                                                        *data_config.data_transforms.inputs])
    shapes = {}
    for i in (0, len(dataset) // 2):
        item = dataset[i]
        shapes[str(i)] = {k: (list(np.asarray(v).shape) if not isinstance(v, str) else f"str:{v[:40]}")
                          for k, v in item.items()}
    report["sample_shapes"] = shapes
    print("PREFLIGHT_OK " + json.dumps(report))


if __name__ == "__main__":
    main()
