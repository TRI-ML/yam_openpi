"""Write the `deploy_metadata.json` sidecar next to a trained openpi checkpoint step dir.

openpi's train.py writes params/ train_state/ assets/ _CHECKPOINT_METADATA but not the self-describing sidecar that
YAM Eval (checkpoints.load_deploy_sidecar), the stations' serve_policy fallback (deploy_metadata.config_from_sidecar) and
the yam-registry ingest read. This backfills it from the TrainConfig (+ the YAM deploy conventions), e.g.

  uv run scripts/write_deploy_sidecar.py --checkpoint-dir ~/rfm_rl/outputs/<run>/checkpoints/<step> \
      --config-name pi05_yam_abc_mm2_h32 --pretrain-source s3://.../pi05_full_yam_abc_mm2_h32_dgx3node/checkpoints/25000
  uv run scripts/write_deploy_sidecar.py --checkpoint-dir <ft ckpt> --config-name pi05_yam_abc_mm2_h32_ft \
      --task-instruction "Place the teabag into the teapot"

Conventions baked in (override with flags): camera_conventions = {right_wrist_camera: rotate180} (the YAM station mounts
the right wrist ZED upside down and every YAM LeRobot export stores it rotated), action_space absolute joint positions,
14-D layout [L joints 0-5, L gripper, R joints 0-5, R gripper] with grippers 0..1 (1 = open).
"""

from __future__ import annotations

import argparse
import json
import pathlib

JOINT14_LAYOUT = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint-dir", required=True, help="<checkpoint_base>/<exp>/<step> (holds params/ and assets/)"
    )
    ap.add_argument("--config-name", required=True, help="registered TrainConfig name the checkpoint was trained with")
    ap.add_argument(
        "--task-instruction", default=None, help="prompt baked into the sidecar (omit for multi-task pretrain ckpts)"
    )
    ap.add_argument(
        "--camera-convention",
        action="append",
        default=None,
        metavar="CAMERA=rotate180|raw",
        help="default: right_wrist_camera=rotate180 (repeatable)",
    )
    ap.add_argument(
        "--action-space", default="absolute_joint14", help="absolute_joint14 (default) or a delta_* recipe name"
    )
    ap.add_argument(
        "--pretrain-source", default=None, help="URI of the pretrain checkpoint this run was initialised from"
    )
    ap.add_argument("--dataset-snapshot-id", default=None, help="yam-registry DatasetSnapshot id of the training data")
    ap.add_argument("--extra", default=None, help="JSON object merged into the sidecar (e.g. '{\"wandb_url\": ...}')")
    ap.add_argument("--force", action="store_true", help="overwrite an existing deploy_metadata.json")
    args = ap.parse_args()

    from openpi.training import config as _config
    from openpi.training import deploy_metadata as _dm

    step_dir = pathlib.Path(args.checkpoint_dir).expanduser()
    if not (step_dir / "params").is_dir():
        raise SystemExit(f"{step_dir} has no params/ dir — pass the <step> directory of an openpi checkpoint")
    out = step_dir / "deploy_metadata.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")

    cfg = _config.get_config(args.config_name)
    meta = _dm.build_metadata(cfg, task_instruction=args.task_instruction, dataset_snapshot_id=args.dataset_snapshot_id)
    conv = {"right_wrist_camera": "rotate180"}
    for kv in args.camera_convention or []:
        cam, _, val = kv.partition("=")
        if val not in ("rotate180", "raw"):
            raise SystemExit(f"bad --camera-convention {kv!r}: value must be rotate180|raw")
        conv[cam] = val
    assets_dir = step_dir / "assets"
    asset_id = meta.get("asset_id") or (cfg.data.repo_id if hasattr(cfg.data, "repo_id") else None)
    meta.update(
        {
            "camera_conventions": conv,
            "action_space": args.action_space,
            "action_layout": "joint14_abs" if args.action_space == "absolute_joint14" else args.action_space,
            "action_names": JOINT14_LAYOUT,
            "gripper_convention": "0..1, 1 = open",
            "control_hz": 30,
            "norm_stats": str(assets_dir / asset_id / "norm_stats.json") if asset_id else str(assets_dir),
            "pretrain_source": args.pretrain_source,
            "checkpoint_step": int(step_dir.name) if step_dir.name.isdigit() else None,
        }
    )
    if args.extra:
        meta.update(json.loads(args.extra))
    out.write_text(json.dumps(meta, indent=2))
    print(
        f"wrote {out}\n  config={meta['config_name']} horizon={meta['action_horizon']} task={meta.get('task_instruction')!r} "
        f"cameras={conv} action_space={args.action_space}"
    )


if __name__ == "__main__":
    main()
