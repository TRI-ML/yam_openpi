"""
Convert "oopsie" (oopsiedata_format_v1) multi-embodiment data to a LeRobot v2.1 dataset
for metadata-conditioned pi05/openpi pretraining smoke tests.

Sibling of convert_yam_video_to_lerobot.py. Unlike that converter (which hard-codes the 14-D YAM
joint layout, 3 fixed cameras, fps30), the oopsie corpus is HETEROGENEOUS across donor labs:
each lab has a different action space (e.g. CMU_Auton xarm7 cartesian_position(7)+gripper_binary(1);
Enatic_OpenArm bimanual joint_position(16)+gripper_position(2)) and a different camera set.

What this converter does:
  * reads episodes straight from S3 (s3://tri-ml-datasets-uw2/anthonyliang/oopsie/<Lab>/), using the
    aws CLI (no boto3 dependency) so it runs under the plain openpi `uv` env;
  * discovers episodes generically: every `*.h5` under a lab prefix (ignoring `.hidden`/`.cache`);
    mp4 keys are resolved RELATIVE TO EACH H5's OWN DIRECTORY from observations/video_paths, so both
    the flat layout (CMU_Auton: lab/<id>.h5 + lab/<id>_<cam>.mp4) and the nested layout
    (USP_MB: lab/<id>/<id>.h5 + lab/<id>/<id>_<cam>.mp4) work with no special-casing;
  * pads each lab's (motion ++ gripper) action vector to a common COMMON_DIM (18) and emits a per-frame
    `action_mask` marking the valid (unpadded) dims, keyed by embodiment;
  * maps each lab's cameras onto the canonical pi05 3-camera layout (base / left_wrist / right_wrist),
    duplicating the base view into any missing wrist slot;
  * emits metadata columns meta_prompt / success / speed_bucket / embodiment_id, with task=language_instruction.
    speed_bucket is a tertile of mean per-step |Δaction| * control_freq across the converted episodes.

The resulting dataset (default local/oopsie_smoke) trains with the NEW pi05_oopsie_smoke config.

Usage (note: run with the plain openpi uv env; the aws CLI must be configured for us-west-2 reads):
  uv run yam_dataset_builder/yam_dataset_builder/convert_oopsie_to_lerobot.py \
      --repo-id local/oopsie_smoke --overwrite
"""

import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import tqdm
import tyro
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

BUCKET = "tri-ml-datasets-uw2"
S3_ROOT = f"s3://{BUCKET}/anthonyliang/oopsie"
COMMON_DIM = 18  # covers the widest sampled embodiment (Enatic bimanual joint16 + gripper2 = 18)
IMG_HW = 224  # convert-time resize; pi05 resizes to 224 anyway, keeps the smoke dataset tiny
FPS = 10
CANONICAL_CAMS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]


@dataclasses.dataclass(frozen=True)
class LabSpec:
    embodiment_id: int
    embodiment_name: str
    # (hdf5 key under actions/, dim) for the primary motion channel then the gripper channel.
    motion_key: str
    motion_dim: int
    gripper_key: str
    gripper_dim: int
    # (hdf5 key under observations/robot_states/, dim) list concatenated to form observation.state.
    state_keys: tuple[tuple[str, int], ...]
    # canonical_slot -> source camera name in observations/video_paths (None => duplicate base).
    camera_map: dict[str, Optional[str]]


# Per-lab assembly specs. Derived from the annotation-coverage probe of actions/ + robot_states/ shapes.
LAB_SPECS: dict[str, LabSpec] = {
    "CMU_Auton": LabSpec(
        embodiment_id=0, embodiment_name="xarm7_single",
        motion_key="cartesian_position", motion_dim=7, gripper_key="gripper_binary", gripper_dim=1,
        state_keys=(("cartesian_position", 7), ("gripper_position", 1)),
        camera_map={"base_0_rgb": "exterior_image_1", "left_wrist_0_rgb": None, "right_wrist_0_rgb": None},
    ),
    "USP_MB": LabSpec(
        embodiment_id=1, embodiment_name="franka_single",
        motion_key="cartesian_position", motion_dim=7, gripper_key="gripper_position", gripper_dim=1,
        state_keys=(("cartesian_position", 7), ("gripper_position", 1)),
        camera_map={"base_0_rgb": "front", "left_wrist_0_rgb": "wrist", "right_wrist_0_rgb": "top"},
    ),
}


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str = "local/oopsie_smoke"
    labs: tuple[str, ...] = ("CMU_Auton", "USP_MB")
    episodes_per_lab: int = 3
    overwrite: bool = False
    cache_dir: Path = Path("/tmp/oopsie_convert_cache")


def _aws_cp(key: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dst), "--only-show-errors"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"aws cp failed for {key}:\n{r.stderr}")


def _list_h5_keys(lab: str, limit: int) -> list[str]:
    r = subprocess.run(
        ["aws", "s3", "ls", f"{S3_ROOT}/{lab}/", "--recursive"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"aws ls failed for {lab}:\n{r.stderr}")
    keys = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[-1]
        if key.endswith(".h5") and ".hidden" not in key and "/.cache/" not in key and "/.datahive/" not in key:
            keys.append(key)
    keys.sort()
    return keys[:limit]


def _str_attr(obj, name) -> Optional[str]:
    v = obj.attrs.get(name)
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return v


def _read_success(f: h5py.File) -> float:
    if "episode_annotations" in f:
        for ann in f["episode_annotations"]:
            s = f["episode_annotations"][ann].attrs.get("success")
            if s is not None:
                return float(s)
    return -1.0


def _assemble_action(f: h5py.File, spec: LabSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (action [T, COMMON_DIM] float32, action_mask [COMMON_DIM] float32)."""
    motion = np.asarray(f[f"actions/{spec.motion_key}"][()], dtype=np.float32).reshape(-1, spec.motion_dim)
    grip = np.asarray(f[f"actions/{spec.gripper_key}"][()], dtype=np.float32).reshape(-1, spec.gripper_dim)
    t = min(motion.shape[0], grip.shape[0])
    valid = spec.motion_dim + spec.gripper_dim
    act = np.zeros((t, COMMON_DIM), dtype=np.float32)
    act[:, : spec.motion_dim] = motion[:t]
    act[:, spec.motion_dim : valid] = grip[:t]
    mask = np.zeros((COMMON_DIM,), dtype=np.float32)
    mask[:valid] = 1.0
    return act, mask


def _assemble_state(f: h5py.File, spec: LabSpec, t: int) -> np.ndarray:
    parts = []
    for key, dim in spec.state_keys:
        arr = np.asarray(f[f"observations/robot_states/{key}"][()], dtype=np.float32).reshape(-1, dim)
        parts.append(arr[:t])
    cat = np.concatenate(parts, axis=1)
    state = np.zeros((t, COMMON_DIM), dtype=np.float32)
    state[:, : cat.shape[1]] = cat
    return state


def _decode_cam(mp4_path: Path, num_frames: int) -> np.ndarray:
    """Decode up to num_frames frames, resized to IMG_HW x IMG_HW, as (n, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while len(frames) < num_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_HW, IMG_HW), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {mp4_path}")
    return np.stack(frames, axis=0)


def main(args: Args) -> None:
    for lab in args.labs:
        if lab not in LAB_SPECS:
            raise ValueError(f"No LabSpec for '{lab}'. Known: {list(LAB_SPECS)}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: download h5 (small), assemble actions/state/meta, collect speeds for tertiles ----
    episodes = []  # list of dicts
    speeds = []
    for lab in args.labs:
        spec = LAB_SPECS[lab]
        h5_keys = _list_h5_keys(lab, args.episodes_per_lab)
        print(f"[{lab}] found {len(h5_keys)} h5 (using {min(len(h5_keys), args.episodes_per_lab)})")
        for key in h5_keys:
            local_h5 = args.cache_dir / key.replace("/", "__")
            _aws_cp(key, local_h5)
            with h5py.File(local_h5, "r") as f:
                instruction = _str_attr(f, "language_instruction") or "do the task"
                operator = (_str_attr(f, "operator_name") or "unknown").split()[0]
                profile_raw = _str_attr(f, "robot_profile")
                ctrl_freq = FPS
                if profile_raw:
                    try:
                        ctrl_freq = float(json.loads(profile_raw).get("control_freq") or FPS)
                    except Exception:
                        pass
                success = _read_success(f)
                action, mask = _assemble_action(f, spec)
                t = action.shape[0]
                state = _assemble_state(f, spec, t)
                valid = spec.motion_dim + spec.gripper_dim
            local_h5.unlink()
            if t < 2:
                print(f"  skip {key}: only {t} frames")
                continue
            dact = np.abs(np.diff(action[:, :valid], axis=0))
            speed = float(dact.mean() * ctrl_freq)
            speeds.append(speed)
            h5_dir = key.rsplit("/", 1)[0]
            episodes.append(dict(
                lab=lab, spec=spec, key=key, h5_dir=h5_dir, instruction=instruction,
                operator=operator, success=success, action=action, mask=mask, state=state, speed=speed,
            ))

    if not episodes:
        print("No episodes assembled; aborting.")
        sys.exit(1)

    speeds_arr = np.asarray(speeds)
    lo, hi = np.quantile(speeds_arr, [1 / 3, 2 / 3])
    def bucket(s: float) -> int:
        return 0 if s <= lo else (1 if s <= hi else 2)
    print(f"speed tertile thresholds: lo={lo:.4f} hi={hi:.4f} (n={len(speeds)})")

    # need the mp4 video_paths; re-open is avoided by re-reading only the string group per episode later.

    # ---- Build LeRobot dataset ----
    out_dir = HF_LEROBOT_HOME / args.repo_id
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            print(f"{out_dir} exists; pass --overwrite.")
            sys.exit(1)

    features = {
        "observation.state": {"dtype": "float32", "shape": (COMMON_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (COMMON_DIM,), "names": None},
        "action_mask": {"dtype": "float32", "shape": (COMMON_DIM,), "names": None},
        "meta_prompt": {"dtype": "string", "shape": (1,), "names": None},
        "success": {"dtype": "float32", "shape": (1,), "names": None},
        "speed_bucket": {"dtype": "int64", "shape": (1,), "names": None},
        "embodiment_id": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for cam in CANONICAL_CAMS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (3, IMG_HW, IMG_HW), "names": ["channels", "height", "width"]}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, robot_type="oopsie_multi", features=features,
        use_videos=True, tolerance_s=1.0, image_writer_processes=0, image_writer_threads=4)

    for ep in tqdm.tqdm(episodes, desc="Converting"):
        spec: LabSpec = ep["spec"]
        action, mask, state = ep["action"], ep["mask"], ep["state"]
        t = action.shape[0]
        # fetch video_paths + mp4s for this episode
        local_h5 = args.cache_dir / ep["key"].replace("/", "__")
        _aws_cp(ep["key"], local_h5)
        with h5py.File(local_h5, "r") as f:
            vpaths = {}
            for cam in f["observations/video_paths"]:
                v = f["observations/video_paths"][cam][()]
                vpaths[cam] = v.decode() if isinstance(v, bytes) else str(v)
        local_h5.unlink()

        # resolve + decode only the unique source cams we need
        needed = {src for src in spec.camera_map.values() if src is not None}
        decoded: dict[str, np.ndarray] = {}
        tmp_mp4_dir = Path(tempfile.mkdtemp(dir=args.cache_dir))
        try:
            for src in needed:
                if src not in vpaths:
                    raise RuntimeError(f"{ep['key']}: camera '{src}' not in video_paths {list(vpaths)}")
                mp4_key = f"{ep['h5_dir']}/{vpaths[src]}"
                local_mp4 = tmp_mp4_dir / f"{src}.mp4"
                _aws_cp(mp4_key, local_mp4)
                decoded[src] = _decode_cam(local_mp4, t)
                local_mp4.unlink()
            n = min([t] + [v.shape[0] for v in decoded.values()])
            # assemble canonical camera frames (duplicate base into missing slots)
            base_src = spec.camera_map["base_0_rgb"]
            cam_frames = {}
            for slot, src in spec.camera_map.items():
                use = src if src is not None else base_src
                cam_frames[slot] = decoded[use]

            meta_prompt = (
                f"<opt={'success' if ep['success'] >= 0.5 else 'failure'}> "
                f"<speed={bucket(ep['speed'])}> <emb={spec.embodiment_name}> <op={ep['operator']}> "
            )
            for i in range(n):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "action_mask": mask,
                    "meta_prompt": meta_prompt,
                    "success": np.array([ep["success"]], dtype=np.float32),
                    "speed_bucket": np.array([bucket(ep["speed"])], dtype=np.int64),
                    "embodiment_id": np.array([spec.embodiment_id], dtype=np.int64),
                    "task": ep["instruction"],
                }
                for slot in CANONICAL_CAMS:
                    frame[f"observation.images.{slot}"] = cam_frames[slot][i]
                dataset.add_frame(frame)
            dataset.save_episode()
            print(f"  + {ep['lab']}/{ep['key'].rsplit('/', 1)[-1]}  frames={n} "
                  f"emb={spec.embodiment_name} opt={'S' if ep['success'] >= 0.5 else 'F'} "
                  f"speed_bucket={bucket(ep['speed'])}")
        finally:
            shutil.rmtree(tmp_mp4_dir, ignore_errors=True)

    print(f"\nDone. {dataset.meta.total_episodes} episodes -> {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
