"""Policy inputs/outputs for the YAM bimanual robot.

State/action: 14D joint positions [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
Cameras: head (base view), left_wrist, right_wrist
"""

import dataclasses
import functools
import logging
import os
import pathlib

import einops
import numpy as np
import yaml

from openpi import transforms
from openpi.models import model as _model


def make_yam_example() -> dict:
    """Creates a random input example for the YAM policy."""
    return {
        "observation/state": np.random.rand(14).astype(np.float32),
        "observation/image_head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the lock and put it into the box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


_DEFAULT_ANNOTATIONS = pathlib.Path(__file__).resolve().parents[3] / "yam/config/language_annotations.yaml"


def _norm_prompt(text: str) -> str:
    return " ".join(str(text).split()).lower().rstrip(".!")


@functools.lru_cache(maxsize=1)
def _lang_aug() -> tuple[str, float, dict[str, list[str]]]:
    """Language augmentation from env. YAM_LANG_AUG_MODE selects the scheme (default "swap"):
      swap:    with probability YAM_LANG_AUG_P (unset/0 = off) replace the recorded prompt by one of its `randomized` paraphrases
      uniform: always draw uniformly from ALL strings of the task (every `original` + every `randomized`), YAM_LANG_AUG_P ignored
    YAM_LANG_ANNOTATIONS points at the LBM-format yaml. Returns (mode, p, {normalized recorded prompt -> candidate strings});
    every `original` of a task maps to the same list."""
    mode = os.environ.get("YAM_LANG_AUG_MODE", "swap").strip().lower()
    p = float(os.environ.get("YAM_LANG_AUG_P", "0") or 0)
    if mode == "uniform":
        p = 1.0
    if mode not in ("swap", "uniform"):
        raise ValueError(f"YAM_LANG_AUG_MODE must be swap or uniform, got {mode!r}")
    if p <= 0:
        return mode, 0.0, {}
    path = pathlib.Path(os.environ.get("YAM_LANG_ANNOTATIONS", _DEFAULT_ANNOTATIONS))
    with open(path) as f:
        tasks = yaml.safe_load(f)["language_dict"]
    mapping = {}
    for t in tasks.values():
        originals = list(t.get("original", []))
        randomized = list(t.get("randomized", []))
        pool = list(dict.fromkeys(originals + randomized)) if mode == "uniform" else randomized
        if not pool:
            continue
        for o in originals:
            mapping[_norm_prompt(o)] = pool
    logging.info(
        "YAM language augmentation: mode=%s p=%.2f, %d recorded prompts -> candidates from %s",
        mode,
        p,
        len(mapping),
        path,
    )
    return mode, p, mapping


def augment_prompt(prompt):
    """Training-time prompt augmentation (see _lang_aug). Unknown prompts pass through unchanged."""
    mode, p, mapping = _lang_aug()
    if p <= 0:
        return prompt
    text = prompt.item() if hasattr(prompt, "item") else prompt
    alts = mapping.get(_norm_prompt(text))
    # np.random is re-seeded per DataLoader worker, so draws differ across workers.
    if alts and (mode == "uniform" or np.random.random() < p):
        return alts[np.random.randint(len(alts))]
    return prompt


@dataclasses.dataclass(frozen=True)
class YAMInputs(transforms.DataTransformFn):
    """Inputs for the YAM bimanual policy.

    Expected inputs:
    - observation/image_head: base camera image
    - observation/image_left_wrist: left wrist camera image
    - observation/image_right_wrist: right wrist camera image
    - observation/state: [14] joint positions
    - actions: [action_horizon, 14] (only during training)
    - prompt: language instruction
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image_head"])
        left_wrist_image = _parse_image(data["observation/image_left_wrist"])
        right_wrist_image = _parse_image(data["observation/image_right_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = augment_prompt(data["prompt"])  # no-op unless YAM_LANG_AUG_P is set (training only)

        return inputs


@dataclasses.dataclass(frozen=True)
class YAMOutputs(transforms.DataTransformFn):
    """Outputs for the YAM bimanual policy.

    Returns the first 14 action dimensions (matching the 14D joint space).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
