"""Policy inputs/outputs for the heterogeneous "oopsie" multi-embodiment corpus.

State/action live in a common COMMON_DIM (18) space produced by convert_oopsie_to_lerobot.py:
each donor lab's (motion ++ gripper) vector is left-aligned and zero-padded to 18, with a
per-frame action_mask marking the valid prefix. Cameras are mapped onto the canonical pi05
3-camera layout (base / left_wrist / right_wrist) at conversion time.

Mirrors yam_policy.YAMInputs/YAMOutputs but keeps the full 18-D space (no 14-D YAM slicing),
so it can serve as the data-transform for the metadata-conditioned pi05_oopsie_smoke config
without touching the YAM transforms.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

COMMON_DIM = 18


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OopsieInputs(transforms.DataTransformFn):
    """Maps repacked oopsie columns to pi0/pi05 model inputs.

    Expected (post-repack) keys: observation/image_head, observation/image_left_wrist,
    observation/image_right_wrist, observation/state [COMMON_DIM], actions [horizon, COMMON_DIM]
    (train only), prompt.
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": _parse_image(data["observation/image_head"]),
                "left_wrist_0_rgb": _parse_image(data["observation/image_left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/image_right_wrist"]),
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
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class OopsieOutputs(transforms.DataTransformFn):
    """Returns the first COMMON_DIM action dimensions (inference only)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :COMMON_DIM])}
