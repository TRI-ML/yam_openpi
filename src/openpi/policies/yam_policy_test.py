# ruff: noqa: SLF001
import numpy as np
import pytest

from openpi.policies import yam_policy

RECORDED = "Place the teabag into the teapot"
TRUNCATED_SORT = "Pick up every utensil and place it into the tray, grouped by type — forks in one compartment, knives"


@pytest.fixture
def aug(monkeypatch):
    def _set(p):
        monkeypatch.setenv("YAM_LANG_AUG_P", str(p))
        yam_policy._lang_aug.cache_clear()

    yield _set
    yam_policy._lang_aug.cache_clear()


def test_off_by_default(aug):
    aug(0)
    assert yam_policy.augment_prompt(RECORDED) == RECORDED
    assert yam_policy.augment_prompt(np.asarray(RECORDED)) is not None


def test_always_paraphrases_from_the_right_task(aug):
    aug(1)
    _, mapping = yam_policy._lang_aug()
    assert len(mapping) >= 11  # 9 tasks, SortUtensils has 3 recorded variants
    alts = set(mapping[yam_policy._norm_prompt(RECORDED)])
    assert len(alts) == 5
    for _ in range(50):
        out = yam_policy.augment_prompt(np.asarray(RECORDED))
        assert out in alts
        assert "table" not in out.lower()  # never the reset task ("take the teabag out ... on the table")


def test_truncated_sortutensils_variant_maps(aug):
    aug(1)
    out = yam_policy.augment_prompt(TRUNCATED_SORT + ".")
    assert out != TRUNCATED_SORT + "."
    assert "utensil" in out.lower()


def test_unknown_prompt_untouched(aug):
    aug(1)
    assert yam_policy.augment_prompt("fold the towel") == "fold the towel"


def test_inputs_transform_applies(aug):
    aug(1)
    ex = yam_policy.make_yam_example()
    ex["prompt"] = RECORDED
    out = yam_policy.YAMInputs()(ex)
    assert out["prompt"] != RECORDED
