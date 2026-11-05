"""Shape check for yam/config/language_annotations.yaml (run: pytest yam/config)."""
import pathlib

import yaml

TASKS = {"HangMugs", "PlaceTeaBagInTeapot", "SweepBeans", "MakeCoffee", "ToastBread", "CleanUpSpill", "BuildSprinkler", "SortUtensils", "BikeRotorToolUse"}
RESET_WORDS = ("out of the teapot", "remove the tea", "take the tea", "on the table", "onto the plate", "back on the plate", "unlid", "take off the lid")


def test_language_annotations():
    d = yaml.safe_load((pathlib.Path(__file__).parent / "language_annotations.yaml").read_text())["language_dict"]
    assert set(d) == TASKS
    for task, v in d.items():
        assert set(v) == {"original", "randomized"}, task
        assert len(v["original"]) >= 1 and len(v["randomized"]) == 5, task
        items = v["original"] + v["randomized"]
        assert len(set(s.lower() for s in items)) == len(items), f"duplicate prompt in {task}"
        assert all(s == s.strip() and not s.endswith(".") for s in v["randomized"]), task
    # the reset task (take the tea bag OUT) must never leak into the forward task's prompts
    for s in d["PlaceTeaBagInTeapot"]["original"] + d["PlaceTeaBagInTeapot"]["randomized"]:
        assert not any(w in s.lower() for w in RESET_WORDS), s
