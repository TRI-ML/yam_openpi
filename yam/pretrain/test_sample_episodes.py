import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from assemble_v21 import parse_sample, sample_episodes  # noqa: E402


def test_sample_episodes():
    ids = list(range(1000))
    a = sample_episodes(ids, 0.1, 0)
    assert len(a) == 100
    assert a == sorted(set(a))
    assert a == sample_episodes(list(reversed(ids)), 0.1, 0)  # order-independent
    assert a != sample_episodes(ids, 0.1, 1)
    assert parse_sample("0.10") == (0.1, 0)
    assert parse_sample("0.10:3") == (0.1, 3)
