"""Per-episode strata sidecar for a stage-2 merged repo (assemble_v21.py --repo): <repo>/meta/episode_strata.jsonl with one
line per episode {episode_index, source, quality, domain, first_frame, length}. `source`/`quality`/`domain` are constant per
episode, so one parquet row each is enough; lengths/order come from meta/episodes.jsonl and are checked against the
per-source frame ranges in meta/assembly.json. Needed by StratifiedMixSampler (data_loader.py)."""

import argparse
import concurrent.futures as cf
import json
import pathlib

import pyarrow.parquet as pq


def _one(args):
    root, ep, chunk_size = args
    p = pathlib.Path(root) / f"data/chunk-{ep // chunk_size:03d}/episode_{ep:06d}.parquet"
    t = pq.ParquetFile(p).read_row_group(0, columns=["source", "quality", "domain"])
    return ep, str(t["source"][0]), int(t["quality"][0].as_py()), int(t["domain"][0].as_py())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    root = pathlib.Path(a.repo_root)
    info = json.load(open(root / "meta/info.json"))
    eps = [json.loads(l) for l in open(root / "meta/episodes.jsonl")]
    eps.sort(key=lambda e: e["episode_index"])
    assert [e["episode_index"] for e in eps] == list(range(len(eps))), "episode indices must be 0..N-1"
    first = {}
    f0 = 0
    for e in eps:
        first[e["episode_index"]] = f0
        f0 += e["length"]
    assert f0 == info["total_frames"], (f0, info["total_frames"])
    with cf.ProcessPoolExecutor(a.workers) as ex:
        rows = dict((r[0], r) for r in ex.map(_one, [(str(root), e["episode_index"], info["chunks_size"]) for e in eps], chunksize=64))
    asm = json.load(open(root / "meta/assembly.json"))["sources"]
    for name, s in asm.items():  # episodes of a source are contiguous: check the range boundaries line up
        assert first[s["first_new_episode"]] == s["first_frame"], (name, first[s["first_new_episode"]], s["first_frame"])
        assert rows[s["first_new_episode"]][1] == name, (name, rows[s["first_new_episode"]])
    out = root / "meta/episode_strata.jsonl"
    with open(out, "w") as f:
        for e in eps:
            _, src, q, d = rows[e["episode_index"]]
            f.write(json.dumps({"episode_index": e["episode_index"], "source": src, "quality": q, "domain": d,
                                "first_frame": first[e["episode_index"]], "length": e["length"]}) + "\n")
    agg = {}
    for e in eps:
        _, src, q, d = rows[e["episode_index"]]
        k = f"{src}/{'success' if q else 'failure'}/{'sim' if d else 'real'}"
        agg.setdefault(k, [0, 0]); agg[k][0] += 1; agg[k][1] += e["length"]
    print(f"wrote {out} ({len(eps)} episodes)")
    for k, (n, fr) in sorted(agg.items()):
        print(f"  {k:40s} eps={n:6d} frames={fr}")


if __name__ == "__main__":
    main()
