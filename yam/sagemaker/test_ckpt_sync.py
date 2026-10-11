"""Regression test for yam/sagemaker/ckpt_sync.sh (the SageMaker marker-file bug).

Simulates what killed the 2026-10-10 job: `*.sagemaker-uploaded` zero-byte markers inside orbax's in-progress tmp dirs
(and inside a finished step). Asserts the syncer (a) never copies in-progress `*.orbax-checkpoint-tmp-*` dirs,
(b) copies finished steps without the markers, (c) writes `.complete` last, and (d) the synced copy restores with orbax.
Run: <venv>/bin/python yam/sagemaker/test_ckpt_sync.py   (jax + orbax from the repo venv; CPU only)
"""
import json, os, pathlib, subprocess, sys, tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, numpy as np, orbax.checkpoint as ocp  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SYNC = HERE / "ckpt_sync.sh"


def save_steps(root: pathlib.Path, steps):
    mngr = ocp.CheckpointManager(root, item_handlers={"params": ocp.PyTreeCheckpointHandler()},
                                 options=ocp.CheckpointManagerOptions(max_to_keep=10, create=True))
    for st in steps:
        mngr.save(st, args=ocp.args.Composite(params=ocp.args.PyTreeSave({"w": np.full((4, 3), float(st)), "b": np.arange(3.0)})))
    mngr.wait_until_finished(); mngr.close()


def touch_markers(d: pathlib.Path):
    n = 0
    for p in list(d.rglob("*")):
        if p.is_file() and not p.name.endswith(".sagemaker-uploaded"):
            (p.parent / (p.name + ".sagemaker-uploaded")).write_bytes(b""); n += 1
    return n


def main():
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td); src = td / "ckpt" / "cfg" / "exp"; dst = td / "s3mirror"
        save_steps(src, [20, 40])
        (src / "wandb_id.txt").write_text("abc123")
        # an in-progress step: orbax tmp dir layout + markers (what SageMaker's agent does)
        tmp = src / "60.orbax-checkpoint-tmp-0"; (tmp / "params.orbax-checkpoint-tmp-1" / "array_metadatas").mkdir(parents=True)
        (tmp / "params.orbax-checkpoint-tmp-1" / "array_metadatas" / "process_0").write_text("{}")
        touch_markers(tmp)
        # a finished step polluted with markers too (the syncer must drop them so orbax can read the copy)
        n_markers = touch_markers(src / "40")
        # a finished step whose item is still being finalized (tmp inside) must be skipped
        half = src / "80"; (half / "params.orbax-checkpoint-tmp-3").mkdir(parents=True); (half / "_CHECKPOINT_METADATA").write_text("{}")

        r = subprocess.run(["bash", str(SYNC), str(src), str(dst), "--once"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        synced = sorted(p.name for p in dst.iterdir())
        assert synced == ["20", "40", "wandb_id.txt"], synced
        assert not list(dst.rglob("*.sagemaker-uploaded")), "markers leaked into the mirror"
        assert not list(dst.rglob("*orbax-checkpoint-tmp*")), "in-progress tmp dir leaked"
        assert (dst / "40" / ".complete").exists() and (dst / "20" / ".complete").exists()
        assert (src / "40" / ".synced").exists() and not (src / "80" / ".synced").exists()
        # re-running must be a no-op (idempotent)
        r2 = subprocess.run(["bash", str(SYNC), str(src), str(dst), "--once"], capture_output=True, text=True)
        assert "syncing step" not in r2.stdout, r2.stdout
        # the mirror restores with orbax (this is exactly what failed with markers in place)
        restored = ocp.PyTreeCheckpointer().restore(dst / "40" / "params")
        assert float(restored["w"][0, 0]) == 40.0 and restored["b"].shape == (3,)
        # (the production failure was at SAVE-finalize: orbax's array_metadata_store json-decodes every file under
        #  array_metadatas/, including a 0-byte marker; restore of a small tree may tolerate it, so we only assert the mirror is clean)
        bug_reproduced = None
        # resume selection = highest numeric step with a .complete sentinel (same rule as sm_entry.sh)
        latest = max(int(p.parent.name) for p in dst.glob("*/.complete"))
        assert latest == 40, latest
        print(json.dumps({"synced": synced, "markers_in_source_step_40": n_markers,                           "orbax_reads_mirror": True, "latest_complete": latest}))
        print("OK")


if __name__ == "__main__":
    main()
