# Multi-host (multi-node) pi05 training on the TRI DGX nodes — MULTIHOST.md (2026-10-09)

Prototype of data-parallel JAX multi-process training for the openpi fork (TRI-ML/dsrl_openpi `isaac-lbm-yam`) so the
ABC-130k + MolmoAct2 pi05 pretrain can run on 2–3 DGX nodes instead of one. Prototyped on dgx03 + dgx06 while the
single-node run was being prepared on dgx04. **Nothing on dgx04 was changed.** Scripts/patch/diff/logs are mirrored at
`/tmp/pi05_multihost/` on Puget and live in `/raid/aliangdw/pi05_abc_molmo/scripts/` on dgx03 and dgx06.

## How it works
One JAX process per node, 8 local GPUs each. `jax.distributed.initialize` (env `OPENPI_NPROC`, `OPENPI_PROC_ID`,
`OPENPI_COORD_ADDR`, `OPENPI_LOCAL_DEVICES`) runs first in `scripts/train.py`; after that `jax.devices()` spans all nodes
and openpi's unchanged mesh (`make_mesh(fsdp_devices=8)` → shape `(num_nodes, 8)`) puts each node's 8 GPUs on one FSDP
row with the nodes along the batch axis. Each process loads a DISJOINT 1/N slice of the merged LeRobot repo (torch
`DistributedSampler(rank=process_index)`, reshuffled per epoch) and `jax.make_array_from_process_local_data` assembles
the global batch (global batch = `--batch-size`, per node = batch/N). Model + optimizer state stay FSDP-sharded over each
node's 8 GPUs exactly as in the single-node run; gradient all-reduce across nodes goes through XLA/NCCL over InfiniBand
(NCCL picked `NET/IB` automatically; OOB over `bond0`). With `OPENPI_NPROC` unset the code path is unchanged.

## Patch (idempotent): `scripts/apply_multihost.py <repo>` — unified diff: `multihost.diff` (181 lines, 3 files)
- `scripts/train.py`: `maybe_init_distributed()` before any JAX call; wandb only on process 0 (others `mode=disabled`);
  tqdm/step prints only on process 0; the camera_views sanity image and a `[multihost-check] process i local actions …
  sha1 …` line are built from the process's addressable shard (global arrays are not fully addressable on one host).
- `src/openpi/training/checkpoints.py`: only process 0 wipes/creates the checkpoint dir, then a `sync_global_devices`
  barrier, then every process polls `mkdir(exist_ok)`/`exists()` until the dir is visible (NFS negative-lookup cache bit
  us: the first attempt died with "Checkpoint root directory … does not exist" on process 1). The checkpoint dir MUST be
  on a filesystem shared by all nodes — orbax writes per-process shards into one directory and process 0 finalizes.
  `/home/aliangdw/...` (NFS, ~190 MB/s single-stream) worked; /datasets is the same NFS server. Not /raid.
- `src/openpi/training/data_loader.py`: the `NotImplementedError` for `jax.process_count() > 1` is replaced by a
  `DistributedSampler` (+ `set_epoch` on every pass) so processes see disjoint shards.
Coexists with dgx04's `apply_patches.py` (telemetry, config, lerobot ts-check skip) — both are applied on dgx03/dgx06.

## Prerequisites per node
Identical `/raid/aliangdw/pi05_abc_molmo` on every node: `code/dsrl_openpi` (+venv, both patch scripts applied),
`lerobot_home/local/<repo>` assembled from the same `stage/` sources with the same `assemble_v21.py` arguments (episode ids
must match — verify `md5sum meta/episodes.jsonl meta/info.json` across nodes; identical for the tiny repo here),
`assets/<cfg>/local/<repo>/norm_stats.json`, `openpi_cache` (pi05_base), `hf_home` (arrow cache rebuilds in minutes if absent).
Staging node-to-node: `stage_dgx03.sh` (from dgx04 over IB, 6 parallel source streams) and `stage_dgx06.sh` (from dgx03,
follows dgx03's `staged_sources.txt`). Measured ~0.9–1.0 GB/s aggregate; **single rsync streams die when too many ssh
connections hit one node at once** (16 concurrent streams → 6 sources truncated; fixed with `stage_molmo_retry.sh` /
`stage_retry5.sh`: ≤4 streams per big source, retry until file counts match the source). Always compare per-source file
counts against the source node before assembling. Node-to-node ssh as aliangdw must work; a freshly power-cycled node
spends its first ~15 min in the Bright node-installer chroot (no /home, /raid, nvidia) — wait, then `mkdir /raid/aliangdw`
as root from the headnode.

## Launch / stop / resume
```
# on any DGX node, as aliangdw
NODES="03 06" CFG=pi05_yam_abc_mm2_h32 CKPT_BASE=/home/aliangdw/pi05_mh_ckpt \
  [XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true"] \
  bash /raid/aliangdw/pi05_abc_molmo/scripts/launch_multihost.sh <exp> [--batch-size 256 --data.repo-id local/<repo> ...]
# per node: tmux session mh, window <exp>_p<i>, log $W/logs/mh_<exp>_p<i>.log; process 0 = coordinator = first node
# resume: same command + --resume          (every process restores from the shared checkpoint dir)
# stop:   on each node  pkill -u aliangdw -f "exp-name <exp>"   (never pkill -f from inside an ssh command whose text
#         contains the pattern — it kills the ssh shell itself)
```
`mh_run.sh` is the per-node runner (sources env.sh, sets OPENPI_*/NCCL env, tees the log). `--batch-size` must be divisible
by N×8.

## Smoke results (dgx03 + dgx06, 2 nodes = 16×A100-80GB, tiny repo = ABC t075+t144+t130, 2,825 eps / 11.9M frames)
Full-parameter pi05 (no LoRA), action_horizon 32, FSDP 8 per node, 32 loader workers per node, wandb project `openpi`.

| run | nodes | global batch | median step (s) | samples/s | notes |
|---|---|---|---|---|---|
| bench base_b128 (dgx03) | 1 | 128 | 2.04 | 62.7 | steps 20→70 wall clock |
| bench xla_b128 (dgx03) | 1 | 128 | 1.92 | 66.7 | `--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true` |
| mh_tiny_b256 | 2 | 256 (128/node) | 2.13 | 120.0 | wandb g8liceaz; = 1.91× single node (96 % scaling) |
| mh_tiny_xla | 2 | 256 | 2.05 | 124.9 | wandb zogekqkn; XLA flags on both nodes |

- All 16 devices in one mesh (`process 0/2 local_devices=8 global_devices=16`), NCCL `Using network IB`.
- Data disjointness: `[multihost-check]` sha1 of the first local batch differs (p0 `f28abccbf6d0`, p1 `29d766fc55f5`);
  loader wait 0.02–0.06 s/step (data loading is not the bottleneck at 2 nodes).
- Loss decreases normally (0.044 → 0.024 over 100 steps at batch 256; same trajectory as the single-node bench).
- Checkpoint to shared NFS: blocking part 15 s, full async save 79–114 s for the 42 GB orbax dir (training continues
  during the async part); `--resume` on both hosts restored step 119 from `/home/aliangdw/pi05_mh_ckpt/...` in 66 s
  and continued 120→139, saved 139, pruned 119 (max_to_keep=1, keep_period kept 100).
- GPU memory: nvidia-smi shows the XLA preallocation (~74.8 GB with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`); real peak at
  batch 64/node was 36.8 GB in the dgx04 smoke, so 128/node fits with headroom; the telemetry `perf/gpu_mem_max_gb`
  field (6.3) does not measure device memory — ignore it.
- JIT + init ≈ 5 min per start (dgx03 arrow cache warm); first start on a node with a cold HF arrow cache adds ~2 min.

## Single-node knobs measured (dgx03, batch 128)
- XLA flags (latency-hiding scheduler + Triton GEMM): 2.04 → 1.92 s/step (−6 %). Safe; recommended for all runs.
- Orbax async checkpointing is ALREADY on (orbax 0.11.13, `enable_async_checkpointing=True` by default): only ~15 s of a
  save blocks training; the "6 min" in the dgx04 smoke was the end-of-run `wait_until_finished`. Nothing to change.
- Dataloader: 32 workers/node, loader wait 0.02–0.06 s/step → not worth tuning now.
- Not possible on A100: bigger per-node batch (128/node is near the 80 GB limit); H100/p5 would be ~1.8× faster per node.

## Recommended switch plan for the dgx04 run (`pi05_full_yam_abc_mm2_h32_dgx04`) → 3 nodes dgx04 + dgx03 + dgx06
Expected: 3 × 8 GPUs, global batch 384 (128/node), step ≈ 2.1–2.2 s (scaling held at 96 % for 2 nodes), ≈ 175–180 samples/s
→ **3× the single-node throughput** (single node: 128 / 2.04 s = 63 samples/s).

| option | steps | samples | wall clock (3 nodes) | vs single node 200k@128 (≈ 113 h + saves) |
|---|---|---|---|---|
| A. same samples (25.6M) | 66.7k @ 384 | 25.6M | ≈ 40 h | 2.8× faster |
| B. 100k @ 384 | 100k | 38.4M (0.75× WAM's 51M) | ≈ 60 h | 1.5× the samples in half the time |
| C. keep 200k | 200k @ 384 | 76.8M (1.5× WAM) | ≈ 120 h | same time, 3× the samples |

Procedure (not performed):
1. **Data**: dgx03/dgx06 have all 202 `stage/` sources (dgx03 verified file-for-file against dgx04: 981,386 files; dgx06
   still staging, see status below). Build the SAME train/val repos as dgx04 on each node: copy from dgx04
   `$W/sources_<exp>.txt`, `lerobot_home/local/abc_mm2_train_v21/meta/assembly.json` and the molmo val-id exclusion file it
   references, re-run `assemble_v21.py` with identical arguments (`--skip` camera-less tasks, `--exclude` molmo val ids),
   then `md5sum meta/episodes.jsonl meta/info.json` must match dgx04's. Copy `assets/pi05_yam_abc_mm2_h32/local/abc_mm2_train_v21/norm_stats.json`.
2. **Code**: on dgx04 run `scripts/apply_multihost.py $W/code/dsrl_openpi` (idempotent; dgx03/dgx06 already have it) and copy
   `mh_run.sh` + `launch_multihost.sh` into `$W/scripts/`.
3. **Checkpoint on shared FS**: stop the dgx04 run at a checkpoint boundary; `rsync -a $W/checkpoints/pi05_yam_abc_mm2_h32/<exp>/
   /home/aliangdw/pi05_mh_ckpt/pi05_yam_abc_mm2_h32/<exp>/` (42 GB, ~4 min over NFS). Keep the exp name.
4. **Launch** from dgx04 (process 0 = coordinator):
   `NODES="04 03 06" CFG=pi05_yam_abc_mm2_h32 CKPT_BASE=/home/aliangdw/pi05_mh_ckpt XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true" bash $W/scripts/launch_multihost.sh <exp> --batch-size 384 --num-train-steps <N> --resume`
   `--resume` restores params + optimizer state + step from the single-node checkpoint onto the 24-device mesh (orbax
   restores by global shape into the target sharding; the step counter and LR schedule continue; only the batch grows).
   If the cross-mesh resume fails, warm-start instead: `--weight-loader.params-path /home/aliangdw/pi05_mh_ckpt/pi05_yam_abc_mm2_h32/<exp>/<step>/params`
   (openpi `CheckpointWeightLoader`) with `--overwrite`, which restarts the LR schedule from warmup.
5. Checkpoint cadence stays 10k; on NFS each save is ~15 s blocking + ~2 min async. Delete `pi05_mh_ckpt/.../mh_tiny_*`
   smoke dirs first (done at the end of this prototype) — /home is 99 % full cluster-wide, keep only what is needed.
6. Validation: the dgx04 `eval_ckpt.py`/`eval_watcher.sh` can run on a 4th node or on dgx04 GPU-free CPU as before; it
   reads the shared checkpoint dir.

Caveats: a hung node kills the whole job (JAX coordination service aborts all processes) — resume from the last 10k
checkpoint; the wandb run id is stored in the shared ckpt dir so the run continues under the same id. Keep
`NCCL_DEBUG=WARN` after the first launch (INFO is very chatty).

## Status at hand-off (15:20 PT)
- dgx03: all 202 sources staged + verified; venv/code patched; tiny repo; smoke + resume + XLA runs done; GPUs idle.
- dgx06: power-cycled 13:03 (was DOWN/hung), back ~13:30; staging from dgx03 in tmux `mh:stage` (72/202 sources, 3.5 TB
  at 15:15, ~0.75 GB/s → all sources ≈ 16:45 PT); venv/code patched; tiny repo identical (md5) to dgx03.
- Smoke checkpoints under /home/aliangdw/pi05_mh_ckpt/pi05_yam_abc_mm2_h32/mh_tiny_* removed after the last test.
