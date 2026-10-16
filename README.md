# Metric-sensitivity study for X-ray scattering VAE reconstructions

Companion codebase for [`plans/metric_study.md`](plans/metric_study.md): which
candidate reconstruction metrics actually detect a VAE dropping fine rings and
sparse Bragg peaks, and whether the answer changes between 512x512 training
resolution and native Pilatus 2M frames (1679x1475, 172um pixels, module gaps).
See the plan for full context, scope, and open questions.

This package is intentionally separate from the training repo
(`cvae_model_for_lse`) — no training code is modified. It imports `vae.py` /
`YParams.py` from there (see `metric_study/reconstruct.py`) rather than
duplicating the model.

## Status — what's real vs. not

- **Implemented and unit-tested**: masks, downsampling, preprocessing,
  geometry-free ring/peak detection, all 8 degradations, all 13 metrics,
  chunked processing with atomic/resumable output. 89 tests, `pytest tests/ -q`,
  ~50s on a laptop CPU, all passing as of this writing.
- **Run once against the real sample data** (`~/Documents/sample_data`, 44
  unique Pilatus 2M frames) with `--limit 1`: manifest/dedup logic confirmed
  correct (44 unique / 56 duplicates, matching the manual analysis in the plan),
  one full image processed correctly end to end, resume-skip logic confirmed.
  **Not yet run to completion on all 44 images** — see Timing below for why,
  and Quickstart for how to do it yourself.
- **Not yet implemented**: `reconstruct` CLI subcommand (needs a trained
  checkpoint — plan §9 Q1), `benchmark` (needs a GPU node), `analyze` (needs a
  production run to analyze), `atlas` export for Embedding Atlas (plan §11).
- **Not tested at all**: the SLURM scripts in `slurm/` — there is no scheduler,
  GPU, or NERSC allocation on this development machine. They are written to
  match the training repo's script conventions but are unverified templates;
  in particular, `run_array.sh`'s per-rank chunk sharding is a placeholder
  (`cli.py run` currently processes chunks sequentially in one process — see
  the script's comments and plan §6.2 for the intended multi-rank design).

## Timing (measured on this machine: CPU only, no GPU)

Per image, with the `dry_run` config section's degradation grid (20
clean+degraded variants x 2 resolutions x 13 metrics): **~100-110 seconds**.
Breakdown from a real profiling run on `23998.jpg`:

| Stage | Time |
|---|---|
| Feature extraction (ridge/spot detection), native, once per image | ~5.3s |
| Feature extraction, 512, once per image | ~0.4s |
| `ssim`, per call, native / 512 | ~1.8s / ~0.18s |
| `ms_ssim`, per call, native / 512 | ~2.4s / ~0.24s |
| every other metric combined, per call, native / 512 | ~0.07s / ~0.01s |

`ssim`/`ms_ssim` at native resolution dominate (~4.3s x 20 variants = ~85s of
the ~105s total) — this is CPU-bound conv2d work with no GPU to offload to, and
matches the plan §3 flag that these are the costliest metrics and should run on
a subsample, not the full 1.5M-image grid, in production. **At 44 images this
is ~75-80 minutes total.** This is a real, useful data point for plan §7 (which
says its estimates should be replaced by measured numbers) — it is NOT
representative of the GPU-based production runtime, and is annotated as CPU
timing everywhere it's cited.

To run the full dry run faster:
- Drop `ssim`/`ms_ssim` from `metrics.names` in `configs/metric_study.yaml`'s
  `dry_run` section for a smoke test in well under a minute total.
- Or use `--limit N` (see below) to process only the first N unique images.
- Or run on any machine with a CUDA GPU — `metric_study/metrics/structural.py`
  is plain `torch`, so it moves to GPU automatically with `.to("cuda")` tensors
  (not yet wired into the CLI's device selection, which is currently CPU-only —
  a small follow-up if you want to test this locally on a GPU machine).

## Setup

```bash
cd /Users/aianalyticscomputing/Documents/Coding/metric_study
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

No third-party GitHub packages are installed (see the plan's
focal-frequency-loss note): `metrics/freq.py` is a vendored reimplementation of
the published formula, not the authors' code.

## Quickstart

```bash
source .venv/bin/activate

# 1. Run the tests (should all pass, ~50s)
pytest tests/ -q

# 2. Quick smoke test: dry run on just 1 image (~100s, see Timing above)
python -m metric_study.cli dry-run \
    --config configs/metric_study.yaml --section dry_run \
    --output-dir outputs/dry_run_smoke --limit 1

# 3. Full dry run: all 44 unique sample images (~75-80 min on CPU; run it in the
#    background and check progress via the log file or `status`, below)
nohup python -m metric_study.cli dry-run \
    --config configs/metric_study.yaml --section dry_run \
    --output-dir outputs/dry_run > /dev/null 2>&1 &

# check progress at any time (safe to run repeatedly):
tail -f outputs/dry_run/dry_run.log
python -m metric_study.cli status --output-dir outputs/dry_run
```

What a successful full dry run should show in `dry_run.log`:
- `Manifest: 100 files, 44 unique by content hash, 56 duplicates, 0 unreadable`
- 5 `Chunk N done: ... rows written, 0 images skipped` lines (chunk_size=10 in
  the `dry_run` config section, so 44 unique images -> 5 chunks)
- `Resume check (re-processing chunk 0): OK (skipped, .done honored)` — this
  confirms the atomic-write + `.done`-marker resume logic (plan §6.3) actually
  works, not just that it's written to look like it does
- `DRY RUN COMPLETE: N metric rows, 0 images skipped`

If you kill it partway through (`pkill -f "metric_study.cli dry-run"`) and
re-run the same command, already-finished chunks are skipped automatically —
that's the resume behavior, not a bug.

## Other CLI commands

```bash
# Build a manifest (scan + content-hash dedup + read check) of a data directory
python -m metric_study.cli manifest --data-dir <dir> --output-dir <out>

# Run the full pipeline (uses/creates a manifest under --output-dir)
python -m metric_study.cli run --config configs/metric_study.yaml --section default --output-dir <out> [--limit N]

# Check progress of any chunked run
python -m metric_study.cli status --output-dir <out>
```

## Layout

See `plans/metric_study.md` §5.2 for the intended layout and rationale; briefly:

```
metric_study/
  config.py        # YParams-style YAML config + validation + config hash
  io.py             # image reading, never raises
  manifest.py       # scan + content-hash dedup -> manifest.parquet
  masks.py          # static Pilatus 2M gap mask + data-driven beamstop/dead/hot masks
  preprocess.py     # log + percentile-clip transform (the current training transform)
  downsample.py     # cv2 INTER_AREA, mask-normalized, matching the training loader's resize
  features.py       # geometry-free ring (Hessian ridge) and peak (LoG blob) detection
  degradations.py   # the 8 controlled corruptions + severity registry
  metrics/          # 13 candidate metrics behind one common signature
  reconstruct.py    # loads a trained checkpoint via the training repo's vae.py
  pipeline.py       # chunked runner: read -> mask -> degrade -> downsample -> metrics -> parquet
  cli.py            # manifest | run | status | dry-run subcommands
tests/              # 89 tests: masks, downsampling, preprocessing, features,
                     # degradations, metrics (optimum/mask-invariance/batching per
                     # metric), config, io, manifest, end-to-end pipeline
configs/metric_study.yaml   # default + dry_run sections
slurm/              # UNTESTED templates, see Status above
plans/metric_study.md       # the full study plan
```

## Known limitations / open questions

Tracked in `plans/metric_study.md` §9. The two that most affect this code:

- **Q2/Q3 (production data format & export transform)**: the sample data is
  8-bit tone-mapped JPEG with no recoverable linear-count information, so
  Poisson-family metrics and "counts-preserving" downsampling are only
  approximate (`pseudo_count=True`) until either the export transform's exact
  parameters or a raw-frame subsample are available.
- **Q1 (training repo checkpoints)**: no trained checkpoint exists on this
  machine yet, so `reconstruct.py` and the "real reconstructions" degradation
  case (plan §2) are implemented but unexercised.
