# Plan: Metric-sensitivity study for X-ray scattering VAE reconstructions (512×512 vs. native Pilatus 2M)

> This is the plan-mode draft. On approval it is copied verbatim to
> `~/Documents/Coding/cvae_model_for_lse/plans/metric_study.md`. No training code is modified.
>
> **Scope (per your review):**
> - Nothing that needs calibration geometry: no azimuthal integration, I(q) or W1-on-I(q) metrics,
>   pyFAI or beam center.
> - Downsampling is one simple **OpenCV** test.
> - Preprocessing is the current **log transform with percentile clipping**.
> - A section on **Embedding Atlas** covers latent-space browsing with side-by-side original and
>   reconstruction views (§11).

## 0. Context and exploration findings

**Why.** The VAE objective (MSE + KL) is dominated by bright, broad rings and barely penalizes missed
fine rings and sparse Bragg peaks. Before changing the loss, we measure which candidate metrics detect
those failures, and whether the answer changes between 512×512 and native frames. The design targets a
later run on ~1.5 M images on NERSC Perlmutter (SLURM).

### 0.1 Training repo: `~/Documents/Coding/cvae_model_for_lse/train/`

| What | Where | Relevance to the study |
|---|---|---|
| Loader | `data_loader.py:56-63`: `Grayscale()` → `transforms.Resize([512,512])` → `ToTensor()` (range [0,1]). Recursive glob of `.jpg/.png/.tif…` at `:69-70`. 80/20 `random_split` with no fixed seed at `:88`. `__getitem__` returns `(None, None)` on error at `:147-149`, which would crash default collate. | **The current downsampling is PIL bilinear with antialias, applied as a direct anisotropic resize** (rows ÷3.28, cols ÷2.88; no crop or pad). **There is no log or percentile step in code**, so that transform must happen upstream, when the PNG/JPEG files are exported (Q3). |
| Model | `vae.py`: `ConvVAE` with 6 stride-2 convs, window attention, `fc_mu`/`fc_logvar`, and a sigmoid decoder. Built with `simple_autoencoder(params)` (`:135`). Requires H and W divisible by 2⁶ and by window 8. | The model is 512-only as configured. The real-reconstruction case uses `encode` → `mu` → `decode` (deterministic, no sampling). |
| Loss | `training_2.py:181-217` `rotation_invariant_vae_loss`: `mean MSE(recon,x)` + `mean MSE(recon_rot,x_rot)` + β·mean KL (β=0.5) + λ_rot·MSE(μ, μ_rot) (λ=5). **No mask**: gaps and beamstop count in the MSE. | The literal baseline is **unmasked mean MSE in exported [0,1] space**. The study reports it alongside the masked version. |
| Augmentation | `training_2.py:220-250` rotates ±180° about the image center, with hflip p=0.5 | Not part of the metric study. Noted because the gaps and beamstop rotate with the image. |
| Checkpoints | `save_checkpoint` / `save_best_checkpoint` (`:390-407`) write `{'model_state_dict', …}` to `checkpoint_dir` = `/pscratch/sd/m/monika/2node_test_vae_rotate/github_push/latent_{latent_dim}/checkpoints/` (`best.ckpt`, `epoch_XXX.ckpt`) | None are local. We need read access on NERSC (Q1). |
| Config / conventions | `YParams.py` loads a named section of a YAML file (ruamel.yaml, `&default` anchors). CLI is argparse. `logging` with one logger per rank (`training_2.py:41-55`). Google-style `Args:` docstrings. **No type hints, no tests.** ruff line-length 120, Python 3.10 (`pyproject.toml`). | The study uses the same YParams-style YAML, argparse, logging and ruff settings. It **adds** type hints and a `tests/` directory; that is additive and does not change existing code. |
| SLURM | `VAE_gpu_500_regular_512.sh`: `-A als -C gpu`, 4 tasks and 4 GPUs per node, `srun`, conda env `/pscratch/sd/m/monika/conda/envs/mlex_pytorch_ae` | The job-array scripts reuse this header style. The study gets its **own new env** (§5.1). |
| Data | `image_dir: /pscratch/sd/m/monika/bl733_data_png_test/` (`default_500.yaml:22`) | Production inputs are probably exported **PNGs** from BL 7.3.3. The samples are JPEG (Q2). |

### 0.2 Sample data: `~/Documents/sample_data`

| Item | Finding | Consequence |
|---|---|---|
| Files | 100 JPEGs, **only 44 unique** by MD5. 27938–28000 cycle through ~7 frames. | The manifest deduplicates by content hash (Q4). Duplicates also leak across the loader's random 80/20 split. |
| Format | 8-bit grayscale (`L`), ≈ quality 90, **no metadata or masks** | Values are already log-transformed and clipped (Q3). Linear counts are not available from these files. |
| Shape | 1679 rows × 1475 cols in all files (the Pilatus 2M layout) | Native resolution confirmed |
| Gaps | Dark cols 487–493 and 981–987 (7 px). Rows at the canonical 17-px gaps (195+212k…211+212k), which look ~21 px wide because of JPEG ringing. 3×8 modules of 487×195. | Static gap mask **dilated by 2 px** (a detector property, no calibration needed) |
| Beamstop and more | A vertical beamstop arm from the bottom edge. Some frames have a horizon or step in the lower third (GI?). 31 always-dark pixels. | Beamstop and dead-pixel masks are data-driven per mask group (§6.6) |
| Classes seen | (a) broad bright: 23976, 23984, 23986, 23992. (b) fine/weak rings: 23998 (calibrant-like), 23993–24000, 23973. (c) peaks/texture: 23989, 23990 (GI rods and spots). Low-signal: 27938–27944, 29968–29974. | All strata exist in the dev set (Q5) |
| Feature widths | Fine rings in 23998 have FWHM ≈ 4–7 native px | At 512 this is ≈ 1.2–2.1 px (row factor 3.28, col factor 2.88) |

---

## 1. Data and preprocessing

### 1.1 Intensity domains
The files hold the **exported representation** `x = round(255 · norm(clip(log(I+ε), P_lo, P_hi)))`,
with the parameters to be confirmed (Q3). The study works with two domains:
- **Model space** `x/255 ∈ [0,1]`: exactly what the VAE sees and reconstructs. All "model-space"
  metrics use it, including the literal training MSE.
- **Pseudo-linear space** `Ĩ = exp(L·x/255) − 1`, where L is the log dynamic range of the export
  (from Q3, or a nominal value). It is used only to apply degradations physically (removing intensity,
  blur, Poisson noise) and for the "linear" metrics. It is re-encoded with the same transform, including
  8-bit quantization, so degraded images look like real exports.
- **If raw frames are available for a subsample (Q2),** the pipeline runs the true export transform
  itself (`preprocess.py`). The pseudo-linear approximation is then replaced by real linear counts for
  that subsample. This also enables the **clip ablation**, with vs. without the upper percentile clip:
  a high clip flattens Bragg peaks before any loss sees them. That ablation cannot be done from exported
  files alone.
- Per-image `saturated_fraction` (pixels at 255) is recorded as a proxy for how much the clip removed.

### 1.2 Evaluation sets (stratified)
- **Classes:** (a) broad bright rings, (b) fine or weak rings, (c) sparse peaks or texture (incl. GI),
  plus (x) low-signal, which is recorded but excluded from sensitivity analysis. Labels are multi-label
  with one primary label.
- **Dev set:** the 44 unique frames, hand-labeled in `data/labels/dev_labels.csv` with a small labeling
  notebook.
- **At scale:** the automatic classifier in §6.5 assigns labels.

### 1.3 Downsampling: a single OpenCV test
- **Study downsampler.** It matches the training geometry: a direct 1679×1475 → 512×512 resize with the
  same anisotropic aspect, so reconstructions and metrics live in the model's own frame:
  ```python
  num = cv2.resize(x * m, (512, 512), interpolation=cv2.INTER_AREA)   # m: float32 valid mask
  cov = cv2.resize(m,     (512, 512), interpolation=cv2.INTER_AREA)   # valid fraction per output pixel
  x512  = np.where(cov > 0, num / cov, 0)                              # mask-normalized area mean
  m512  = cov >= tau                                                   # tau = 0.9 (config)
  ```
  - `INTER_AREA` does exact fractional-area averaging, so `x512·cov` × (1679·1475/512²) preserves the
    valid-pixel sum. That is unit-tested, and in pseudo-linear space it is what keeps Poisson statistics
    meaningful.
  - Mask normalization stops gap zeros from darkening neighboring outputs.
  - `m512` removes output pixels that are mostly gap or beamstop.
- **One-time consistency test** on the 44 dev frames: `cv2 INTER_AREA` vs. the training loader's
  `transforms.Resize([512,512])` (PIL bilinear + antialias). We report the per-image MAE, the max
  difference, and the difference restricted to fine-ring and spot pixels. If the two agree closely,
  conclusions from the study transfer to the training loader. If they do not, that is itself a finding
  about the loader. No other downsampling methods are compared.
- It runs on CPU in DataLoader workers (≈2–4 ms/frame, `cv2.setNumThreads(1)` per worker).

## 2. Controlled degradations

- Applied **at native resolution** in pseudo-linear space (or linear, where raw frames exist), inside the
  valid mask.
- Then re-encoded to model space and downsampled with §1.3, so both resolutions see the same corruption.
- Each degradation is a pure function `degrade(img, mask, features, severity, rng) -> img'`.

**Geometry-free feature decomposition** (on GPU, computed once per target):
- **Background B:** a gap-inpainted (normalized convolution) large-scale morphological opening and
  median.
- **Ridges (rings):** a multi-scale Hessian ridge filter at σ ∈ {1, 1.5, 2, 3, 4, 6, 8} px. The best
  scale gives each ridge's width (FWHM ≈ 2.355σ).
  - "fine" means width ≤ w_fine (8 px);
  - "broad" means width ≥ w_broad (12 px) with high excess intensity.
- **Spots:** small-scale LoG plus local maxima of I − B above kσ. Compact components (1–400 px, low
  elongation) that are not on a long ridge.

| Degradation | Parameterization | Severity grid (config) |
|---|---|---|
| Fine-ring attenuation | `I' = I − a·(I − B)` on one fine ridge component (narrowest, weakest and median tested separately) | a ∈ {0, .1, …, 1} |
| Bragg-peak removal | The same operator on a random fraction f of spots, with local B from a 3-px surrounding ring | f ∈ {.1, .25, .5, 1} × a ∈ {.5, 1} |
| Gaussian blur | Mask-normalized `(G*(I⊙M))/(G*M)` | σ ∈ {0.5, 1, 1.5, 2, 3, 4, 6} px |
| Feature displacement (stand-in for radial shift; no beam center) | Move one ring or spot component δ px along its local normal (Hessian eigenvector). The vacated region is filled with B. | δ ∈ {0.5, 1, 2, 3, 5} px |
| Global scaling | `I' = s·I` | s ∈ {0.5, 0.8, 0.9, 1.1, 1.25, 2} |
| Additive offset | `I' = I + c·median(I_valid)` | c ∈ {0.01, 0.05, 0.1, 0.25} |
| Poisson resampling (noise floor) | `Poisson(λ·I)/λ` at mean counts/pixel ∈ {1, 10, 100, 1000}. Flagged as pseudo-count unless raw frames are used. | 4 levels × 3 seeds |
| Broad-ring damage (control) | The attenuation operator on the brightest broad ridge | a ∈ {0, .1, …, 1} |

**Matched severity:**
- Each attenuation is also recorded as ΔI_removed and ΔMSE_model.
- This allows fine vs. broad comparison at equal physical damage *and* at equal MSE damage, which is the
  central argument.

**Degrade after downsampling:** a small ablation only (dev set plus ~2 k images). It shows whether
corruption commutes with downsampling. It is not part of the main grid.

**Real reconstructions:**
- The pipeline loads `best.ckpt` (and optionally 2–3 epoch checkpoints) through
  `vae.simple_autoencoder(params)` with the original YAML and `latent_dim`.
- It encodes the loader-equivalent 512 input and decodes `mu`, then scores it with the same metric suite
  as `degradation="model:<ckpt>"`. The pred is 512; the native-resolution comparison upsamples it with
  `cv2.INTER_LINEAR`.
- It also stores `mu` for §11.
- This is the real-world check: do metrics that react to synthetic fine-ring loss also rank real
  reconstructions the way a human would? The check is a 100-image blind visual ranking.

## 3. Metrics

Common signature, in `metric_study/metrics/`:
```python
def metric(pred: Tensor, target: Tensor, mask: Tensor, geometry: None = None, **cfg) -> Tensor  # (B,)
```
- `geometry` is kept for future I(q) metrics and is unused now.
- Inputs are `(B, H, W)` float32 with a bool mask, and the same code runs at both resolutions.
- `masked_fill_like(pred, target, mask)` sets `pred := target` on invalid pixels, so local and FFT
  metrics are exactly insensitive to masked pixels.

| # | Metric | Domain and notes | Native? | 1.5 M? |
|---|---|---|---|---|
| 1 | MSE, linear | Pseudo-linear. **Plus the literal training loss:** unmasked mean MSE in model space, and its masked version. | yes | yes |
| 2 | MSE and L1 on `log(I+ε)`, and on `asinh(I/s)` | Pseudo-linear → transformed. Log-MSE ≈ model-space MSE up to the clip, which is reported as a check. | yes | yes |
| 3 | Poisson NLL (offset so optimum = 0) and Poisson χ² | Pseudo-linear. **Flagged as pseudo-count**; valid only on the raw-frame subsample. | yes | yes |
| 4 | SSIM and MS-SSIM | Model space. `pytorch_msssim` with Gaussian σ=1.5, win=11, K1=0.01, K2=0.03, data_range=1. Averaged over the mask eroded by win/2, with a per-scale downsampled mask. 5 scales need a min side > 160: OK at 512 (coarsest 32 px) and native. A win=7 variant runs at 512. | yes (~10–20 ms/img GPU) | subsample |
| 5 | −Pearson r | Model and pseudo-linear. Scale- and offset-invariant by design. | yes | yes |
| 6 | Focal frequency loss | Official `EndlessSora/focal-frequency-loss` (pinned commit), α=1, patch_factor ∈ {1, 4}, model space, after `masked_fill_like`. Native is padded to 1680×1536 for the FFT. | yes | subsample |
| 9 | Peak-weighted MSE | Model space. `W = 1 + β·S(target)`, where S is the §2 ridge-and-spot map of the target, recomputed on the fly. β ∈ {10, 100}. | yes | yes |
| — | ~~I(q) error, W1 on I(q)~~ | **Deferred.** They need calibration and can be added later behind the same signature. | — | — |

**Feasibility flags:**
- Poisson metrics are invalid on exported 8-bit files except as pseudo-counts.
- MS-SSIM and FFL are the costliest and run on the full grid only for the subsample.
- Nothing is infeasible at native resolution for evaluation.
- The model is 512-only, so native-resolution *training* with any loss would need architecture changes;
  that is out of scope. §4.4 measures cost at both sizes anyway.

## 4. Analysis (reads stored Parquet only, never recomputes metrics)
1. **Sensitivity curves:** per (metric, domain, degradation, resolution), with median and IQR, faceted
   by class.
2. **Normalized sensitivity:**
   - `d′ = (m(deg) − m(clean)) / σ_Poisson(m)`;
   - `R_fine/noise` and `R_peak/noise`;
   - `R_fine/broad` and `R_peak/broad` at matched ΔI and at matched ΔMSE_model, compared to MSE's own ratio.
3. **Resolution agreement:**
   - Spearman ρ between native and 512 across images, per metric × degradation × class, with bootstrap
     CIs;
   - a breakdown by native feature width (1–3, 3–6, 6–12, >12 px).
4. **Compute cost:** a separate `benchmark` job with fwd and fwd+bwd time and peak GPU memory
   (B ∈ {8, 32, 64}; 64 is the training batch), native and 512, on one A100.
5. **Per-class breakdowns** everywhere.
6. **Real reconstructions:** metric values on `model:<ckpt>` vs. the human visual ranking (Kendall τ).
   This also tests whether the unmasked training MSE is dominated by gaps and beamstop.

**Figures:**
- F1: sensitivity-curve grid;
- F2: heatmap of R_fine/broad and R_peak/broad;
- F3: Poisson d′;
- F4: native-vs-512 Spearman heatmap;
- F5: d′ vs. feature width;
- F6: cost vs. sensitivity;
- F7: example panels on dev frames;
- F8: metric–metric correlation;
- F9: real-reconstruction metric vs. human rank;
- F10: clip ablation (raw subsample only);
- F11: cv2 vs. torchvision resize differences.

**Summary table, one row per metric:**
- d′ fine and d′ peaks (native / 512);
- R_fine/broad and R_peak/broad (native / 512);
- d′ Poisson and invariances passed;
- min-over-class ρ native-vs-512;
- Kendall τ with the human ranking;
- ms and GB per batch fwd+bwd;
- differentiable (yes/no) and recommendation.

## 5. Engineering

### 5.1 Environments
- **Study env (new; the existing conda env is not reused):**
  - Python 3.10 to match the repo;
  - torch and torchvision (same major versions as training), opencv-python-headless, numpy, scipy,
    scikit-image, pillow, pyarrow, polars, pytorch-msssim, focal-frequency-loss (git commit),
    ruamel.yaml, pytest;
  - on NERSC, a venv on top of `module load pytorch` or a new conda env on `$CFS`, pinned in
    `envs/metric_study.lock`.
- **Visualization env (new, separate):** Python ≥ 3.11 with `embedding-atlas` (0.24 requires Python
  ≥ 3.11, pandas ≥ 3, pyarrow ≥ 24; §11). It is separate so its pins cannot disturb the study env.

### 5.2 Layout
Repo conventions are flat modules, YParams YAML, argparse, per-rank logging, Google-style docstrings and
ruff with line length 120. On top of those the study adds type hints and pytest. It lives in a new
subpackage, and nothing in `train/` is edited: `train/` modules are only imported, via `sys.path`, or via
`pip install -e train/` in the new env.
```
metric_study/
  metric_study.yaml   # YParams-style named sections: `default`, `dry_run`, `production`
  config.py           # YParams wrapper + validation + config hash
  io.py               # PNG/JPEG (PIL/cv2) now; raw frames (fabio/tifffile) if Q2 -> (img, meta, status)
  manifest.py         # scan -> manifest.parquet (path, size, file hash, content hash, shape, dtype, bit depth, mask_group)
  masks.py            # static Pilatus gap mask (+dilation) and data-driven beamstop/dead masks per group
  preprocess.py       # export transform (log + percentile clip) and its pseudo-inverse
  downsample.py       # cv2 INTER_AREA with mask normalization; torchvision-equivalence check
  features.py         # background, ridge and spot maps; class features; classifier
  degradations.py     # pure functions + registry; RNG keyed on (seed, image_id, deg, sev, rep)
  metrics/            # pixel.py, poisson.py, structural.py, freq.py, weighted.py
  reconstruct.py      # load train/vae.py checkpoint; mu + recon per image
  pipeline.py         # chunk runner: DataLoader -> GPU -> degrade -> downsample -> metrics -> Parquet
  atlas_export.py     # §11 Parquet for Embedding Atlas
  cli.py              # manifest | masks | features | run | reconstruct | benchmark | dry-run | status | analyze | atlas
  analysis/           # reads Parquet only
  slurm/              # run_array.sh etc. (header style from VAE_gpu_500_regular_512.sh)
  tests/
```

### 5.3 Tests
Tests use pytest with synthetic rings and spots plus 3 dev frames.
- **Every metric:**
  - identical inputs give the optimum;
  - masked-pixel changes have no effect (bit-exact);
  - batched results equal per-image results;
  - native and 512 shapes both run;
  - `gradcheck` passes where the metric is claimed to be differentiable.
- **Invariances:** Pearson is scale- and offset-invariant. Documented non-invariances are asserted as
  such.
- **Downsampling:**
  - the valid-pixel sum is preserved;
  - a constant image stays constant;
  - cov ∈ [0, 1];
  - masked regions stay masked;
  - on smooth images, cv2 and torchvision `Resize` agree within tolerance.
- **Preprocessing:** export → pseudo-inverse → export round-trips within 1 grey level. There is a golden
  test against the real export script once provided (Q3).
- **Degradations:**
  - severity 0 is the identity;
  - high-λ Poisson approaches the identity;
  - a=1 removes ≥ 95 % of the component's excess;
  - seeds are deterministic regardless of chunking.
- **Reconstruction:** loading a checkpoint and building from params matches `train/vae.py` shapes.
  `mu` is deterministic in eval mode.
- **Pipeline:**
  - kill and resume gives identical Parquet;
  - corrupt and wrong-shape files are skipped and recorded.

### 5.4 Reproducibility
- `metric_study.yaml` holds the seed, OpenCV flag, τ, export parameters, degradation grids, metric
  settings, class thresholds, chunk size, subsample sizes and checkpoint paths.
- Each output row has `config_hash` and a git SHA.
- The study uses its own fixed-seed split and dedup. The loader's unseeded `random_split` is not used.

## 6. Scaling to 1.5 M images

### 6.1 Input format and I/O
- **Expected format:** exported 8-bit PNG (training `image_dir`) or JPEG (samples), 1679×1475 (Q2).
- **Reading:** a streaming `Dataset` over manifest rows, decoded with `cv2.imread(..., IMREAD_UNCHANGED)`
  so 16-bit PNGs are also handled.
- A custom `collate_fn` drops failed samples. The repo's `(None, None)` return would crash default
  collate.

**Cache vs. on the fly:**

| | Native float32 | 512 float32 | 512 uint8 |
|---|---|---|---|
| per image | 9.9 MB | 1.05 MB | 0.26 MB |
| 1.5 M images | 14.9 TB | 1.6 TB | 0.4 TB |

Inputs are ~0.6 TB (JPEG) or ~1–2 TB (PNG).

**Decision: on the fly.** Resizing (~3 ms) is cheap next to decoding (~7–15 ms). The grid loops over
all degradations per loaded image, so each file is read once. Degraded images are regenerated from the
seed, never stored.

### 6.2 Parallelism (Perlmutter GPU: 4× A100, 64 cores; `-A als`)
- **Unit of work:** a chunk of 2,000 manifest rows (750 chunks).
- **SLURM:** a job array with one node per task and `srun --ntasks-per-node=4 --gpus-per-task=1`.
- **Per rank:** `DataLoader(num_workers=14, pin_memory, prefetch_factor=4, persistent_workers)`.
  - CPU workers decode and resize the clean image.
  - The GPU computes feature maps, degradations batched over severities, metrics, and VAE inference.
  - Degraded native images are copied back to pinned host memory for `cv2.resize` in a thread pool, to
    keep INTER_AREA exact.
- **Precomputed once:** gap mask, per-group masks, SSIM windows, Hessian and LoG kernels, and the cuFFT
  plan warmed on the first batch.

### 6.3 Fault tolerance
- **Chunk outputs:** written to `.tmp`, then renamed atomically, then marked with a `.done` file
  containing the config hash, row count and SHA.
- **Resume:** the same array is resubmitted, and chunks with a matching `.done` are skipped. A crash at
  900 k therefore loses ≤ 1 chunk.
- **SLURM flags:** `--requeue` and `--signal=B:USR1@180` for clean shutdown.
- **Skipped files:** every read and decode is wrapped, and failures go to `skipped/chunk=<id>.parquet`
  (path, stage, error).
- **Watchdog:** a chunk fails only if more than 5 % of its files are skipped.
- **Logging and status:** per-rank logs follow the repo's `setup_logger` style, plus JSON progress lines.
  A `status` subcommand reports done, missing and skipped chunks.

### 6.4 Outputs (Parquet, zstd)
- `manifest.parquet`: ~100 MB.
- `features.parquet`: 1.5 M rows, ~0.5 GB. Class features and labels, mask group, saturated fraction.
- `metrics.parquet`: wide, one row per (image_id, resolution, degradation, severity, rep), ~20 metric
  columns plus ΔI and ΔMSE.
  - Pass B: 30 k × ~70 × 2 ≈ 4.2 M rows ≈ 0.4 GB.
  - Pass A: 1.5 M × ~6 × 2 ≈ 18 M rows ≈ 1.5 GB.
- `latents/`: `mu` as float16, 1.5 M × latent_dim 512 ≈ 1.5 GB, with per-image reconstruction metrics.

### 6.5 Degradation budget and stratification
- **Pass A: all 1.5 M, cheap.**
  - Manifest, dedup, masks, features and class labels.
  - VAE reconstruction with metrics plus `mu`.
  - Baseline metrics:
    - native → 512 → native round-trip loss;
    - Poisson at 1 level;
    - fine-ring a=1 and peaks f=1 where detected.
- **Pass B: stratified subsample, full grid.**
  - Strata = class × mask group × intensity tercile.
  - ≥ 2,000 per class, a floor of 200 per group, ≈ 20–30 k total.
  - Plus the degrade-after-downsample ablation (~2 k images).
- **Subsample justification:**
  - Spearman SE ≈ (1−ρ²)/√(n−3). At n = 2,000 and ρ = 0.8, the 95 % CI is ±0.017.
  - The bootstrap CI of median d′ is ≈ 2.5 % of the IQR.
  - A group floor of 200 gives ±0.05 on ρ.
- **Optional Pass C:** the shortlisted metrics × key degradations on all 1.5 M.
- **Classifier (geometry-free).**
  - Features:
    - ridge-width distribution weighted by excess intensity;
    - thin-ridge count and spot count/sizes;
    - orientation coherence;
    - SNR;
    - saturated fraction;
    - a horizon indicator.
  - Threshold rules come first, with a gradient-boosted fallback.
  - Validated on the 44 dev frames plus a 500-image stratified audit set. We require precision ≥ 90 % and
    recall ≥ 80 % per class before Pass B sampling.

### 6.6 Heterogeneity
- **Manifest** records shape, dtype, bit depth and path-derived run or beamtime.
- **Mask groups** are defined by shape plus a clustered always-dark signature.
- **Per-group masks** come from ~200 frames, dilated 2 px, versioned, with a QC sheet.
- **Missing metadata:** no metric needs geometry, so everything still runs and the frame is only flagged.
- **Other shapes:** recorded and excluded from Pass B.

### 6.7 Dry-run mode
- `python -m metric_study.cli dry-run --config metric_study.yaml --section dry_run` runs the exact
  production path on `~/Documents/sample_data`: manifest → masks → features → Pass A (with a checkpoint,
  if one is supplied) → Pass B → analysis → atlas export.
- It uses 10-image chunks, a forced crash with resume, and an injected corrupt file.
- It prints a validation report.
- It is runnable as a one-node SLURM job on NERSC.

## 7. Estimates (replaced by `benchmark` measurements)

**Dev run** (44 unique frames × ~70 variants × 2 resolutions):
- ≈ 1 h on laptop CPU;
- ≈ 2 min on one A100;
- < 50 MB of output.

**Full production run:**

| Pass | Assumption | Wall-clock | Compute |
|---|---|---|---|
| Manifest and masks | ~5 ms/file, 256 procs | ~0.5 h, 4 CPU nodes | ~2 node-h |
| A (1.5 M: recon + ~6 variants) | ~50 img/s/GPU (decode-bound) | ~1 h on 8 nodes | ~8–16 node-h |
| B (30 k × ~70 variants) | ~40 ms/variant → ≈ 23 GPU-h | ~1.5 h on 4 nodes | ~6–8 node-h |
| UMAP of `mu` for §11 (cuML, 1.5 M × 512) | minutes on 1 GPU | < 0.5 h | < 1 node-h |
| Optional C | ≈ 19 GPU-h | ~1.5 h on 4 nodes | ~5 node-h |

- **Storage:** ≈ 4 GB of outputs, plus up to ~2 GB of atlas files.
- **Uncertainty:** ±3×. The main unknowns are Lustre small-file throughput and CPU resize of degraded
  variants.

## 8. Decision criteria

**(a) 2–3 losses.** A metric qualifies at 512 in model space if:
1. median d′(fine, a=1) ≥ 5 and d′(peaks, f=1) ≥ 5 in classes (b) and (c);
2. R_fine/broad and R_peak/broad at matched ΔI are ≥ 5× MSE's;
3. d′(Poisson) ≤ 1;
4. median per-image Spearman(metric, severity) ≥ 0.95;
5. Kendall τ ≥ 0.5 with the human ranking of real reconstructions;
6. differentiable, with fwd+bwd ≤ 3× MSE time and ≤ 2× memory at batch 64.

From the qualifiers, choose a complementary set (pairwise correlation < 0.7). For example:
- one per-pixel term (log/asinh-L1);
- one structural or frequency term (MS-SSIM or FFL);
- peak-weighted MSE.

Each is added to KL and the rotation-alignment term.

**Also recommended regardless:** mask the gaps and beamstop in the training loss if §4.6 shows they
dominate MSE. If the raw-frame clip ablation shows the export clip removes peak signal, change the export.

**(b) 512 vs. native vs. hybrid.**
- **512 is acceptable** if, in classes (b) and (c):
  - ρ(native, 512) ≥ 0.9;
  - d′₅₁₂ ≥ 0.8·d′_native for fine and peak loss;
  - features ≥ 3 native px wide stay detectable (d′ ≥ 3).
- **Hybrid:** a 512 model plus a native-resolution loss on crops around detected features, comparing the
  upsampled reconstruction with the native target. Use it if loss is concentrated in 3–6 px features.
  The I(q) hybrid waits for calibration.
- **Native or patch-based training:** needed if scientifically relevant 1–3 px features are undetectable
  at 512 for all metrics. This requires model changes, outside this study.

## 9. Open questions
- **Q1. Checkpoints:** read access and exact path on NERSC (`…/latent_512/checkpoints/best.ckpt`?).
  - Which checkpoints and latent_dims matter?
  - Is `training_2.py` the entry point that produced them?
- **Q2. Production inputs:**
  - Are the 1.5 M files the exported 8-bit PNGs in `bl733_data_png*`, or JPEGs like the samples?
  - Where are they on NERSC?
  - Do **raw Pilatus frames** (EDF/TIFF/CBF, 32-bit) exist for at least a subsample? They would unlock
    true linear and Poisson metrics and the clip ablation.
- **Q3. Export transform:** where is the script that makes the PNGs?
  - log base and ε;
  - P_lo and P_hi;
  - per-image or global percentiles;
  - clip before or after log;
  - 8 or 16 bit;
  - how gaps and beamstop are handled.
  A reference raw→PNG pair would allow a golden test.
- **Q4. Duplicates:** 56 of 100 sample files are byte-identical. Is that expected in production?
  Duplicates also leak across the loader's random train/val split.
- **Q5. Labels:** please confirm the provisional dev-set classes.
- **Q6. GI frames:** should frames like 23990 be their own class or go in (c)?
- **Q7. Masks:** do beamstop or bad-pixel masks exist, or should they be data-driven?
- **Q8. NERSC:** QOS (regular or preempt), GPU-hour budget, env location (`$CFS`?).
- **Q9. Atlas hosting:** is an SSH tunnel to a NERSC node acceptable, or do you want a static export
  viewed locally, or NERSC Spin for sharing?

## 10. Verification once implemented
1. `pytest metric_study/tests` passes.
2. The dry run on `~/Documents/sample_data` passes its report checks:
   - 44 unique images, with duplicates flagged;
   - resume produces identical output;
   - sum preservation holds;
   - cv2 vs. torchvision differences are reported.
3. Visual QC of masks, degradation panels (F7) and the atlas by you.
4. `benchmark` on one Perlmutter node replaces the §7 estimates.
5. Pass A on ~10 k images first, then the full array.

## 11. Latent-space browsing with Embedding Atlas (apple.github.io/embedding-atlas)

**Verdict: yes, it fits,** with two design choices driven by scale. This was checked against its docs and
PyPI metadata, not yet installed or tested.
- **What it offers:**
  - a WebGPU scatter of "up to a few million points";
  - cross-filtering and search;
  - built-in viewers for image columns in tooltips and the instances view;
  - input as Parquet, CSV or JSONL (or a pandas DataFrame for the Jupyter widget and Streamlit
    component);
  - precomputed coordinates via `--x/--y` (otherwise it runs UMAP itself on a `--vector` column);
  - `--host/--port` for serving;
  - `--export-application` for a standalone site.
  - Image values can be URLs, data URLs, base64, or `{"bytes": …}` objects.
- **Latents:** `reconstruct.py` (Pass A) stores the deterministic `mu` per image with its reconstruction
  metrics. We project with PCA-50 followed by UMAP, using **cuML on one GPU** for all 1.5 M points. The
  built-in UMAP is CPU `umap-learn`, which is too slow at this size.
- **Side-by-side view:**
  - Each point gets **one composite image**: `original | reconstruction | |residual|`, 3 × 128 px,
    stored as `{"bytes": PNG}`, ≈ 15–25 KB.
  - A single composite guarantees the layout. Relying on how several image columns stack in the tooltip
    is not guaranteed.
  - Separate `original` and `recon` columns can be added too.
  - A larger 3 × 256 version is available for the instances view in the subsample atlas.
- **Scale split** (points scale; embedded images don't: 1.5 M × 20 KB ≈ 30 GB):
  1. **Full atlas:** 1.5 M points with coordinates and scalar columns (class, mask group, reconstruction
     metrics, saturated fraction, run/beamtime) and no images. About 200 MB, for spotting clusters and
     outliers.
  2. **Image atlas:** a stratified 50 k subsample with composite thumbnails, about 1 GB. It is
     oversampled for classes (b) and (c) and for the worst reconstructions under each shortlisted metric.
  - Optionally, full-set images via a URL column served by a small HTTP server (Q9).
- **Running it (separate Python ≥ 3.11 env):**
  - on NERSC: `embedding-atlas atlas_50k.parquet --x umap_x --y umap_y --image composite --port 5055`
    on a compute or login node, plus `ssh -L 5055:localhost:5055 perlmutter`;
  - or the Jupyter widget in NERSC JupyterHub;
  - or `--export-application atlas.zip`, then open it locally.
  - Exact widget arguments are verified at implementation time.
- **Use in the study:**
  - color by pattern class and by per-image metric disagreement (e.g. MSE is low but peak-weighted MSE
    is high);
  - this shows latent regions where reconstructions drop fine rings or peaks, and supplies candidates for
    the human ranking in §4.6.
- **Output:** `atlas_export.py` → `atlas_full.parquet` and `atlas_50k.parquet`. It reads Pass A outputs
  only.
