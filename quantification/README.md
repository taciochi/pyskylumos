# Capture quantification

This repository-local workflow compares the six canonical PySkyLumos models with one
2048×2448 Thorlabs polarimetric capture. It is not installed as part of the public Python
API, and the 86.4 MiB capture remains local rather than entering Git or a distribution.

## Prepare the inputs

From the repository root, extract only the archive's input folder:

```console
mkdir -p quantification/input
unzip -j pyskylumos_quantification.zip 'pyskylumos_quantification/input/*' \
  -x '*/.DS_Store' -d quantification/input
```

The manifest in [`capture.toml`](capture.toml) records the expected SHA-256 digest of every
one of the 15 files. The script verifies all files before loading the four TIFFs used for
scoring: Unprocessed, Intensity, DoLP and Azimuth. Quad View and the JPEG previews are
preserved but are not quantitative inputs.

Install the repository and optional plotting/image dependencies, then run all models:

```console
python -m pip install -e ".[quantification]"
MPLBACKEND=Agg python quantification/calibrate_sun_time.py \
  --config quantification/capture.toml \
  --output-dir quantification/output
MPLBACKEND=Agg python quantification/quantify_models.py \
  --config quantification/capture.toml \
  --output-dir quantification/output
```

The calibration command is a dry run unless `--write-config` is supplied. It searches three
hours on either side of the manifest time at one-minute resolution, projects the astronomical
Sun through the configured stereographic camera, and selects the 45-pixel-radius raw-image
aperture with the greatest saturated fraction. DOP, AOP, and the candidate sky models do not
enter this calibration. For this capture it independently selects the minute
`2025-09-22T15:05 UTC`; the old `16:05` value was local British Summer Time. A one-second
refinement is recorded only as a diagnostic because the saturated starburst cannot justify
second-level capture-time accuracy. To reproduce the atomic manifest update explicitly, run:

```console
MPLBACKEND=Agg python quantification/calibrate_sun_time.py \
  --config quantification/capture.toml \
  --output-dir quantification/output \
  --write-config
```

Pass `--models RAYLEIGH BERRY` to evaluate a subset, or `--show` to display each residual
figure after saving it. The canonical choices are `RAYLEIGH`, `DEPOLARIZED_RAYLEIGH`,
`ASYMMETRIC`, `BERRY`, `PAN`, and `QUEEN`. Paths are resolved independently of the current
working directory.

Two further scripts characterize that result rather than produce it. Both are considerably
slower than the main run, and both accept `--quick` for a reduced smoke test:

```console
MPLBACKEND=Agg python quantification/sensitivity.py \
  --config quantification/capture.toml \
  --output-dir quantification/output
MPLBACKEND=Agg python quantification/validate_time_recovery.py \
  --config quantification/capture.toml \
  --output-dir quantification/output
```

[Sensitivity and uncertainty](#sensitivity-and-uncertainty) and
[time-recovery validation](#time-recovery-validation) below describe what each one answers.

## What is measured

The workflow decodes the vendor DoLP TIFF as `code / 4095` and its Azimuth TIFF as an axial
angle over 180°. The manifest's +90° adjustment converts the vendor analyzer convention to
the PySkyLumos sensor convention. Arithmetic means pool DoLP over each 2×2 analyzer tile;
AOP uses the mean of `exp(2j*AOP)` so values around −90° and +90° remain adjacent.

Every model is run with its package defaults. Its DOP is then multiplied exactly once by the
manifest's empirical `dop_scale = 0.7`; measured DOP is not scaled. This is a declared
comparison assumption, not a universal upper bound on daylight polarization. DOP errors are
normalized by 0.7, AOP errors use a 180° period and are normalized by 90°, and the ranking is
the root-mean-square of those two normalized RMSE values with equal weight.

The common mask is restricted to the lens's **usable image circle**: sensor pixels no further
than `camera.usable_image_radius_pixels = 800` from the optical centre. Beyond that lies the
camera rim, and comparing models over it means ranking how well each one fits a mechanical
edge. The limit is measured from the capture, not assumed:

| Evidence | Result |
|---|---|
| Azimuthal median of `Snapshot_Intensity_0001.tif`, normalized to its r = 400–600 px level | 1.41× at r = 850 px, 0.87× at 860 px, 0.52× at 880 px, 0.28× at 900 px |
| Vendor AOP coherence, the modulus of the mean of `exp(2j*AOP)` over 16×16 native blocks | ≥ 0.994 out to r = 850 px, then 0.977, 0.933 at 900 px, 0.833 at 1000 px, 0.474 at 1050 px |
| Illuminated-disc centre against the geometric centre the mask assumes | offset of about 20 px |

The rim therefore begins near r = 850 px. 800 px clears that onset by roughly 50 px, which
also absorbs the 20 px centre offset. Under the manifest's stereographic mapping it is an
altitude cut of 15.1°, keeping 74% of the hemisphere by solid angle; `results.json` records
both equivalents. The limit is stored in pixels rather than degrees on purpose: the rim is a
property of the sensor and lens, while `focal_length_micrometers` is an uncalibrated
assumption, so a pixel limit cannot drift back onto the rim if that assumption is corrected.

`camera.altitude_min_deg = 1.0` no longer sets the comparison field. It is only the
simulation-domain clip that keeps the CIE luminance formula, which is undefined below the
horizon, away from below-horizon rays.

The mask also removes measured values at or above 4090 counts and excludes a 10° region
around the Sun. AOP additionally requires measured DOP of at least 0.05. These masks depend
only on the capture and geometry, not on the model being scored.

Residual figures therefore show two circular boundaries, and neither is produced by a sky
model. The outer edge of the disc is `usable_image_radius_pixels`, the field limit above; each
figure's title states it in both pixels and degrees. The smaller blank patch inside the disc is
the projected 10° solar exclusion, and only that one should cover the visible solar starburst
after time calibration. If it does not, the remaining likely causes are camera pitch or roll,
optical-centre error, lens distortion, yaw error, or incorrect manifest geometry; the time
calibration does not estimate those quantities.

## Metrics and ranking

For any residuals `e`, MAE is `mean(abs(e))` and RMSE is
`sqrt(mean(e²))`. MAE describes the typical absolute discrepancy; RMSE weights large local
discrepancies more strongly. AOP residuals are axial: they are wrapped into `[-90°, 90°)` so
that, for example, 89° and −89° differ by 2° rather than 178°.

`metrics.csv` uses these columns:

| Column | Meaning | Lower is better? |
|---|---|---|
| `rank` | Position after sorting by `polarization_score`; it is not an MAE rank | Yes |
| `model` | Canonical model name | Not applicable |
| `polarization_score` | Equal-weight combination `sqrt((dop_nrmse² + aop_nrmse²) / 2)` | Yes |
| `dop_mae`, `dop_rmse` | DOP errors after the one shared 0.7 simulation scale | Yes |
| `dop_nmae`, `dop_nrmse` | Corresponding DOP errors divided by 0.7 | Yes |
| `dop_count` | Number of accepted 2×2 analyzer tiles used for DOP | Not an error metric |
| `aop_mae_deg`, `aop_rmse_deg` | Axial AOP errors in degrees | Yes |
| `aop_nmae`, `aop_nrmse` | Corresponding AOP errors divided by 90° | Yes |
| `aop_count` | Accepted analyzer tiles after also requiring measured DOP ≥ 0.05 | Not an error metric |
| `raw_mae_counts`, `raw_rmse_counts` | Residual ADC counts after affine alignment | Yes |
| `raw_nrmse` | Affine-aligned raw RMSE divided by 4095 | Yes |
| `raw_r_squared` | Fraction of measured raw variance described after alignment; it can be negative | Higher is better |
| `raw_gain`, `raw_offset` | Fitted coefficients in `measured ≈ gain × simulated + offset` | Diagnostic |
| `raw_count` | Number of accepted native pixels used for raw agreement | Not an error metric |

The normalized DOP and AOP terms make the combined score dimensionless before they receive
equal weight. Consequently, `rank` answers “which model has the smallest combined normalized
DOP/AOP RMSE?”, not “which has the smallest MAE?” and not “which has the best AOP?”. To answer
either of those questions, sort by `dop_mae`, `aop_mae_deg`, or another column and label that
ordering explicitly.

AOP can reasonably be the primary endpoint when orientation is the scientific objective and
DOP is expected to be more sensitive to clouds, aerosols, exposure or the empirical 0.7
scale. In that case, report the AOP ordering as the primary result and DOP and the combined
rank as secondary results; do not silently reinterpret the existing `rank` column.

## Shared radiance and raw diagnostic

CIE types 1–15 are compared once against the measured Intensity TIFF using an affine
exposure fit. The lowest-RMSE type (lower index on an exact tie) is reused for every model.
The script then predicts each native analyzer pixel with the deterministic PySkyLumos
micro-polarizer equation and extinction ratio 0.99. It does not inject random defects,
sensor noise or a second DOP scale. The four analyzer sub-grids are simulated separately so
each raw pixel retains its exact viewing geometry without constructing unnecessary full-size
intermediate fields.

Raw ADC agreement is affine-aligned independently for each model and reported in counts,
normalized RMSE and R². It is a separate sensor/radiance diagnostic and never affects the
polarization ranking.

## Sensitivity and uncertainty

`quantify_models.py` reports one ranking under one set of declared assumptions. A ranking that
holds only at those values is not a result, so [`sensitivity.py`](sensitivity.py) re-derives it
two ways. Both import the scoring functions from `quantify_models` unchanged, so the numbers
agree with the main run by construction.

The **sweep** recomputes the whole ranking at every value of four assumptions: `dop_scale`
(0.50–0.90), `usable_image_radius_pixels` (730–875 px), `sun_exclusion_deg` (5–20°) and
`aop_min_measured_dop` (0.02–0.15). The CIE type is reselected for every setting. Simulated
fields are cached per model and CIE type, because the DOP scale is a pure multiplier and
sweeping it must not force resimulation. The widest radius reaches into the camera rim on
purpose, so the excluded annulus stays visible in the record rather than hidden by it.

The **moving-block bootstrap** addresses a different problem: the several hundred thousand
accepted analyzer tiles come from one frame and are spatially correlated, so they are not that
many independent observations. Contiguous blocks of 32×32 tiles, 64 native pixels on a side,
are resampled with replacement 1000 times at 95% confidence, and every model is scored on the
*same* resampled blocks so the pairwise differences are paired. It runs in two regimes, as
published and on a tighter field, to show which conclusions survive shrinking the radius.

For the current capture the sweep records 34 rank changes across 18 settings. At r ≤ 800 px
`DEPOLARIZED_RAYLEIGH` scores 0.1488 with a 95% interval of [0.1377, 0.1604] and 10 of 15
model pairs separate; at r ≤ 730 px, 13 of 15 separate. Those intervals describe **this frame
only**. Blocks of adjacent tiles are not independent observations, and no amount of resampling
converts a single capture into a sample of skies.

## Time-recovery validation

`calibrate_sun_time.py` recovered a one-hour error on this capture, but one observation
establishes no operating envelope. [`validate_time_recovery.py`](validate_time_recovery.py)
supplies one: it renders frames whose acquisition time is known exactly, runs the **shipped**
recovery on them — imported, not reimplemented — and scores the answer against that truth. Six
sweeps perturb the frame away from an exact inverse of the recovery model: solar elevation
(5–55°, drawn from across the year at the same site), aperture radius, exposure fraction,
multiplicative noise SNR, camera yaw error and cloud occlusion including a fully hidden Sun.

Of 58 trials, 55 returned a time. Across clear-sky trials 74% land on the exact minute and the
median absolute error is 60 s, but the maximum is 2580 s: 14.5% of accepted recoveries are
*silent failures*, meaning a confident answer wrong by more than ten minutes. A silent failure
is worse than a refusal, so the script measures them separately rather than folding them into a
mean.

**Pose aliasing** explains the largest class of them. A yaw error rotates the projected Sun,
and the search absorbs that by sliding along the solar track, producing a compact and entirely
wrong detection. The fitted exchange rate is 189 s per degree at R² = 0.998, so 0.32° of yaw
error buys a full minute of apparent time. No single-frame image statistic can detect this: the
detection is genuine, the pose is not. That number is why the manifest treats minute precision
as authoritative and records the one-second refinement as a diagnostic only.

### Why there is no shipped containment guard

The remaining silent failures — over-exposed frames, and frames whose Sun is hidden so that
auto-exposure renormalizes onto diffuse bright sky — look separable on rendered frames. The
**containment** statistic, `1 - eroded annulus saturated fraction / aperture saturated
fraction`, catches 5 of the 6 image-detectable cases across the sweeps at a cost of one false
rejection in 47, and on that evidence alone it would ship as an admissibility test.

**It does not transfer to the instrument, and the script measures that rather than assuming
it.** `measure_real_capture_containment` scores the real capture on the same axis as the
rendered frames and records the result next to the scorecard as `real_capture_control`, with
`transfers_to_real_capture` recording the verdict:

| Frame | Containment | Truth |
|---|---|---|
| Repository capture, r = 45 px | **0.626** | correct detection |
| Rendered clear sky | 1.000 | correct detection |
| Rendered Sun fully hidden | 0.816–0.828 | wrong by ~29 minutes |
| Rendered exposure 0.50 | 0.381 | wrong by 43 minutes |

The real correct detection scores *below* every rendered hidden-Sun failure, so the ordering is
inverted and no threshold on this statistic separates the two. The cause is the renderer: a CIE
radiance peak has no lens flare, glare or blooming, so its starburst is compact — saturated
fraction falls from 1.00 at r = 15 px to 0.03 at r = 180 px — whereas the instrument's falls
only from 0.71 to 0.24 over the same span, a profile nearly identical in shape to the rendered
hidden-Sun failure.

`calibrate_sun_time.py` therefore **reports** containment and the score-curve shape in
`sun_time_calibration.json` under `detection_diagnostics` and enforces neither. The opt-in
`--minimum-containment` gate exists for an instrument where a threshold has been established,
and deliberately ships with no default. Establishing one needs real captures of the failure
cases, or a renderer that reproduces flare — neither of which this repository has.

## Outputs

The entire output folder is gitignored.

| Artifact | Contents |
|---|---|
| `metrics.csv` | Human-readable scalar metrics in combined-score rank order |
| `results.json` | Package version, capture and camera configuration, checksums, assumptions, masks, the retained field as `usable_image_radius_pixels` with its `usable_image_radius_altitude_deg` and `usable_image_sky_fraction` equivalents, vendor-convention validation, CIE sweep and complete per-model metrics |
| `model_ranking.png` | Combined normalized-RMSE score for every selected model |
| `cie_selection.png` | Affine intensity NRMSE for CIE types 1–15 and the selected type |
| `<model>_residuals.png` | Measured, simulated and residual DOP/AOP fields plus one representative raw analyzer sub-grid for a model |
| `sun_time_search.csv` | Every coarse and diagnostic fine UTC candidate and its image score |
| `sun_time_calibration.json` | Search settings, checksums, baseline and selected projections, displacement, diagnostic refinement, and the reported-only `detection_diagnostics` (containment and score-curve shape) |
| `sun_time_calibration.png` | Raw-image solar track, apertures and score-versus-time plot |
| `sensitivity.csv` | One row per model and swept setting: both ranks, the DOP and AOP metrics and the combined score |
| `sensitivity.json` | Sweep settings, baseline ranks, rank-change count, CIE type per setting, and both bootstrap regimes with intervals and paired model comparisons |
| `sensitivity_ranks.png` | Combined rank against each swept assumption, one panel per parameter |
| `bootstrap_scores.png` | Bootstrap score intervals per model, as published and on the tighter field |
| `time_recovery_trials.csv` | Every closed-loop trial: render settings, recovered time, time and angular errors, and guard statistics |
| `time_recovery_summary.csv` | Per-sweep aggregate: failure rate, median and maximum time error, plateau width and silent-failure rate |
| `time_recovery.json` | Search and rendering settings, declared assumptions, totals, the containment scorecard with its real-capture control and `transfers_to_real_capture` verdict, and the pose-aliasing fit |
| `time_recovery.png` | Six-panel characterization: solar elevation, aperture radius, exposure, pose aliasing, occlusion outcomes and the containment guard |

## Interpretation limits

This is one partly cloudy, Sun-visible capture whose TIFF metadata does not preserve time,
location or orientation. Those values were recovered from the earlier scripts and remain
explicit assumptions in the manifest. The image-anchored search corrects only the
time-derived Sun placement; it does not estimate camera pitch, roll, optical-centre error or
lens distortion. The results are an observational comparison, not a clear-sky validation
dataset or evidence of general model superiority.

The ranking depends on the comparison radius, so always report the retained field alongside it
and treat a ranking quoted without one as incomplete. `sensitivity.py` sweeps the radius from
730 px to 875 px, the last of which deliberately reaches into the rim so the effect stays
visible rather than hidden.

Two distinct effects appear in that sweep and must not be conflated. Only the widest settings
admit a meaningful amount of rim: measured against the fitted disc centre, 0% of the 800 px
mask and 0.5% of the 845 px mask lie past the 855 px rim onset, against 4.5% at 875 px and 30%
at the 1025 px mask this workflow previously used. The ordering, however, already changes at
about 825 px, which is clean sky: `DEPOLARIZED_RAYLEIGH` leads inside that radius and
`ASYMMETRIC` leads outside it, because the two disagree most about the band between roughly
12° and 15° altitude. That crossover is a real near-horizon difference between the models, not
a rim artefact.

The published 800 px result therefore sits about 25 px, or 1.5° of altitude, inside a genuine
crossover, and its margin over `ASYMMETRIC` there is small: 0.1488 against 0.1604. The radius
is chosen to exclude the rim, not to select a winner, and the ranking should be reported with
that proximity stated.

The 20 px offset between the illuminated-disc centre and the assumed optical centre is
absorbed by the 800 px limit's margin, but it is not corrected; optical-centre error remains
one of the quantities this workflow does not estimate. Two azimuth sectors also contain ground
obstructions whose intensity collapses from about r = 730 px, well inside the rim. The radius
limit does not remove those, and no obstruction mask is applied.

Pixels within one image are also spatially correlated; their large count is not a substitute
for independent observations across days and solar elevations. Publication-level validation
would require calibrated camera geometry and polarimetric response, repeated clear-sky
captures, session-level uncertainty intervals, and sensitivity analyses for the 0.7 scale and
mask thresholds.
