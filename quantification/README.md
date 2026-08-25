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

The common mask keeps finite sky pixels at least 1° above the horizon, removes measured
values at or above 4090 counts, and excludes a 10° region around the Sun. AOP additionally
requires measured DOP of at least 0.05. These masks depend only on the capture and geometry,
not on the model being scored.

The circular blank region in residual figures is therefore intentional: it is the projected
10° solar exclusion mask, not a feature produced by a sky model. After time calibration it
should cover the visible solar starburst. If it does not, the remaining likely causes are
camera pitch or roll, optical-centre error, lens distortion, yaw error, or incorrect manifest
geometry; the time calibration does not estimate those quantities.

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

## Outputs

The entire output folder is gitignored.

| Artifact | Contents |
|---|---|
| `metrics.csv` | Human-readable scalar metrics in combined-score rank order |
| `results.json` | Package version, capture and camera configuration, checksums, assumptions, masks, vendor-convention validation, CIE sweep and complete per-model metrics |
| `model_ranking.png` | Combined normalized-RMSE score for every selected model |
| `cie_selection.png` | Affine intensity NRMSE for CIE types 1–15 and the selected type |
| `<model>_residuals.png` | Measured, simulated and residual DOP/AOP fields plus one representative raw analyzer sub-grid for a model |
| `sun_time_search.csv` | Every coarse and diagnostic fine UTC candidate and its image score |
| `sun_time_calibration.json` | Search settings, checksums, baseline and selected projections, displacement and diagnostic refinement |
| `sun_time_calibration.png` | Raw-image solar track, apertures and score-versus-time plot |

## Interpretation limits

This is one partly cloudy, Sun-visible capture whose TIFF metadata does not preserve time,
location or orientation. Those values were recovered from the earlier scripts and remain
explicit assumptions in the manifest. The image-anchored search corrects only the
time-derived Sun placement; it does not estimate camera pitch, roll, optical-centre error or
lens distortion. The results are an observational comparison, not a clear-sky validation
dataset or evidence of general model superiority.

Pixels within one image are also spatially correlated; their large count is not a substitute
for independent observations across days and solar elevations. Publication-level validation
would require calibrated camera geometry and polarimetric response, repeated clear-sky
captures, session-level uncertainty intervals, and sensitivity analyses for the 0.7 scale and
mask thresholds.
