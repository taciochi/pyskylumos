# Changelog

All notable changes to PySkyLumos are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Section numbers referenced below point into the
[mathematical reference](README.md#mathematical-reference), where every formula change is
attributed to its source.

## [0.1.2] — 2026-08-29

The importable package is unchanged again; this release touches only the repository-local
capture workflow. It adds detection diagnostics to the image-anchored time calibration and
records, as a measurement rather than a claim, that the containment statistic those diagnostics
expose is **not** a valid accept/reject test on real data.

### Added

- `detection_diagnostics` in `sun_time_calibration.json`: the winning detection's aperture and
  eroded-annulus saturated fractions, their containment ratio, and the shape of the
  score-versus-time curve (peak, median, peak-to-median ratio, half-peak width, fraction above
  half peak). Reported unconditionally; never enforced.
- `calibrate_sun_time.py --minimum-containment`, an **opt-in** gate with no default. It exists
  for an instrument where a threshold has been established from real failure captures.
- `measure_real_capture_containment` in
  [`quantification/validate_time_recovery.py`](quantification/validate_time_recovery.py), the
  control the synthetic sweeps cannot supply. `time_recovery.json` now carries
  `real_capture_control` and a `transfers_to_real_capture` verdict beside the guard scorecard.

### Changed

- `erode_mask` and `containment_statistics` moved from `validate_time_recovery.py` into
  `calibrate_sun_time.py`, and the validation script imports them. The characterized code is
  now literally the shipped code, as the search already was.
- **The containment guard is documented as not shipping, and why.** On rendered frames it
  catches 5 of 6 image-detectable silent failures for one false rejection in 47. On the
  repository capture a *correct* detection scores 0.626, below every rendered hidden-Sun
  failure at 0.816–0.828, so the ordering is inverted and no threshold separates them. The
  cause is the renderer: a CIE radiance peak carries no lens flare, glare or blooming, so its
  starburst is far more compact than the instrument's. See
  [the quantification README](quantification/README.md#why-there-is-no-shipped-containment-guard).

## [0.1.1] — 2026-08-25

No behaviour changed in the importable package. Every module in the wheel is byte-identical to
0.1.0 except `_version.py`, so upgrading cannot alter a result. This release updates the
repository-local capture comparison that ships in the source distribution, and the project
description shown on PyPI.

### Changed

- **The capture comparison is bounded by the lens's usable image circle.**
  [`quantification/capture.toml`](quantification/capture.toml) gains
  `camera.usable_image_radius_pixels`, and the evaluation mask now drops every pixel beyond it.
  The previous mask reached 1025 px — past the lens image circle — so the camera rim entered
  every score. The rim onset is measured from the capture itself at 850–860 px, where the
  azimuthal median intensity collapses and the vendor AOP field decoheres; the shipped limit of
  800 px clears it with enough margin to absorb the 20 px optical-centre offset. `results.json`
  records the equivalent altitude and sky fraction, and every residual figure now states the
  retained field in its title.
- **`camera.altitude_min_deg` no longer sets the comparison field.** It is now only the
  simulation-domain clip that keeps the CIE luminance formula, undefined below the horizon, away
  from below-horizon rays. Its value is unchanged at 1.0.

### Added

- [`quantification/sensitivity.py`](quantification/sensitivity.py) re-derives the ranking across
  every declared assumption and attaches moving-block bootstrap intervals to the scores.
- [`quantification/validate_time_recovery.py`](quantification/validate_time_recovery.py)
  characterizes the image-anchored time recovery against rendered frames whose acquisition time
  is known exactly, reporting its failure modes rather than a single success.

## [0.1.0] — 26.08.2026

The first release with a documented provenance for every formula. It doubles the number of sky
models, states the units in the constructor names, and adds a public exception hierarchy.

**If you are upgrading from 0.0.6, read the two breaking changes first.**

### Breaking

- **`sky_model="PAN"` now selects a different model.** Up to and including 0.0.6, `PAN` reported
  a Berry quartic field placed at Pan's offsets and remapped through OpenSky's
  intensity-to-DoLP conversion. That model is now called **`QUEEN`** and reproduces the
  historical results exactly. `PAN` now implements Pan et al. (2023) as published, reporting
  `abs(ω)` directly and referencing AOP to the local meridian.
  **To keep 0.0.6 behaviour, change `sky_model="PAN"` to `sky_model="QUEEN"`.**
  `"QEN"` is accepted as an alias for the name used in early drafts.
- **Sky fields are now `float64`.** `Rayleigh` cast its degree of polarization to `float32` in
  0.0.6. All sky fields and metadata are now `float64`; ADC counts and the reconstructed sensor
  DOP and AOP remain `float32`. Code that asserted on the old dtype needs updating.

### Deprecated

Six arguments were renamed to state their unit and their meaning. The old names still work but
emit `DeprecationWarning`, and **will be removed in 0.2.0**. Passing both names for the same
quantity raises `ConfigurationError`.

The first five are constructor arguments. The sixth is an argument of
`Engine.get_initial_azimuth_altitude`, renamed so that it matches the name
`OpticalConjugator.get_azimuth_altitude` has always used for the very same hook; the two
methods disagreed in 0.0.6, so code written against the conjugator did not transfer to the
Engine. Calls that passed the hook **positionally** are unaffected and emit no warning.

| Old name | New name | Why it changed |
|---|---|---|
| `sensor_pixel_size_square_micrometers` | `sensor_pixel_pitch_micrometers` | The value is a linear pitch, not an area |
| `tolerance` | `polarizer_tolerance_radians` | States both what it bounds and its unit |
| `pixel_saturation_ratio` | `auto_exposure_saturation_fraction` | It is an auto-exposure target, not a per-pixel ratio |
| `adc_resolution` | `adc_resolution_bits` | States the unit |
| `signal_to_noise_ratio` | `multiplicative_noise_snr` | States that the noise is multiplicative |
| `custom_lens_conjugation_type` | `custom_lens_conjugation` | The argument holds a callable, not a projection-type string, and the name now agrees with `OpticalConjugator` |

### Added

- **`DEPOLARIZED_RAYLEIGH`**, implementing the anisotropic-molecule phase-matrix correction of
  Wu et al. (2014) with a configurable `depolarization_ratio` defaulting to the dry-air
  approximation `0.0279` of Bodhaine et al. (1999).
- **`ASYMMETRIC`** (alias `ASQ`), which gives all four quartic neutral points independent
  positions along one covariant signed solar meridian. Options: `arago_offset`,
  `fourth_offset`, `normalisation`, `dop_max` and `out_of_range`.
- `model_options`, a dictionary forwarded to the chosen model's constructor. Passing an option
  a model does not accept raises `InputTypeError` and lists the accepted ones.
- **Integrated rigid sensor tilt.** Passing both `sensor_azimuthal_tilt_radians` and
  `sensor_tilt_angle_radians` to `Engine.simulate_sky_polarization` reinterprets the direction
  grid as sensor-local, rotates the rays into world coordinates, and transports AOP into the
  tilted analyzer frame. The correction varies per pixel; it is not a single angle added to the
  field. See [section 10](README.md#10-conventions).
- `Engine.convert_local_meridian_aop_to_sensor`, for feeding direct `Pan` output into the
  sensor pipeline on an untilted camera.
- `random_seed` on `Engine`, shared by the micro-polarizer defect map and the sensor noise, so
  a whole measurement is reproducible.
- A public exception hierarchy in `pyskylumos.exceptions`: `PySkyLumosError` and the three
  subclasses `ConfigurationError`, `InputValidationError` and `InputTypeError`. Each also
  subclasses the matching builtin, so existing `except ValueError` and `except TypeError`
  handlers keep working.
- Two warning classes, `NeutralPointRangeWarning` and `PanFidelityWarning`, exported from
  `pyskylumos.sky_models`.
- `pyskylumos.__version__`, guaranteed to match the distribution metadata.
- A `py.typed` marker, making the package [PEP 561](https://peps.python.org/pep-0561/) typed.
- `QuarticSkyModel`, the abstract base shared by the four quartic models.
- The [mathematical reference](README.md#mathematical-reference): every implemented formula
  attributed to a publication, eighteen errata for Pan et al. (2023), and an explicit audit of
  Berry, Dennis and Lee (2004) recording that none were found there.
- [`examples/`](examples/), one runnable plotting script per model; [`quantification/`](quantification/),
  comparing all six models against a polarimetric capture; and `audit/`, a set of standalone
  verification scripts including an independently written numerical oracle.
- Continuous integration across Python 3.12, 3.13 and 3.14, on Linux, macOS and Windows,
  covering minimum-dependency resolution, strict typing, branch coverage and packaging.

### Changed

- Requires Python `>=3.12,<3.15`, up from `>=3.12.2` with no upper bound. Dependency floors
  moved to `astropy>=8,<9` (from `>=7.1.0`) and `numpy>=2.4,<3` (from `>=2.3.1`).
- Validation now raises the package exception types rather than bare `ValueError` and
  `TypeError`, with messages that name the offending argument and its accepted domain.
- `accuracy=True` reports a missing optional dependency as `ModuleNotFoundError` and an
  unreachable DE430 kernel as `RuntimeError`, instead of silently falling back to the built-in
  ephemeris.
- Optional dependencies are split into the `examples`, `quantification`, `test`, `quality`,
  `release`, `dev` and `jpl` extras.

### Fixed

- **Berry metadata transposition** (erratum **B1**). `Berry` returned the sun and anti-sun
  azimuth and elevation transposed with respect to one another.
- **`altitude_min_clip=0` was ignored by `Pan`.** The mask was applied under a truthiness test,
  so the documented value `0` — the horizon, and the value the quick start uses — skipped
  masking entirely. All models now test against `None`, so `0` masks as documented.
- **`Pan` did not mask its scattering angle.** Radiance, DOP and AOP were masked but the
  scattering-angle field was left unmasked, so it disagreed with the other three below the clip
  altitude.
- Corrections to Pan et al. (2023) as implemented, each documented individually in
  [section 8](README.md#8-errata--pan-et-al-2023): the solar radius mixing stereographic and
  orthographic forms (**E5**), the sign error in the second root's denominator (**E6**), the
  missing reciprocal that stopped the fourth factor being an antipode (**E7**), the
  Babinet/Brewster pairing that placed Brewster past the zenith (**E8**), the rotation
  covariance that moved AOP by twice the scene rotation (**E9**), and the 90° disagreement
  between two definitions of the same azimuth (**E13**).

## [0.0.6] — 2026-02-02

Last release of the original three-model package: `RAYLEIGH`, `BERRY` and `PAN`, with the
division-of-focal-plane sensor pipeline and an explicit `sun_position` override.
