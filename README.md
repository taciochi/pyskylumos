# PySkyLumos

[![PyPI version](https://img.shields.io/pypi/v/pyskylumos.svg)](https://pypi.org/project/pyskylumos/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyskylumos.svg)](https://pypi.org/project/pyskylumos/)
[![CI](https://github.com/taciochi/pyskylumos/actions/workflows/ci.yml/badge.svg)](https://github.com/taciochi/pyskylumos/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

PySkyLumos answers two questions:

1. **What does the polarization pattern of the daytime sky look like?**
   Given a place, a time and a set of viewing directions, it computes the degree of
   polarization, the angle of polarization and the relative radiance of the sky.
2. **What would a polarization camera actually record if you pointed it at that sky?**
   It pushes those ideal fields through a fisheye lens, a micro-polarizer array, sensor
   noise and an analog-to-digital converter, then reconstructs polarization the way a real
   camera's pipeline would.

It is written for research in atmospheric optics, bio-inspired navigation, robotics,
computer vision and synthetic-dataset generation.

Every established formula in the package is traced to its source. Package-specific
constructions, numerical choices and literature-motivated assumptions—most notably the
`AsymmetricQuartic` extension—are labelled as such, and every place where the implementation
departs from a printed equation is documented in the
[mathematical reference](#mathematical-reference).

## Contents

**Using the package**

- [Installation](#installation)
- [Vocabulary](#vocabulary)
- [Quick start](#quick-start)
- [How a simulation flows](#how-a-simulation-flows)
- [Choosing a sky model](#choosing-a-sky-model)
- [Configuring the camera](#configuring-the-camera)
- [What a simulation returns](#what-a-simulation-returns)
- [Sky brightness: the CIE sky types](#sky-brightness-the-cie-sky-types)
- [Where the sun comes from](#where-the-sun-comes-from)
- [Camera pose: rotation and tilt](#camera-pose-rotation-and-tilt)
- [Errors and warnings](#errors-and-warnings)
- [Visual examples](#visual-examples)
- [Quantifying models against a capture](#quantifying-models-against-a-capture)
- [Limits of the sensor model](#limits-of-the-sensor-model)
- [Development](#development)

**Reference**

- [Mathematical reference](#mathematical-reference) — derivations, provenance, errata
- [Conventions](#10-conventions) — frames, masking, sensor equations
- [What is and is not modelled](#11-what-is-and-is-not-modelled)
- [Changes and migration](CHANGELOG.md) — release history, renamed parameters, model renames
- [License and citation](#license-and-citation)

## Installation

PySkyLumos requires Python 3.12–3.14 and depends only on NumPy and Astropy.

```console
python -m pip install pyskylumos
```

Optional extras. Only `examples` and `jpl` are useful from an ordinary installation; the rest
assume a repository checkout. `quantification` drives the local capture comparison, `test`,
`quality` and `release` each reproduce one [CI job](#development), and `dev` is their union.

| Extra | Install command | What it adds |
|---|---|---|
| `examples` | `pip install "pyskylumos[examples]"` | Matplotlib, needed to run the plotting scripts in [`examples/`](examples/) |
| `jpl` | `pip install "pyskylumos[jpl]"` | The JPL DE430 solar ephemeris, used when you pass `accuracy=True` |
| `quantification` | `pip install -e ".[quantification]"` | Pillow and Matplotlib for the repository-local [capture comparison](quantification/) |
| `test` | `pip install -e ".[test]"` | pytest, pytest-cov, and the Matplotlib and Pillow the test suite imports |
| `quality` | `pip install -e ".[quality]"` | Ruff and mypy, for the formatting, linting and strict-typing checks |
| `release` | `pip install -e ".[release]"` | build, twine and pip-audit, for packaging and dependency auditing |
| `dev` | `pip install -e ".[dev]"` | Everything above except `jpl`, plus pre-commit. This is the one to install for development |

For an editable checkout of the repository:

```console
python -m pip install -e ".[examples]"
```

**Typing.** PySkyLumos ships a `py.typed` marker, so it is a
[PEP 561](https://peps.python.org/pep-0561/) typed package: mypy and Pyright read the inline
annotations of every public class and method without a separate stub package. The package
itself is checked in mypy's strict mode.

## Vocabulary

If you have not worked with sky polarization before, these six terms cover almost
everything in this README.

| Term | Meaning |
|---|---|
| **Degree of polarization (DOP)** | How strongly polarized the light from one sky direction is, from 0 (unpolarized) to 1 (fully polarized). Also written DoLP for the *linear* degree of polarization, which is what this package models. |
| **Angle of polarization (AOP)** | The orientation of the polarization. It is an *axis*, not an arrow: 10° and 190° describe the same state, so AOP only ever spans a 180° range. This package returns it in **radians**. |
| **Neutral point** | A direction in the sky where the degree of polarization falls to zero, so the angle of polarization is undefined. The classical named ones are Babinet (above the sun), Brewster (below the sun) and Arago (above the anti-sun). They are also called *polarization singularities*. |
| **Scattering angle** | The angle between the sun and the sky direction you are looking at, measured from the observer. It drives the whole Rayleigh pattern. |
| **AltAz frame** | Astropy's horizontal coordinate system: **altitude** is the angle above the horizon (0° horizon, 90° zenith) and **azimuth** runs 0° at North, increasing towards East. All sky coordinates crossing this API use it, in **degrees**. |
| **Micro-polarizer array** | The physical sensor this package models: a division-of-focal-plane camera with a tiny wire-grid polarizer bonded over every pixel, in a repeating 2×2 tile of 0°, 45°, 90° and 135° analyzers. |

**Units, once and for all.** Angles that describe a *sky direction* (`azimuths`, `altitudes`,
`altitude_min_clip`, `azimuth_rotation_angle`) are in **degrees**. Every angle the package
*returns*, and the two sensor-tilt arguments, are in **radians**. DOP and radiance are
dimensionless.

## Quick start

```python
from astropy.coordinates import EarthLocation
from astropy.time import Time

from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern

# 1. Describe the analyzer mosaic: which pixels of each 2x2 tile carry which
#    polarizer orientation. This layout is the common "0 45 / 90 135" pattern.
wire_grid_orientations_slicing = {
    0:   SlicingPattern(start_row=0, start_column=0, step=2),
    45:  SlicingPattern(start_row=0, start_column=1, step=2),
    90:  SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}

# 2. Build the camera. See "Configuring the camera" for every parameter.
engine = Engine(
    sensor_pixel_pitch_micrometers=2.2,
    lens_conjugation_type="thin",
    number_pixels_vertical=64,
    number_pixels_horizontal=64,
    lens_focal_length_micrometers=3500,
    polarizer_tolerance_radians=0.0,
    extinction_ratio=0.99,
    auto_exposure_saturation_fraction=0.9,
    adc_resolution_bits=12,
    multiplicative_noise_snr=50,
    wire_grid_orientations_slicing=wire_grid_orientations_slicing,
    random_seed=0,
)

# 3. Ask the lens which sky direction each pixel looks at. Both grids are
#    64 x 64 arrays of degrees. altitude_min_clip=0 discards below-horizon rays.
azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)

# 4. Simulate the sky itself. A list of times keeps an explicit time axis;
#    a scalar or one-dimensional Time object works too.
observation_location = EarthLocation(lat=53.4, lon=-2.96, height=50)

sky_parameters, names = engine.simulate_sky_polarization(
    sky_model="QUEEN",
    observation_location=observation_location,
    times=Time(["2024-07-01T12:00:00"]),
    cie_sky_type=4,
    altitudes=altitudes,
    azimuths=azimuths,
    model_options={"dop_max": 1.0},
)
sky = dict(zip(names, sky_parameters))

# 5. Push the ideal sky through the sensor. Every AOP that Engine returns is
#    already in the analyzer frame, so it can be fed straight in.
measurement = engine.simulate_measurement(
    degree_of_polarization=sky["degree of polarization"],
    angle_of_polarization=sky["angle of polarization"],
    radiance=sky["radiance"],
)

print(sky["degree of polarization"].shape)  # (1, 64, 64)  one time, full pixel grid
print(measurement["dop"].shape)             # (1, 32, 32)  one time, demosaicked grid
```

**Why the measured grid is half the size.** Each 2×2 tile of the sensor holds four different
analyzer orientations, and all four are needed to recover one polarization state. A 64×64
sensor therefore yields a 32×32 map of reconstructed DOP and AOP. This is the real behaviour
of a division-of-focal-plane camera, not an approximation.

## How a simulation flows

```
   Engine.get_initial_          Engine.simulate_sky_          Engine.simulate_
   azimuth_altitude()           polarization()                measurement()
  ┌────────────────────┐      ┌─────────────────────┐      ┌────────────────────┐
  │ the lens maps each │      │ the sky model gives │      │ Malus's law, pixel │
  │ pixel to a viewing │ ───▶ │ DOP, AOP, radiance  │ ───▶ │ defects, noise,    │
  │ direction          │      │ for every direction │      │ ADC, then Stokes   │
  └────────────────────┘      └─────────────────────┘      └────────────────────┘
     azimuths, altitudes          sky fields                  measured fields
        (H, W) degrees            (T, H, W) float64           (T, H/2, W/2) float32
```

`T` is the number of observation times, `H × W` the pixel grid. Steps 1 and 2 are
independent of the camera hardware; you can supply your own direction grids instead of
calling `get_initial_azimuth_altitude`, as long as they are two-dimensional and in degrees.

The three underlying layers are also usable on their own:

| Module | Contains | Use it directly when |
|---|---|---|
| [`pyskylumos.sky_models`](src/pyskylumos/sky_models/) | The six models `Rayleigh`, `DepolarizedRayleigh`, `AsymmetricQuartic`, `Berry`, `Pan`, `QuEEN`; the two warnings `NeutralPointRangeWarning` and `PanFidelityWarning` | You want the physics only, with no camera involved |
| [`pyskylumos.sensor`](src/pyskylumos/sensor/) | `OpticalConjugator`, `MicroPolarizer`, `SensorChip`, `StokesCalculator`, `SlicingPattern` | You want to model one stage of the imaging chain |
| [`pyskylumos.engine`](src/pyskylumos/engine/) | `Engine` | You want the whole pipeline, with frames and camera pose handled for you — this is the recommended entry point |

`sky_models` also exports two abstract base classes, `SkySimulator` and its quartic subclass
`QuarticSkyModel`. They are the extension points for writing a new model, not simulators you
can instantiate; every concrete model above derives from one of them.

One caveat when using `sky_models` directly: `Pan` returns its angle of polarization in
Pan's own local-meridian reference frame, which is *not* the frame the sensor expects. See
[AOP reference frames](#aop-reference-frames). `Engine` converts it for you.

## Choosing a sky model

All six models share the same geometry and the same CIE radiance; they differ in **where
they put the neutral points** and **what quantity they report as the degree of
polarization**.

| `sky_model` | Neutral points | Reported degree of polarization | Accepted `model_options` |
|---|---|---|---|
| `"RAYLEIGH"` | Two, exactly at the sun and the anti-sun | `sin²γ / (1 + cos²γ)`, the textbook single-scattering law | None accepted |
| `"DEPOLARIZED_RAYLEIGH"` | Two, exactly at the sun and the anti-sun | `sin²γ / [1 + cos²γ + 4(1 − Δ)/(3Δ)]`, Rayleigh scattering with molecular anisotropy | `depolarization_ratio` |
| `"ASYMMETRIC"` (`"ASQ"`) | Four, with independently configurable anti-solar distances along the solar meridian | `dop_max · abs(ω) / (2 − abs(ω))`, using a normalized half-angle modulus | `arago_offset`, `fourth_offset`, `normalisation`, `dop_max`, `out_of_range` |
| `"BERRY"` | Four, at a fixed 15° above and below both the sun and the anti-sun | `abs(ω)`, Berry's normalized *intensity of polarization* | None accepted |
| `"PAN"` | Four, at distances that vary with solar elevation (Pan's measured fits) | `abs(ω)`, exactly as Pan Eq. (14) prints it | `out_of_range` |
| `"QUEEN"` | Four, at the same Pan-derived distances | `dop_max · abs(ω) / (2 − abs(ω))`, the OpenSky intensity-to-DoLP remap | `dop_max`, `out_of_range` |

Here `γ` is the scattering angle and `ω` is Berry's complex quartic polarization field, whose
four zeros *are* the four neutral points. `abs(ω)` denotes its modulus.

**In plain terms:**

- **`"RAYLEIGH"`** is the physical baseline. Single scattering off air molecules, no fitting,
  no free parameters. Its neutral points sit exactly on the sun and anti-sun, which is why
  real skies — where they sit away from those positions — need a split-neutral-point model.
- **`"DEPOLARIZED_RAYLEIGH"`** adds the established anisotropic-molecule correction of Wu
  et al. (2014). At the default dry-air approximation `depolarization_ratio=0.0279`, it
  suppresses peak DOP by **5.43%** and changes the shape by only **2.79%** relative to a pure
  constant rescale of Rayleigh. This is a percent-level correction, not a new sky theory: it
  retains only the solar and anti-solar zeros and does not model aerosols, turbidity, ground
  albedo, horizon depolarization, or spectral variation of gas composition.
- **`"ASYMMETRIC"`** (alias **`"ASQ"`**) gives all four quartic neutral points independent
  positions along one covariant signed solar meridian. Pan's measured fits place Brewster and
  Babinet; options place Arago and the fourth point. The default breaks Berry's antipodal
  pairing and uses a deterministic numerical full-sphere scan to keep the modulus bounded.
- **`"BERRY"`** replaces the two Rayleigh neutral points with four, splitting each into a
  pair 15° apart. This is the structure actually seen in the sky. The split is fixed.
- **`"PAN"`** keeps Berry's field but moves the four points according to Pan et al.'s
  measured regression against solar elevation, so the pattern changes through the day. It
  reports Pan's published `abs(ω)`.
- **`"QUEEN"`** — **Qu**artic **E**levation **E**xplained **N**eutralities — is the same
  geometry as `PAN` but reports a *conventional* degree of linear polarization, obtained by
  running `abs(ω)` through OpenSky's remap. Use it when you want values that behave like a
  measurable DoLP, or when you need continuity with PySkyLumos 0.0.6.

> **Upgrading from 0.0.6:** the meaning of `sky_model="PAN"` changed. The model that shipped
> as `PAN` up to and including 0.0.6 is now called `QUEEN`, and reproduces those historical
> results exactly. `PAN` now selects the Pan-derived formulation described above. `"QEN"` is
> accepted as an alias of `"QUEEN"` for the name used in early drafts. The full migration —
> this rename, the five renamed constructor arguments and everything else that changed — is in
> the [changelog](CHANGELOG.md).

`model_options` is a plain dictionary forwarded to the chosen model's constructor. Passing an
option a model does not accept raises `InputTypeError` and lists the ones it does accept.

| Option | Models | Default | Meaning |
|---|---|---|---|
| `depolarization_ratio` | `DEPOLARIZED_RAYLEIGH` | `0.0279` | Molecular depolarization ratio on `[0, 0.5]`. Zero recovers ideal Rayleigh exactly; the default is a conventional dry-air approximation, not a universal wavelength- or composition-independent value. |
| `arago_offset` | `ASYMMETRIC` | `"brewster"` | Arago distance: `"brewster"`, `"babinet"`, or a finite constant in degrees. No published solar-elevation regression exists for this distance. |
| `fourth_offset` | `ASYMMETRIC` | `"brewster"` | Fourth-point distance under the same selector policy. The default follows one qualitative observation in Horváth and Varjú, not a fitted regression. |
| `normalisation` | `ASYMMETRIC` | `"peak"` | `"peak"` uses a deterministic numerical sphere scan and three local refinements; `"berry"` uses the generalized closed-form Berry scale and can exceed one when roots are asymmetric. |
| `dop_max` | `ASYMMETRIC`, `QUEEN` | `1.0` | Peak degree-of-polarization scale on `(0, 1]`. |
| `out_of_range` | `ASYMMETRIC`, `PAN`, `QUEEN` | `"warn"` | What to do when the solar elevation falls outside the 0°–64° range over which Pan's neutral-point fits were measured: `"warn"` emits `NeutralPointRangeWarning`, `"raise"` turns it into an error, `"ignore"` stays silent. Values are extrapolated, never clamped. |

## Configuring the camera

`Engine.__init__` takes twelve current parameters. They fall into three groups. Five older
names remain as deprecated aliases and are listed separately below.

### The lens

| Parameter | Meaning |
|---|---|
| `lens_conjugation_type` | Which projection maps a sky direction onto the sensor plane. See the table below. |
| `lens_focal_length_micrometers` | Focal length in micrometers. Must be positive. Together with the sensor size it sets the field of view. |
| `sensor_pixel_pitch_micrometers` | Centre-to-centre distance between neighbouring pixels, in micrometers. Must be positive. |
| `number_pixels_vertical`, `number_pixels_horizontal` | Pixel counts. Each must be divisible by the mosaic step (2), so that all four analyzer orientations sample the same number of pixels. |

`lens_conjugation_type` selects one of six projections. Writing `r` for the distance of a
pixel from the sensor centre, `f` for the focal length and `φ` for the zenith angle of the
ray that lands on it:

| `lens_conjugation_type` | Mapping | Notes |
|---|---|---|
| `"thin"` | `φ = arctan(r/f)` | Ordinary rectilinear ("pinhole") lens. Cannot reach the horizon at any finite sensor size. |
| `"stereographic"` | `φ = 2·arctan(r/2f)` | Conformal fisheye: preserves angles locally, so circles stay circles. |
| `"equi_angle"` | `φ = r/f` | Equidistant fisheye. Zenith angle is linear in radius — the projection Pan's camera uses. |
| `"equi_solid_angle"` | `φ = 2·arcsin(r/2f)` | Equal-area fisheye. Errors if the sensor radius exceeds `2f`, where the mapping is undefined. |
| `"orthogonal"` | `φ = arcsin(r/f)` | Orthographic fisheye. Errors if the sensor radius exceeds `f`. |
| `"custom"` | Whatever you supply | Pass a callable as `custom_lens_conjugation` to `Engine.get_initial_azimuth_altitude`. It receives `complex_sensor_plane` and `lens_focal_length_micrometers`, and must return a real float array of altitudes in radians with the same shape. |

`OpticalConjugator.get_azimuth_altitude` takes the same hook under the same name, so the
argument does not change when you drop from `Engine` down to the conjugator.
`custom_lens_conjugation_type` is accepted as a deprecated alias on the `Engine` method and
**is removed in 0.2.0**; see the [changelog](CHANGELOG.md).

For a fisheye that fills a square sensor with a 180° field of view, set
`lens_focal_length_micrometers = sensor_radius / (π/2)` with
`sensor_radius = pixel_pitch × (pixels − 1) / 2`. That is what the
[example scripts](examples/) do.

### The micro-polarizer array

`wire_grid_orientations_slicing` is a dictionary mapping each analyzer orientation, **in
degrees**, to the `SlicingPattern` that says which pixels carry it. The contract is strict,
because the Stokes reconstruction depends on it:

- The keys must be exactly `0`, `45`, `90` and `135`.
- Every `SlicingPattern` must use `step=2`.
- `start_row` and `start_column` must be `0` or `1`, and the four patterns must cover all
  four positions of the 2×2 tile without overlapping.

Anything else raises `ConfigurationError`. `SlicingPattern(start_row=1, start_column=0,
step=2)` means "rows 1, 3, 5, … and columns 0, 2, 4, …", i.e. NumPy's `[1::2, 0::2]`.

| Parameter | Meaning |
|---|---|
| `extinction_ratio` | Analyzer quality, on `[0, 1]`. `1.0` is a perfect polarizer; `0.0` passes light regardless of its polarization and destroys all signal. Real wire-grid arrays sit near `0.99`. |
| `polarizer_tolerance_radians` | Manufacturing defect bound, on `[0, π/2]`. Each pixel's analyzer is offset by a fixed random angle drawn uniformly from `[−t, +t]`. The defect map is static per `Engine` and depends on `random_seed`. `0.0` gives a perfect array. |

### The sensor chip

| Parameter | Meaning |
|---|---|
| `auto_exposure_saturation_fraction` | On `(0, 1]`. The brightest *finite* pixel in a frame is mapped to this fraction of ADC full scale, so `0.9` leaves 10% headroom. Exposure is derived independently for each time slice. |
| `adc_resolution_bits` | Unsigned ADC bit depth, 1 through 24. Counts run from `0` to `2^bits − 1`. The 24-bit ceiling keeps every count exactly representable in `float32`. |
| `multiplicative_noise_snr` | Positive signal-to-noise ratio for relative Gaussian noise, applied as `signal × (1 + N(0, 1/SNR))`. Higher is cleaner. This is proportional noise, not an additive read-noise floor. |
| `random_seed` | Optional non-negative integer, shared by the polarizer defects and the sensor noise. Set it to make a run reproducible; leave it `None` for fresh randomness. |

### Deprecated parameter names

Five constructor arguments were renamed in 0.1.0 to state their units and meaning. The old
names still work but emit `DeprecationWarning`, and **are removed in 0.2.0**. Passing both
names for the same quantity raises `ConfigurationError`.

| Old name (deprecated) | Current name | Why it changed |
|---|---|---|
| `sensor_pixel_size_square_micrometers` | `sensor_pixel_pitch_micrometers` | The value is a linear pitch, not an area |
| `tolerance` | `polarizer_tolerance_radians` | States both what it bounds and its unit |
| `pixel_saturation_ratio` | `auto_exposure_saturation_fraction` | It is an auto-exposure target, not a per-pixel ratio |
| `adc_resolution` | `adc_resolution_bits` | States the unit |
| `signal_to_noise_ratio` | `multiplicative_noise_snr` | States that the noise is multiplicative |

## What a simulation returns

`Engine.simulate_sky_polarization` returns a `(values, names)` pair: a sequence of arrays and
a tuple of labels in matching order. Zip them into a dictionary, as the quick start does.

There are two kinds of entry. **Field quantities** have one value per sampled direction and
per time, shape `(T, H, W)`. **Metadata** describes the geometry of the whole frame, so it has
shape `(T, 1, 1)` and broadcasts against the fields. All of them are `float64`.

| Name | Kind | Meaning |
|---|---|---|
| `degree of polarization` | Field | Normally bounded in `[0, 1]`; `ASYMMETRIC` with non-default `normalisation="berry"` is an analytical compatibility mode whose asymmetric modulus can exceed one. See the per-model table for the exact quantity. |
| `angle of polarization` | Field | Radians, in the active sensor/analyzer frame |
| `radiance` | Field | Relative CIE sky radiance, dimensionless and unnormalized |
| `scattering angle` | Field | Radians, angle between the sun and the sampled direction |
| `sun azimuth`, `sun elevation` | Metadata | Radians, position of the sun |
| `anti-sun azimuth`, `anti-sun elevation` | Metadata | Radians, the point diametrically opposite the sun |

`RAYLEIGH` and `DEPOLARIZED_RAYLEIGH` return these eight entries. The four quartic models
(`ASYMMETRIC`, `BERRY`, `PAN`, `QUEEN`) additionally report the position of each
of the four polarization singularities, giving sixteen entries in total instead of eight:

| Name | Classical name | Where it is |
|---|---|---|
| `above sun singularity point azimuth` / `... elevation` | Babinet point | Above the sun on the solar vertical |
| `below sun singularity point azimuth` / `... elevation` | Brewster point | Below the sun on the solar vertical |
| `above anti-sun singularity point azimuth` / `... elevation` | Arago point | Above the anti-sun; independently placed by `ASYMMETRIC`, otherwise Brewster's antipode |
| `below anti-sun singularity point azimuth` / `... elevation` | Second Brewster ("fourth") point | Below the anti-sun; independently placed by `ASYMMETRIC`, otherwise Babinet's antipode |

```python
print("Babinet point elevation:", sky["above sun singularity point elevation"])
print("Brewster point elevation:", sky["below sun singularity point elevation"])
print("Arago point elevation:", sky["above anti-sun singularity point elevation"])
```

The name tuples are immutable and stable, so they are safe to rely on as a schema.

`Engine.simulate_measurement` returns a dictionary with two `float32` arrays, `"dop"` and
`"aop"`, both of shape `(T, H/2, W/2)`. Reconstructed DOP is bounded to `[0, 1]`; pixels that
were dark or masked come back as `NaN`.

### Masking with NaN

`NaN` is reserved throughout the package to mean "no valid value here", and it propagates
end to end:

- `altitude_min_clip` masks every field below the given altitude — DOP, AOP, radiance and
  scattering angle all become `NaN` there. Metadata is never masked.
- A `NaN` in any input to `simulate_measurement` masks that pixel, and any reconstructed
  value that depends on it.
- Positive-infinite radiance is *not* a mask. It is an explicit over-range sentinel meaning
  "brighter than this simulation can express": wherever the analyzer transmits anything at
  all, the pixel saturates at ADC full scale. Over-range pixels are left out of the exposure
  calculation, so the finite pixels in the same frame stay usable. Negative infinity is
  invalid and raises.

Note the deliberate asymmetry in `altitude_min_clip`: the optical conjugator **clips** to it
(every pixel must yield a usable direction) while the sky models **mask** with it. Both
behaviours are pinned by tests; see [the conventions section](#altitude_min_clip-clips-in-the-conjugator-masks-in-the-sky-models).

## Sky brightness: the CIE sky types

`cie_sky_type` is an integer from 1 to 15 selecting one of the standard sky luminance
distributions of CIE S 011 / ISO 15469. It affects the `radiance` field only — never DOP or
AOP. Each type combines a **gradation** (how brightness changes from horizon to zenith) with
an **indicatrix** (how brightness changes around the sun).

| `cie_sky_type` | Standard description |
|---|---|
| 1 | Overcast, steep luminance gradation towards the zenith, azimuthal uniformity |
| 2 | Overcast, steep luminance gradation, slight brightening towards the sun |
| 3 | Overcast, moderate luminance gradation, azimuthal uniformity |
| 4 | Overcast, moderate luminance gradation, slight brightening towards the sun |
| 5 | Sky of uniform luminance |
| 6 | Partly cloudy, no gradation towards the zenith, slight brightening towards the sun |
| 7 | Partly cloudy, no gradation towards the zenith, brighter circumsolar region |
| 8 | Partly cloudy, no gradation towards the zenith, distinct solar corona |
| 9 | Partly cloudy, with the sun obscured |
| 10 | Partly cloudy, with a brighter circumsolar region |
| 11 | White-blue sky with a distinct solar corona |
| 12 | CIE Standard Clear Sky, low turbidity |
| 13 | CIE Standard Clear Sky, polluted atmosphere |
| 14 | Cloudless turbid sky with a broad solar corona |
| 15 | White-blue turbid sky with a broad solar corona |

For a clear sky, `12` is the standard choice; the examples in this repository use `4`. The
returned radiance is **relative**: it is normalized against the zenith and the solar
position, and carries no photometric unit.

## Where the sun comes from

The sun's position drives the entire pattern, so there are three ways to obtain it.

| How | What you pass | When to use it |
|---|---|---|
| Default | Nothing; just `times` and `observation_location` | Normal use. Astropy's built-in ephemeris is accurate to well under an arcminute for the sun. |
| High accuracy | `accuracy=True` | You need the JPL DE430 kernel. Requires `pip install "pyskylumos[jpl]"`. |
| Explicit | `sun_position=SkyCoord(...)` | You want a specific sun position — for reproducible figures, parameter sweeps over solar elevation, or comparison against a published figure. |

An explicit `sun_position` takes precedence over `accuracy` and bypasses ephemeris lookup
entirely, so it needs neither the optional dependency nor any downloaded kernel.

With `accuracy=True`, Astropy downloads and caches the DE430 kernel on first use. Running
offline therefore requires a pre-populated Astropy cache; if the kernel cannot be reached,
the package raises a `RuntimeError` that says so rather than silently falling back.

### Solar elevation range for `ASYMMETRIC`, `PAN` and `QUEEN`

Pan's neutral-point fits were regressed from measurements taken over solar elevations of
**0° through 64°**. Outside that window the linear fits are extrapolated and never clamped,
and `NeutralPointRangeWarning` is emitted. Choose the behaviour with
`model_options={"out_of_range": "warn" | "raise" | "ignore"}`, or promote it to an error
across a test session:

```console
pytest -W error::pyskylumos.sky_models.NeutralPointRangeWarning
```

A second, distinct warning fires above a solar elevation of about 75.95°, where the fitted
Babinet offset goes negative and the point crosses to the far side of the sun, making its
label meaningless. Both effects are properties of the published regression, not of this
implementation; see [section 6](#6-pans-empirical-neutral-point-offsets).

## Camera pose: rotation and tilt

By default the camera looks straight up, and the direction grid returned by
`get_initial_azimuth_altitude` is simultaneously sensor-local and world AltAz. Two separate
mechanisms change that, and they do genuinely different things.

### Uniform in-plane offset: `azimuth_rotation_angle`

A degree-valued analyzer offset, subtracted from AOP after every other transformation. It
models rotating the camera about its own optical axis, as far as the polarization angles are
concerned. It does **not** move the sampling grid, so the sky directions are unchanged.

### Physical 3D tilt: the paired tilt arguments

To tip the camera away from vertical, pass **both**
`sensor_azimuthal_tilt_radians` and `sensor_tilt_angle_radians` to
`simulate_sky_polarization`. Supplying only one raises `InputValidationError`; omitting both,
or passing zero for both, gives the untilted result exactly.

```python
sky_parameters, names = engine.simulate_sky_polarization(
    sky_model="QUEEN",
    observation_location=observation_location,
    times=Time(["2024-07-01T12:00:00"]),
    cie_sky_type=4,
    altitudes=altitudes,  # still the sensor-local grid from the conjugator
    azimuths=azimuths,    # Engine rotates it into world coordinates for you
    sensor_azimuthal_tilt_radians=0.0,
    sensor_tilt_angle_radians=0.35,
)
```

With tilt enabled, `Engine` does three things in order:

1. Treats the supplied grid as **sensor-local**, and rotates every ray into world
   coordinates.
2. Evaluates the sky model at those rotated world directions. `altitude_min_clip` is applied
   *after* the rotation, against world altitude, so the horizon mask stays physical.
3. Transports the world-frame AOP into the tilted analyzer frame.

Step 3 is the subtle one. **The correction is not a single angle added to the whole image.**
Each ray has its own stereographic tangent basis, so the shift varies from pixel to pixel;
the derivation is in [section 10](#10-conventions).

**The tilt convention.** It is right-handed. The horizontal rotation axis is
`(cos a, −sin a, 0)` in North-East-Up coordinates, for `a = sensor_azimuthal_tilt_radians`:
`a = 0` selects the North axis, and increasing `a` turns that axis towards West. At `a = 0`,
a positive `sensor_tilt_angle_radians` moves the East horizon towards the zenith and the
zenith towards West.

### Two direction-only helpers

`Engine.tilt_sensor` and `Engine.rotate_sensor` exist for backward compatibility and operate
on **directions only**. They cannot transform an AOP field, because no polarization values
are passed to them. If the result will enter the sensor pipeline, use the integrated tilt
arguments above instead.

| Method | Does | Does not |
|---|---|---|
| `tilt_sensor(azimuths, altitudes, azimuthal_tilt, tilt_angle)` | Rotates a direction grid from sensor to world coordinates. Angles in degrees; the two tilt parameters in radians. | Transport AOP into the tilted frame |
| `rotate_sensor(azimuths, rotation_angle)` | Adds a degree-valued yaw to a sampling-azimuth array | Alter AOP at all |

Geometric AOP transport covers ray geometry only. It does not model lens-induced
polarization, incidence-angle-dependent micro-polarizer response, or any other
Mueller-matrix optical effect.

## Errors and warnings

All package exceptions subclass `PySkyLumosError` **and** a matching built-in, so existing
`except ValueError` / `except TypeError` handlers keep working.

| Exception | Also a | Raised when |
|---|---|---|
| `ConfigurationError` | `ValueError` | An object is built with an invalid value — an unknown lens type, a broken analyzer mosaic, a missing required argument, both a deprecated and a current name for the same parameter |
| `InputValidationError` | `ValueError` | A runtime input has a valid type but an invalid value, rank or shape — DOP outside `[0, 1]`, negative radiance, a non-2D direction grid, only one of the two tilt arguments |
| `InputTypeError` | `TypeError` | A public input has the wrong Python type — a non-string `sky_model`, a non-`Time` `times`, an unaccepted `model_options` key |
| `PySkyLumosError` | `Exception` | Base class; catch this to catch every package error at once |

```python
from pyskylumos import ConfigurationError, InputValidationError, PySkyLumosError
```

Two failures raise **plain standard-library exceptions on purpose**, because they report a
broken environment rather than a misuse of the API. Both come from the optional JPL ephemeris
path, and neither is a `PySkyLumosError`, so `except PySkyLumosError` will not swallow them:

| Exception | Raised when |
|---|---|
| `ModuleNotFoundError` | `accuracy=True` was passed but the optional ephemeris dependency is absent. The message tells you to `pip install "pyskylumos[jpl]"` |
| `RuntimeError` | `accuracy=True` was passed and the DE430 kernel could not be downloaded or found in the Astropy cache. The package never silently falls back to the built-in ephemeris |

Three warnings signal situations that are legal but worth knowing about.

| Warning | Emitted by | Meaning |
|---|---|---|
| `NeutralPointRangeWarning` | `AsymmetricQuartic`, `Pan`, `QuEEN` | The solar elevation is outside the 0°–64° range Pan's fits were measured over, or above ~75.95° where the Babinet offset turns negative. Values are extrapolated. |
| `PanFidelityWarning` | `Pan.simulate_sky` called directly | The returned AOP is in Pan's published local-meridian frame and is **not** ready for the sensor. Emitted once per simulator. `Engine` suppresses it because it performs the conversion itself. |
| `DeprecationWarning` | Constructors | A pre-0.1 parameter name was used. It will be removed in 0.2.0. |

Both package warnings are importable:

```python
from pyskylumos.sky_models import NeutralPointRangeWarning, PanFidelityWarning
```

## Visual examples

The [`examples/`](examples/) folder holds six independently runnable scripts — one per sky
model. Each plots the theoretical DOP, AOP and relative CIE radiance beside the DOP and AOP
reconstructed by the simulated sensor, so you can see what the imaging chain does to the
ideal field.

```console
python -m pip install -e ".[examples]"
python examples/queen.py
```

Figures are written to `examples/output/` and then displayed. Use `--no-show` for a headless
run, or `--output PATH` to choose a destination. All six use the same deterministic setup: an
untilted 128×128 equi-angle fisheye, CIE sky type 4, a fixed sun at 137° azimuth and 33°
altitude, and a seeded 12-bit noisy sensor. See [`examples/README.md`](examples/README.md)
for the details.

## Quantifying models against a capture

[`quantification/`](quantification/) contains a repository script that evaluates all six
canonical models against a local Thorlabs polarization-camera capture. It reports corrected
180°-periodic AOP errors, DOP errors after one shared empirical 0.7 scale, and a separate
affine-aligned raw-image diagnostic. One globally selected CIE radiance type and one
capture-derived mask are reused for every model. That mask is bounded by the lens's usable
image circle, measured from the capture itself, so the camera rim never enters a score.

```console
python -m pip install -e ".[quantification]"
MPLBACKEND=Agg python quantification/calibrate_sun_time.py \
    --config quantification/capture.toml \
    --output-dir quantification/output
MPLBACKEND=Agg python quantification/quantify_models.py \
    --config quantification/capture.toml \
    --output-dir quantification/output
```

The calibration command is read-only unless `--write-config` is passed. It searches the raw
solar starburst independently of the polarization models and supports the corrected capture
minute of 15:05 UTC. Two further scripts characterize that result rather than produce it:
`sensitivity.py` re-derives the ranking across every declared assumption and attaches
moving-block bootstrap intervals, and `validate_time_recovery.py` measures the operating
envelope of the time recovery against rendered frames whose acquisition time is known exactly.
The large capture and generated reports are deliberately gitignored.
The scripts verify the local files against the hashes in `quantification/capture.toml` before
running. See the
[quantification guide](quantification/README.md) for extraction, calibration, metric and rank
definitions, output schemas, mask interpretation and the capture's limitations. In
particular, this single partly cloudy observation is not a clear-sky validation dataset and
cannot establish general model superiority.

## Limits of the sensor model

The sensor is a **normalized synthetic response, not a radiometrically calibrated camera.**
It is designed to reproduce the *structure* of polarization measurement error — mosaic
demosaicking, analyzer defects, quantization, proportional noise — not absolute photometry.

Concretely, this means:

- Exposure is derived **independently for every time frame**, so ADC counts from two frames
  are not comparable as radiances. A dim frame and a bright frame both fill the ADC range.
- There is no calibrated exposure time, gain, full-well capacity, shot noise or additive
  read-noise floor. Noise is purely multiplicative.
- Reconstructed DOP is clipped into `[0, 1]`. That guarantees a bounded, physical output but
  does not make the estimate unbiased: at low signal-to-noise, values pile up at 1.

The full equations, the exact reconstruction contract and the clipping rationale are in
[the conventions section](#normalized-sensor-response).

## Development

```console
git clone https://github.com/taciochi/pyskylumos
cd pyskylumos
python -m pip install -e ".[dev]"
pre-commit install
```

`pre-commit` runs `ruff-check --fix` and `ruff-format` on every commit. Everything else runs in
CI.

### What CI checks

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) has five independent jobs. They run in
parallel, so the order below is by purpose rather than by execution:

| Job | Environment | Commands |
|---|---|---|
| `tests` | Python 3.12, 3.13 and 3.14 | `python -m pytest -q -W error`, then `audit/independent_oracle.py`, `audit/runtime_audit.py`, `audit/readme_quickstart.py` and `audit/check_markdown_tables.py` |
| `minimum-dependencies` | Python 3.12 with `numpy==2.4.*` and `astropy==8.0.*` | `python -m pytest -q -W error`, proving the declared floors are real |
| `platform-smoke` | Python 3.14 on Ubuntu, macOS and Windows | `audit/clean_install_smoke.py` |
| `quality` | Python 3.12 | `ruff format --check`, `ruff check`, `mypy`, `python -m pytest --cov=pyskylumos --cov-branch --cov-fail-under=90`, `python -m pip check`, `pip-audit` |
| `distributions` | Python 3.12 | `python -m build`, `twine check dist/*`, `audit/check_distribution.py dist`, then installs the wheel and the sdist into fresh virtualenvs and re-runs the smoke test and quick start from outside the source tree |

Ruff and mypy are configured in `pyproject.toml`; both take their paths from there, so
`ruff format --check src tests audit quantification` and a bare `mypy` reproduce the job
exactly. Tests run with `-W error`, so any unexpected warning fails the build.

### The `audit/` scripts

`audit/` holds standalone verification scripts that are deliberately not part of the test
suite. They are excluded from the sdist, and each runs as `python audit/<script>.py` from the
repository root.

| Script | In CI | What it does |
|---|---|---|
| `independent_oracle.py` | Yes | Restates the source equations from scratch, importing nothing from `pyskylumos.sky_models`, and compares the result with all six public models |
| `runtime_audit.py` | Yes | Probes public API and sensor-pipeline failure modes |
| `readme_quickstart.py` | Yes | Executes this README's quick start for all six model names |
| `check_markdown_tables.py` | Yes | Structurally validates every table in every live Markdown file |
| `clean_install_smoke.py` | Yes | Public-API smoke test, written to run from outside the source tree against an installed wheel |
| `check_distribution.py` | Yes | Validates artifact size, contents, metadata and version agreement; takes the `dist` directory as an argument |
| `benchmark.py` | No | Times a reference workload and records peak RSS |
| `compatibility_006.py` | No | Compares the released 0.0.6 wheel against the frozen regression arrays |
| `dtype_probe.py` | No | Records the public output dtype of every Engine model path, reusing the engine that `clean_install_smoke.py` builds |

`check_markdown_tables.py` is why magnitudes are written `abs(z)` inside tables rather than with
vertical bars: a bare `|` would be read as a column delimiter. Equations outside tables use the
conventional notation. The script validates every non-hidden `*.md` file outside `audit/`,
`build/` and `dist/`, so this README, the changelog and the two subdirectory guides are all
covered.

### Running the tests offline

One test is marked `remote_data` because it downloads the JPL DE430 kernel. Deselect it when
you have no network access or no pre-populated Astropy cache:

```console
python -m pytest -q -W error -m "not remote_data"
```

The installed version is available as `pyskylumos.__version__` and always matches the
distribution metadata.

---

# Mathematical reference

This is the single mathematical reference for PySkyLumos. Every implemented formula is
attributed to a publication, a reference implementation, or an explicit package decision.
Where the implementation departs from a printed equation, the departure carries an errata
identifier and a justification. Nothing is changed silently.

You do not need this section to use the package. It exists so that a result produced by
PySkyLumos can be traced, line by line, back to the literature.

### Sections

1. [Scope and notation](#1-scope-and-notation)
2. [The shared geometry](#2-the-shared-geometry)
3. [Berry's quartic field](#3-berrys-quartic-field)
4. [Global AOP phase convention](#4-global-aop-phase-convention)
5. [Polarization quantities and the OpenSky DoLP conversion](#5-polarization-quantities-and-the-opensky-dolp-conversion)
6. [Pan's empirical neutral-point offsets](#6-pans-empirical-neutral-point-offsets)
7. [Model-by-model formula sheet](#7-model-by-model-formula-sheet)
8. [Errata — Pan et al. (2023)](#8-errata--pan-et-al-2023)
9. [Errata — Berry, Dennis and Lee (2004)](#9-errata--berry-dennis-and-lee-2004)
10. [Conventions](#10-conventions)
11. [What is and is not modelled](#11-what-is-and-is-not-modelled)

### Sources

| Key | Reference |
|---|---|
| **Berry 2004** | Berry M. V., Dennis M. R. and Lee R. L. Jr, *Polarization singularities in the clear sky*, New J. Phys. **6** (2004) 162, [doi:10.1088/1367-2630/6/1/162](https://doi.org/10.1088/1367-2630/6/1/162) |
| **Berry 2015** | Berry M. V., *Nature's optics and our understanding of light*, Contemp. Phys. **56** (2015) 2–16, [doi:10.1080/00107514.2015.971625](https://doi.org/10.1080/00107514.2015.971625) |
| **Wu 2014** | Wu L.-H., Zhang J., Fan Z.-G. and Gao J., *An analytical model for skylight polarization pattern with multiple scattering*, Acta Phys. Sin. **63** (2014) 114201, [doi:10.7498/aps.63.114201](https://doi.org/10.7498/aps.63.114201) |
| **Bodhaine 1999** | Bodhaine B. A., Wood N. B., Dutton E. G. and Slusser J. R., *On Rayleigh Optical Depth Calculations*, J. Atmos. Ocean. Technol. **16** (1999) 1854–1861, [doi:10.1175/1520-0426(1999)016%3C1854:ORODC%3E2.0.CO;2](https://doi.org/10.1175/1520-0426(1999)016%3C1854:ORODC%3E2.0.CO;2) |
| **Pan 2023** | Pan P., Wang X., Yang T., Pu X., Wang W., Bao C. and Gao J., *High-similarity analytical model of skylight polarization pattern based on position variations of neutral points*, Opt. Express **31**(9) (2023) 15189, [doi:10.1364/OE.489534](https://doi.org/10.1364/OE.489534) |
| **OpenSky 2024** | Moutenet A., Poughon L., Toulon B., Serres J. R. and Viollet S., *OpenSky: A Modular and Open-Source Simulator of Sky Polarization Measurements*, IEEE Trans. Instrum. Meas. **73** (2024), [doi:10.1109/TIM.2024.3374965](https://doi.org/10.1109/TIM.2024.3374965) |
| **OpenSky code** | Moutenet et al., [`Simu_Berry.py`](https://github.com/MoutenetA/OpenSky/blob/main/Python/Simu_Berry.py), the OpenSky reference implementation |
| **Wang 2016** | Wang et al., *An analytical model for the celestial distribution of polarized light, accounting for polarization singularities, wavelength and atmospheric turbidity*, J. Opt. **18** (2016) 065601, [doi:10.1088/2040-8978/18/6/065601](https://doi.org/10.1088/2040-8978/18/6/065601); nearest prior art rather than the source of this package's covariant construction |
| **Horváth and Varjú** | Horváth G. and Varjú D., *Polarized Light in Animal Vision*, chapters 4–5, especially Table 4.1 and pp. 24–31; four-point measurements and the qualitative fourth-point distance observation |

---

## 1. Scope and notation

| Symbol | Meaning |
|---|---|
| `φ` | Zenith angle of a sky direction: 0 at the zenith, π/2 at the horizon |
| `α` | Azimuth of a sky direction |
| `φ_s`, `α_s` | Zenith angle and azimuth of the sun |
| `α_p` | Azimuth of the observed point |
| `δ_s` | Solar **elevation**, equal to `90° − φ_s`, in degrees |
| `δ_B`, `δ_Ba` | Angular distance from the sun to the Brewster and Babinet points |
| `δ_Ar`, `δ_4` | Configured Arago and fourth-point distances in `ASYMMETRIC` |
| `ξ` | Stereographic coordinate of a sky direction; a complex number |
| `ω` | Berry's complex polarization field. `abs(ω)` is the intensity of polarization and `arg(ω)/2` is the polarization orientation |
| `A` | Berry's splitting parameter, `tan(δ/4)` |
| `γ` | Scattering angle between the sun and the observed point |
| `δ_m` | Molecular depolarization ratio used by `DEPOLARIZED_RAYLEIGH` |
| `Δ_m` | Molecular King factor, `(1 − δ_m) / (1 + δ_m/2)` |

**Units.** Angles are radians internally and degrees at the API boundary. `azimuths` and
`altitudes` are supplied in degrees; every returned angle — AOP, scattering angle and all
neutral-point metadata — is in radians. DOP and radiance are dimensionless.

**Notation inside tables.** `abs(z)` denotes the complex modulus, conventionally written with
vertical bars. Tables use the spelled-out form so that a magnitude is never misread as a
Markdown column delimiter; equations outside tables keep the conventional notation.

**Shapes.** For `T` observation times and an `H × W` sampling grid, field quantities have
shape `(T, H, W)` and metadata `(T, 1, 1)`. `Rayleigh` and `DepolarizedRayleigh` each return
four fields and four metadata entries; the four quartic models return the same four fields
and twelve metadata entries, the extra eight being the positions of the four singularities.

**Sign of the azimuth.** Azimuths follow the Astropy `AltAz` convention: 0 at North,
increasing towards East. All six models use `e^{+iα}`; see [section 10](#10-conventions) and
erratum E3 in [section 8](#8-errata--pan-et-al-2023).

---

## 2. The shared geometry

Implemented in [`StereographicQuartic.py`](src/pyskylumos/sky_models/StereographicQuartic.py).

Everything the four quartic models have in common lives here: projection, inverse projection,
signed solar-meridian placement, explicit-root fields and half-angle modulus evaluation.

### 2.1 The stereographic coordinate

> **Berry 2015, Eq. (37)** — `project()`

```
ξ(φ, α) = tan(φ/2) · e^{iα}
```

The zenith maps to the origin, the horizon to the unit circle `|ξ| = 1`, and the visible
hemisphere to the unit disc. The nadir maps to infinity. The map is **conformal**: it
preserves local angles, which is exactly what makes it legitimate to read a polarization
orientation directly off the plane.

The inverse, `unproject()`, is `φ = 2·arctan|ξ|` and `α = arg ξ`.

Berry writes the radius as `(1 − tan(θ/2)) / (1 + tan(θ/2))`, with `θ` the *elevation*. That
is the same quantity, since `tan(45° − θ/2) = tan(φ/2)`. PySkyLumos uses the zenith-angle
form because it composes directly with the half-angle identity in §2.3.

### 2.2 The antipodal map

> **Berry 2004, §2** — `antipode()`

```
ξ  ↦  −1 / ξ*
```

The reciprocal conjugate. This sends a direction to the diametrically opposite direction on
the sky sphere; the test suite verifies the separation is exactly 180°. Note the
**conjugate** — `−1/ξ` is a different map and is not antipodal.

### 2.3 Placing a point at an angular offset from the sun

> **Berry 2004, Eq. (2.6)** — `offset_along_solar_vertical()`

This is the identity that makes the neutral-point offsets *exact* angular distances rather
than an approximation, so it is worth deriving rather than asserting.

A point offset by `δ` from the sun along the solar vertical great circle has azimuth `α_s`
and zenith angle `φ_s ∓ δ` (minus above the sun, plus below). Its stereographic coordinate
follows straight from §2.1:

```
ξ = tan((φ_s ∓ δ)/2) · e^{iα_s}
```

Write `t = tan(φ_s/2)` and `a = tan(δ/2)`. The tangent addition and subtraction formulas
give `tan(x ± y) = (tan x ± tan y)/(1 ∓ tan x tan y)`, hence

```
above the sun:   ξ_Ba = e^{iα_s} · (t − a) / (1 + t·a)
below the sun:   ξ_B  = e^{iα_s} · (t + a) / (1 − t·a)
```

These are the Möbius forms printed by Berry and Pan. **The `a = tan(δ/2)` halving is the
stereographic projection itself, not a fudge factor.**

Berry's own parameter is `A = tan(δ/4)` because his `δ` is the *pairwise* separation between
the two points of a pair (`δ = 4·arctan A`, Berry Eq. 2.4), so each point sits at `δ/2` from
the sun and `A = tan((δ/2)/2) = tan(δ/4)`. Pan parameterises by the individual sun-to-point
distance instead, so his `A_± = tan(δ_±/2)`. The two are consistent; only the meaning of the
symbol `δ` changes.

---

## 3. Berry's quartic field

> **Berry 2004, Eq. (4.2)** — `omega()`

```
                −4 (ξ − ξ_B)(ξ − ξ_Ba)(ξ + 1/ξ_B*)(ξ + 1/ξ_Ba*)
ω(ξ)  =  ─────────────────────────────────────────────────────────────
          (1 + |ξ|²)² · |ξ_B + 1/ξ_B*| · |ξ_Ba + 1/ξ_Ba*|
```

The four zeros of this quartic *are* the four neutral points: Brewster (below the sun),
Babinet (above the sun), Arago (above the anti-sun, the antipode of Brewster) and the second
Brewster or "fourth" point (below the anti-sun, the antipode of Babinet).

`ω` encodes the polarization as the complex Stokes combination `⟨(E_x + iE_y)²⟩`
(Berry 2004, Eq. 2.1), so `|ω|` is the intensity of polarization and `arg(ω)/2` the
orientation of the polarization direction.

Three parts of the expression earn their place, and none of them is decoration:

* **`(1 + |ξ|²)²` in the denominator** makes `|ω|` *antipodally invariant*, so that
  `|ω(−1/ξ*)| = |ω(ξ)|`. Berry Eq. (2.7) without it has the right zeros but the wrong global
  modulus.
* **The two moduli in the denominator** normalize the peak to `max|ω| = 1`. This is verified
  numerically over the whole sphere, and — importantly, since Berry only claims it for the
  symmetric case — it holds for Pan's *asymmetric* offsets too, for solar elevations from 0°
  to 80°.
* **The leading `−4`** fixes the absolute orientation. Since `−1 = e^{iπ}` contributes `π/2`
  to `arg(ω)/2`, **dropping the minus sign rotates every AOP by exactly 90°.** A test pins
  it.

In the code, `omega()` takes only the two solar-side roots and constructs the anti-solar pair
itself, so the reciprocal-conjugate relationship cannot be broken by a typo in one branch.

---

## 4. Global AOP phase convention

The angle of polarization is evaluated as

```
AOP = ½ · arg[ω(ξ) · e^{-2iα_s}]
```

The `e^{-2iα_s}` factor is the *global-angle convention* used by the OpenSky reference
implementation in `Simu_Berry.py`. It is not printed in Berry or Pan, so PySkyLumos credits
OpenSky for the convention and documents separately why this package's arbitrary-solar-azimuth
coordinate system requires it. It is applied by `AsymmetricQuartic`, `Berry`, `QuEEN` and
`Pan` alike.

Pan Eq. (15) writes `ln(ω/|ω|)/(2i)`. Since `ω/|ω| = exp(i·arg ω)`, that is `arg(ω)/2` modulo
the axial period π, so `numpy.angle` evaluates the same phase without taking a complex
logarithm. The `e^{-2iα_s}` factor is a separate, additional correction.

### The physical requirement

Rigidly rotate the whole scene by `β` about the vertical: the sun moves `β` in azimuth and
every sky point moves with it. The polarization pattern is attached to the sun, so it turns
with the sun, and every polarization direction must turn by exactly `β`:

```
rotate the scene by β   ⇒   AOP → AOP + β
```

This is what "the pattern follows the sun" means. It also matters in practice, because the
solar azimuth sweeps through roughly 180° over a day.

### Why the bare quartic violates it

`ω` is a product of **four** factors `(ξ − ξ_k)`. Under the rotation, the evaluation point
becomes `ξe^{iβ}` and every root becomes `ξ_k e^{iβ}`, so each factor picks up `e^{iβ}`:

```
numerator    →  e^{4iβ} × numerator
denominator  →  unchanged        (it is built only from moduli)
```

Therefore `ω → ω·e^{4iβ}` and `arg(ω)/2 → arg(ω)/2 + 2β` — **twice** the required rotation.

The reason is that `ω` is a *squared* field and must gain `e^{2iβ}`; the quartic gains
`e^{4iβ}`, which is one spurious factor of `e^{2iβ}` too many. In this construction the roots
were rotated onto the sun by `e^{iα_s}`, so the spurious factor is exactly `e^{2iα_s}` — and
dividing it out is precisely what the convention above does.

Berry never met this problem because he writes his roots (Eq. 2.6) with the sun pinned to the
`+y` axis: his `ζ_±` are purely imaginary and `α_s` never varies. Pan generalises to
arbitrary solar azimuth (Eq. 12) without adjusting the phase; see erratum E9 in
[section 8](#8-errata--pan-et-al-2023).

### Measured behaviour

The table below rotates one fixed configuration and reports what each convention does to the
angle of polarization. The setup is:

| Setting | Value |
|---|---|
| Solar elevation `δ_s` | 30° |
| Solar azimuth at `β = 0` | 100° |
| Observed point | 40° zenith angle, held 55° in azimuth east of the sun |
| Neutral-point offsets | Pan's fits at `δ_s = 30°`: Babinet 25.73°, Brewster 49.34° |
| Scene rotation `β` | Applied to the sun and the observed point together |

Because AOP is an axis rather than an arrow, both the values and the changes are taken
modulo 180°.

| Scene rotation `β` | Uncompensated AOP | Uncompensated change | Compensated AOP | Compensated change | DOP |
|---|---|---|---|---|---|
| 0° | 78.73° | 0°, the reference | −21.27° | 0°, the reference | 0.53916 |
| 10° | −81.27° | **20°** | −11.27° | **10°** | 0.53916 |
| 20° | −61.27° | **40°** | −1.27° | **20°** | 0.53916 |
| 40° | −21.27° | **80°** | 18.73° | **40°** | 0.53916 |
| 120° | −41.27° | **60°**, because 240° mod 180° is 60° | −81.27° | **120°** | 0.53916 |

Read the two "change" columns against `β`: the compensated column delivers `Δ = β`, exactly
as the physical requirement demands, while the uncompensated column delivers `2β`. The last
row is not an exception — `2β = 240°`, which wraps to 60° on the 180° axial period. DOP is
identical in every row, which confirms this is purely a phase issue and touches nothing else.

### Two independent confirmations

1. **Rayleigh limit.** Forcing both offsets to `10⁻⁶` rad collapses the quartic to the
   single-scattering case. `½·arg[ω e^{-2iα_s}]` then reproduces this package's independently
   written `Rayleigh.__get_aop` exactly. The uncompensated form does not.
2. **Covariance.** `AOP(rotated) = AOP + β` holds to better than `10⁻⁹` degrees for
   `β ∈ {37°, 123°, 270°}`, with DOP bit-identical.

Both are asserted in `tests/test_stereographic_quartic.py`.

### Reference frames

`½·arg[ω e^{-2iα_s}]` is measured from the frame's x-axis, that is, from azimuth 0. Two of
the models then reference it differently:

| Model | AOP is measured from | Behaviour under a scene rotation `β` |
|---|---|---|
| `AsymmetricQuartic`, `Berry`, `QuEEN` | The fixed world stereographic frame | `AOP → AOP + β` |
| `Pan` | The local meridian, i.e. minus `α_p` (Pan Eq. 23) | **Invariant** — a meridian angle cannot change when the whole scene turns |

Conversion between them: `AOP_frame = AOP_pan + α_p (mod π)`.

This table describes the model constructors themselves. At the high-level API boundary,
`Engine.simulate_sky_polarization` applies that conversion to `Pan` automatically, so every
AOP returned by `Engine` is in the active analyzer frame and can enter the sensor pipeline
directly. Without 3D tilt, that analyzer frame is aligned with the fixed world chart. With
tilt, `Engine` additionally performs the spatially varying basis transport described in
[section 10](#10-conventions).

---

## 5. Polarization quantities and the OpenSky DoLP conversion

### What the sources actually say

Berry 2004 §4 normalizes `max|ω| = 1` and calls `|ω|` the *intensity of polarization* — the
**unnormalized** degree. He explicitly declines to divide by the daylight intensity. Pan
Eq. (14) sets `DoP = |ω|` directly. **Both published sources therefore use `|ω|` itself.**

`Berry` and `Pan` implement exactly that. For `Berry`, the public label stays
`degree of polarization` for result-schema compatibility, but the quantity is specifically the
paper's normalized-to-unit-maximum polarization intensity, not a physical DOP divided by
total daylight intensity.

### The OpenSky conversion, used by QuEEN

```
DOP = dop_max · |ω| / (2 − |ω|)
```

This is a compressive remap of `[0, 1]` onto `[0, 1]` that fixes both endpoints. Writing
`m = abs(ω)` for the normalized polarization intensity:

| `m = abs(ω)`, what Berry and Pan report | `m/(2−m)`, what QuEEN reports | Implied unpolarized fraction `U/E² = 2(1−m)` |
|---|---|---|
| 0.00 | 0.0000 | 2.0000 |
| 0.25 | 0.1429 | 1.5000 |
| 0.50 | **0.3333** | 1.0000 |
| 0.75 | 0.6000 | 0.5000 |
| 0.90 | 0.8182 | 0.2000 |
| 1.00 | 1.0000 | 0.0000 |

Reading the table: the mapping always pulls values *down* except at the two fixed endpoints,
most strongly in the middle, where a modulus of 0.5 becomes a DoLP of one third. The third
column is the unpolarized power that the derivation below implies for that modulus, in units
of `E²`; it falls to zero exactly when the light becomes fully polarized.

**Where the factor of 2 comes from.** It follows from the Rayleigh intensity decomposition
used by OpenSky, and is not a tuning constant. For a scattering angle `γ`, take the two
resolved intensities as

```
I_perpendicular = E²
I_parallel      = E² cos²γ
```

The polarized part is their difference, `P = I_perpendicular − I_parallel = E² sin²γ`. The
unpolarized contribution has equal power in two orthogonal transverse components, so
`U = 2 I_parallel = 2E² cos²γ`. With `m = P/E² = sin²γ`:

```
P/E² = m
U/E² = 2(1 − m)
DoLP = P/(P + U) = m/(2 − m)
```

So the coefficient 2 is fixed by the two-component unpolarized contribution, not chosen to
make plotted values look convenient. In the limit of zero neutral-point splitting,

```
sin²γ / (2 − sin²γ) = sin²γ / (1 + cos²γ),
```

which is exactly the Rayleigh DoLP. OpenSky extends this conversion by substituting Berry's
quartic `m = |ω|`. That extension is physically motivated and Rayleigh-consistent, but it is
**not** a multiple-scattering radiative-transfer derivation. QuEEN adopts the OpenSky
extension.

The OpenSky implementation also supplies a multiplicative `DoLP_Max`; PySkyLumos exposes it
as `dop_max`, defaulting to 1.0, on `QuEEN` and through
`Engine.simulate_sky_polarization(model_options={"dop_max": ...})`. Within PySkyLumos this
mapping belongs to QuEEN alone.

Because Berry's normalization bounds `|ω| ≤ 1`, both Berry's published modulus and QuEEN's
mapped output remain bounded. Tests assert this across solar elevations from −10° to 90°.

### Known limitation: the polarization maxima fall on the horizon

Berry 2004 §4 warns that Eq. (4.2) — equivalently Brewster's 1847 `|sin θ_P+ sin θ_P−|` —
predicts polarization maxima **on the horizon**, at the two points 90° from the
singularities, whereas observations show none there. Berry attributes the discrepancy to
strong multiple scattering near the horizon and proposes an ad-hoc *horizon function* `h(r)`
with `h(0) = 1` at the zenith and `h(1) = 0` at the horizon, for example
`h(r) = cos(πr/2)^{1/10}`.

**PySkyLumos inherits this limitation and does not correct it.** Measured at solar elevation
30° with Pan's offsets, `max|ω| = 1.00000` occurs at an altitude of about 0°, 90° from the
solar meridian; along the horizon `|ω|` runs from 0.2736 to 1.0000. QuEEN's `|ω|/(2−|ω|)` map
does **not** help, because it depends on `|ω|` alone and not on position: where `|ω| = 1` on
the horizon, the mapped DOP is still 1.

In practice, passing `altitude_min_clip` excludes the affected band. The behaviour is pinned
by `test_polarization_maximum_falls_on_the_horizon` so that it is documented rather than
surprising. Adding `h(r)` remains an open option.

---

## 6. Pan's empirical neutral-point offsets

Implemented in [`NeutralPointOffsets.py`](src/pyskylumos/sky_models/NeutralPointOffsets.py),
and used by `AsymmetricQuartic`, `Pan` and `QuEEN`.

> **Pan 2023, Eq. (26)** — `babinet_offset_deg()`
> ```
> δ_Babinet(δ_s) = 42.53 − 0.56·δ_s
> ```

> **Pan 2023, Eq. (25)** — `brewster_offset_deg()`
> ```
> δ_Brewster(δ_s) = 37.34 + 0.49·δ_s      (δ_s ≤ 27°)
> δ_Brewster(δ_s) = 56.84 − 0.25·δ_s      (δ_s > 27°)
> ```

Both are in degrees and are functions of solar elevation `δ_s`, also in degrees. They were
regressed from measurements taken on 5 August 2022 at Hefei University of Technology.

### Measured range: 0° to 64° of solar elevation

Outside that range the linear fits are **extrapolated, never clamped**.
`check_elevation_range()` reports the condition according to an `out_of_range` policy —
`'warn'` (the default), `'raise'` or `'ignore'` — by emitting `NeutralPointRangeWarning`. To
make extrapolation a hard error across a test session:

```console
pytest -W error::pyskylumos.sky_models.NeutralPointRangeWarning
```

### Three properties of the published fit, reproduced as printed

**The 27° discontinuity.** At the breakpoint the two branches of Eq. (25) give 50.57° and
50.09°, a jump of **0.48°**. The breakpoint itself takes the lower branch (`δ_s ≤ 27`), as
printed. The effect on the resulting fields is mild and local: across `δ_s = 26.999°` to
`27.001°`, `ΔDOP ≤ 0.006` and `ΔAOP` has a median of 0.15°, a 99th percentile of 0.74° and a
maximum of 4.98°. That maximum occurs close to a neutral point, where the polarization
direction is undefined and therefore turns quickly. No smoothing is applied: the
discontinuity is a property of the published regression.

**A negative Babinet offset above 75.9464°.** `δ_Babinet` crosses zero at
`42.53/0.56 = 75.9464…°`. Above that, the fitted "Babinet" point crosses to the far side of
the sun and its label stops being meaningful. A second, distinct warning reports this case.

**The offsets are large.** Across the measured range, `δ_Babinet` runs from 42.53° down to
6.69°, and `δ_Brewster` runs from 37.34° up to 50.57° and back down to 40.84°. Those are well
beyond the 10–20° reported in the classical literature, and beyond Pan's own Monte-Carlo
section (7–17° and 10–22° for `δ_s` between 30° and 60°). One consequence is directly
observable: at a solar elevation of 33°, the Brewster point sits at an altitude of
**−15.59°**, below the horizon, so only the Babinet and Arago points are visible. This is a
property of the published fit, noted here because it is surprising.

---

## 7. Model-by-model formula sheet

Throughout: `t = tan(φ_s/2)` and `a = tan(δ/2)`, as derived in §2.3.

The **Modification** column says how the implementation relates to the printed original.
"None; implemented directly" means the printed equation is used unchanged. Anything else
cites an errata identifier from [section 8](#8-errata--pan-et-al-2023) or
[section 9](#9-errata--berry-dennis-and-lee-2004), or states an explicit package decision.

### Rayleigh — [`Rayleigh.py`](src/pyskylumos/sky_models/Rayleigh.py)

| Quantity | As implemented | Source | Modification |
|---|---|---|---|
| Scattering angle | `SkyCoord.separation` | Great-circle geometry | Replaces Pan Eq. (1), which as printed carries a spurious outer `cos` (**E1**) |
| AOP | `arctan(tan(arctan(N/D) + α_p))`, with `N = sin φ_p cos φ_s cos Δα − cos φ_p sin φ_s` and `D = cos φ_s sin Δα` | Single-scattering theory | Frame-referenced. Not Pan Eq. (2), which is dimensionally impossible as printed (**E2**) |
| DOP | `sin²γ / (1 + cos²γ)` | Single-scattering theory | None; implemented directly |
| Neutral points | Two, at the sun and the anti-sun | Single-scattering theory | None; this is the baseline geometry |

### Depolarized Rayleigh — [`DepolarizedRayleigh.py`](src/pyskylumos/sky_models/DepolarizedRayleigh.py)

This is the anisotropic-molecule phase-matrix correction in Wu 2014, Eqs. (3)–(5). For a
molecular depolarization ratio `δ_m`, define the King factor

```text
Δ_m = (1 − δ_m) / (1 + δ_m/2)
```

and evaluate the direct analytical law

```text
DOP(γ) = sin²γ / [1 + cos²γ + 4(1 − Δ_m)/(3Δ_m)].
```

| Quantity | As implemented | Source | Modification |
|---|---|---|---|
| Scattering angle | Inherited unchanged from `Rayleigh` | Great-circle geometry | None |
| AOP | Inherited unchanged from `Rayleigh` | Single-scattering theory | Fixed-world-chart convention |
| DOP | `sin²γ / [1 + cos²γ + 4(1 − Δ_m)/(3Δ_m)]` | Wu 2014 Eqs. (3)–(5) | Only the first, molecular-scattering phase-matrix term is implemented |
| Default ratio | `δ_m = 0.0279` | Bodhaine 1999 | Conventional dry-air approximation; configurable on `[0, 0.5]` |
| Neutral points | Two, at the sun and anti-sun | Inherited from `Rayleigh` | No Babinet, Brewster or Arago splitting |

At `δ_m = 0`, `Δ_m = 1` and the equation reduces element for element to ideal Rayleigh. At
the default, `Δ_m = 0.958725775`, the additive anisotropy term is `0.057401502`, and the
maximum DOP at `γ = 90°` is `0.945714564`. No clipping is needed: the denominator is
analytically at least the numerator over the validated interval.

The model changes DOP only. It inherits Rayleigh AOP, CIE radiance, scattering angle,
masking, and its eight-field return schema. Wu's later second-scattering approximation is
not included because it needs an aerosol Mueller term and constants without a general
published parameter mapping. Consequently this model is a percent-level molecular
correction—5.43% peak suppression and only 2.79% shape departure from a pure constant
rescale—not a multiple-scattering, turbidity, or spectral atmosphere model.

### Berry — [`Berry.py`](src/pyskylumos/sky_models/Berry.py)

| Quantity | As implemented | Source | Modification |
|---|---|---|---|
| Coordinate | `ξ = tan(φ_p/2) e^{iα_p}` | Berry 2015 Eq. (37) | None; implemented directly |
| Offsets | `δ_B = δ_Ba = 15°`, i.e. `A = tan(30°/4)` | Berry 2004 Eq. (2.4) | Separation fixed at 30°, the midpoint of the range Berry reports |
| Roots | `(t ± a)/(1 ∓ t·a) · e^{iα_s}`, with antipodes `−1/ξ*` | Berry Eqs. (2.6) and (2.7) | None; implemented directly |
| Field | `ω`, Eq. (4.2), including the leading `−4` | Berry Eq. (4.2) | None; implemented directly |
| AOP | `½·arg[ω e^{-2iα_s}]` | OpenSky `Simu_Berry.py` | Global fixed-world-chart convention; covariance and Rayleigh-limit behaviour are independently tested (§4) |
| DOP-labelled output | `abs(ω)` | Berry Eq. (2.1), §4 and Eq. (4.2) | The published intensity of polarization, an unnormalized degree; not divided by total daylight intensity |
| Metadata | `directional_offset_by`, at ±15° | Astropy AltAz geometry | Sun and anti-sun azimuth and elevation were transposed before 0.1.0 (**B1**) |

### Pan — [`Pan.py`](src/pyskylumos/sky_models/Pan.py)

| Quantity | As implemented | As published | Modification |
|---|---|---|---|
| Coordinate | `ξ = tan(φ_p/2) e^{+iα_p}` | `ξ = r e^{−iφ}`, Eq. (7) | **E3** — handedness paired with Pan's clockwise camera azimuth; `ω` is provably unchanged |
| Solar radius | `t = tan(φ_s/2)` | `y_s = r·cos δ_s`, Eq. (9) text | **E5** — the printed form mixes stereographic and orthographic radii |
| Offset parameter | `a = tan(δ_±/2)` | Eq. (16) | None; implemented directly |
| Below-sun root | `e^{iα_s}(t + a_B)/(1 − t·a_B)` | Eq. (12), labelled Babinet | **E8** — the printed pairing puts Brewster past the zenith |
| Above-sun root | `e^{iα_s}(t − a_Ba)/(1 + t·a_Ba)` | Eq. (12), denominator `(1 − a y_s)` | **E6** — sign error in the printed denominator |
| Anti-solar roots | `−1/ξ*` | Eq. (13), fourth factor `(µ + µ*_−)` | **E7** — missing reciprocal |
| Field | `ω`, Eq. (4.2) with the `−4` | Eq. (13) | Replaced by Berry's normalized Eq. (4.2) field |
| Offsets | Eqs. (25) and (26) | Eqs. (25) and (26) | Range and discontinuity documented (**E17**, **E18**) |
| **DOP** | **`abs(ω)`** | **Eq. (14)** | None; implemented as printed |
| **AOP** | **`wrap[½·arg(ω e^{-2iα_s}) − α_p]`** | Eqs. (15), (23) and (24), with OpenSky's global phase convention | `ln(ω/abs(ω))/(2i)` is evaluated as `arg(ω)/2` modulo π. **E9** adds the global phase; **E13** uses Eq. (20)'s meridian rather than Eq. (23)'s complement |
| Metadata | `directional_offset_by` at Pan's distances | Astropy AltAz geometry | None; reports Pan's interpreted angular offsets |
| Camera mapping | Not applied; use `OpticalConjugator('equi_angle')` | Eqs. (17)–(20) | **E11**, **E12**, **E15** |

### QuEEN — [`QuEEN.py`](src/pyskylumos/sky_models/QuEEN.py)

**QuEEN — Quartic Elevation Explained Neutralities — is a hybrid analytical variant, not a
wholly independent physical theory.** Berry supplies the quartic field and its normalization,
Pan supplies the elevation-dependent empirical offsets of Eqs. (25) and (26), and OpenSky
supplies the global AOP convention and the intensity-to-DoLP conversion.

| Quantity | As implemented | Source | Modification |
|---|---|---|---|
| Coordinate | `ξ = tan(φ_p/2) e^{iα_p}` | Berry 2015 Eq. (37) | None; implemented directly |
| Offsets | `δ_Brewster(δ_s)`, `δ_Babinet(δ_s)` | Pan Eqs. (25) and (26) | Elevation-dependent, rather than Berry's fixed split |
| Roots | Exact half-angle placement (§2.3), with antipodes `−1/ξ*` | Berry Eqs. (2.6) and (2.7) | None; implemented directly |
| Field | `ω`, Eq. (4.2), including the leading `−4` | Berry Eq. (4.2) | None; implemented directly |
| AOP | `½·arg[ω e^{-2iα_s}]`, referenced to the fixed world frame | OpenSky `Simu_Berry.py` | Applied to the Pan-positioned Berry field (§4); Engine transports it to a tilted analyzer frame when requested |
| DOP | `dop_max·abs(ω)/(2−abs(ω))`, with `dop_max` defaulting to 1.0 | OpenSky 2024 and `Simu_Berry.py` | Applied to the Pan-positioned Berry field (§5) |
| Metadata | The inverse projection of the quartic's own four roots | The internal quartic roots | Reported positions are by construction the zeros used to build the field |

### The exact relationship between Pan and QuEEN

The two direct model classes place their four roots identically. Their field relations are:

```
AOP_queen = AOP_pan + α_p            (mod π)
DOP_queen = DOP_pan / (2 − DOP_pan)  (at dop_max = 1)
```

`Engine` adds `α_p` to Pan's local-meridian AOP before returning it. Consequently, for the
same geometry, `Engine` returns **identical** Pan and QuEEN AOP fields and identical
theoretical CIE radiance; their only substantive returned-field difference is DOP. If QuEEN's
OpenSky remap were replaced by `DOP = |ω|` with unit peak scaling, Pan and QuEEN would also
return identical Engine-level DOP. These relationships are asserted in `tests/test_pan.py`
and `tests/test_engine_model_dispatch.py`.

### AsymmetricQuartic — [`AsymmetricQuartic.py`](src/pyskylumos/sky_models/AsymmetricQuartic.py)

`ASYMMETRIC` is PySkyLumos's covariant explicit-root realization of the asymmetric extension
discussed in OpenSky §V: each neutral point has its own signed position along the solar
meridian. Pan Eqs. (25)–(26) still determine the solar Brewster and Babinet distances. Arago
and the fourth point use the corresponding fitted distance selected by `arago_offset` and
`fourth_offset`, or a finite constant in degrees.
With the defaults, both anti-solar distances use the fitted Brewster distance. This does not
place the roots on top of one another: Arago and the fourth point lie on opposite sides of the
anti-sun, each at that same angular distance.

For solar zenith `φ_s` and azimuth `α_s`, roots use one signed coordinate:

```text
ξ(ψ) = tan(ψ/2) exp(iα_s)
ψ = (φ_s + δ_B, φ_s − δ_Ba, φ_s − π + δ_Ar, φ_s − π − δ_4).
```

The phase is the argument of the explicit quartic. Its modulus is the half-angle product

```text
S(P) = product over k of sin(Γ_k/2),
cos Γ_k = cos φ_p cos ψ_k + sin φ_p sin ψ_k cos(α_p − α_s).
```

The half-angle is essential. A full-angle product would be antipodally invariant even when
the configured roots are not and would manufacture four false neutral points at their
antipodes. Tests explicitly reject that alternative.

| Quantity | As implemented | Provenance and limitation |
|---|---|---|
| Roots | Four independently placed signed-meridian roots | PySkyLumos construction using Berry stereographic geometry, OpenSky §V motivation and Pan fits for the solar pair |
| AOP | `½·arg[ω exp(−2iα_s)]` | Fixed-world convention inherited from Berry and OpenSky |
| DOP | `dop_max · M / (2 − M)` | OpenSky remap, where `M` is the normalized half-angle modulus |
| `"berry"` normalization | Generalized Berry chord/conformal scale | Closed form and exact in the antipodally paired limit; asymmetric configurations can exceed one |
| `"peak"` normalization | Inverse maximum from a deterministic 257×512 sphere scan and three 65×65 refinements | Default and bounded by construction, but numerical rather than closed form |

With `arago_offset="brewster"`, `fourth_offset="babinet"` and
`normalisation="berry"`, the model recovers QuEEN at machine precision. Its raw normalized
field also recovers Berry when the solar distances are fixed at Berry's 15° split.

There is no published regression of Arago or fourth-point distance against solar elevation.
The default `fourth_offset="brewster"` rests on Horváth and Varjú's qualitative observation
that the fourth point lies at about the Brewster distance, not on a fitted law. Validation in
this repository establishes analytic limits, topology and literature anchors. The
repository-local quantification workflow adds one exploratory, partly cloudy capture, but
that single observation does not establish a better general real-sky fit than QuEEN.

---

## 8. Errata — Pan et al. (2023)

Eighteen points in Pan et al. (2023) where the printed text and this implementation differ,
or where the paper is ambiguous enough that the reading has to be stated. Every entry gives
the equation, the problem, the mathematically consistent reading, and what PySkyLumos does.
**Nothing in this section is applied silently.**

Each entry carries one of four outcomes:

| Outcome | Meaning |
|---|---|
| **Replaced** | PySkyLumos does not use the printed equation at all; it uses an independent formulation |
| **Corrected** | The printed equation is used, with a documented correction |
| **As printed** | Implemented exactly as published, with the consequence documented |
| **Documented** | No code is affected; the point is recorded for readers of the paper |

### Index

| ID | Pan equation | The issue in one line | Outcome |
|---|---|---|---|
| **E1** | (1) | Spurious outer `cos` in the scattering-angle formula | Replaced |
| **E2** | (2) | An angle multiplies a ratio inside `arccos`, which is dimensionally impossible | Replaced |
| **E3** | (7) | The stereographic coordinate has the opposite handedness to Berry | Corrected |
| **E4** | (8) | The symbol `r` is overloaded across three different meanings | Documented |
| **E5** | (9) text | The solar radius mixes a stereographic radius with an orthographic factor | Corrected |
| **E6** | (9), (11), (12) | Sign error in the second root's denominator | Corrected |
| **E7** | (10), (13) | The fourth factor is missing its reciprocal, so it is not the antipode | Corrected |
| **E8** | (11) | Babinet and Brewster are paired to the wrong Möbius forms | Corrected |
| **E9** | (15) | The bare quartic is not rotation covariant, so AOP moves by `2β` | Corrected |
| **E10** | (14) | Not an error; recorded because `DoP = abs(ω)` is easy to mistake for a normalized DoLP | As printed |
| **E11** | (18) | The camera mapping assumes a normalized radius and a 180° field of view | Replaced |
| **E12** | (19) | The camera mapping is missing its `/(n/2)` normalization | Replaced |
| **E13** | (20) vs (23) | Two definitions of the same azimuth differ by 90° | Corrected |
| **E14** | (22) | `Q` and `U` are swapped inside the Stokes AOP formula | Documented |
| **E15** | (17) | A single-quadrant `arctan(y/x)`, superseded by Eq. (20) | Replaced |
| **E16** | §3.1 text | The text names `δ+` where it describes `δ−` | Documented |
| **E17** | (25) at 27° | The two branches disagree by 0.48° at the breakpoint | As printed |
| **E18** | (26) above 75.9464° | The fitted Babinet offset goes negative | As printed |

### Detail

**E1 · Eq. (1), the scattering angle — Replaced.**
As printed: `cos θ = cos(cos δ cos δ_s + sin δ sin δ_s cos(φ−φ_s))`. The outer `cos` is
spurious: its body is already the zenith-angle form of the great-circle law, whereas Fig. 1
defines `δ` as elevation. The consistent reading is
`cos θ = sin δ sin δ_s + cos δ cos δ_s cos(φ−φ_s)`. PySkyLumos uses neither form; it calls
`SkyCoord.separation`, which computes the exact great-circle separation and carries no risk
of a trigonometric-identity slip.

**E2 · Eq. (2), the Rayleigh AOP — Replaced.**
As printed: `AoP = arccos(sin(φ_s−φ)/sin θ × (π/2 − δ_s))`. This is dimensionally
impossible — an angle multiplies a ratio inside `arccos`. A consistent reading would be
`arccos[sin(φ_s−φ) cos δ_s / sin θ]`. PySkyLumos does not use it: `Rayleigh.py` keeps its own
`arctan` formulation, since Pan's Rayleigh recapitulation is background material rather than
part of his model.

**E3 · Eq. (7), handedness of the stereographic coordinate — Corrected.**
As printed: `ξ = r e^{2i(−½φ)} = r e^{−iφ}`, which is the opposite handedness to Berry
Eq. (2.2). It is self-consistent only alongside Pan's clockwise camera azimuth of Eq. (20).
PySkyLumos uses `ξ = tan(φ/2) e^{+iα}` in the AltAz frame. **`ω` is provably unchanged by
this:** all four roots and `ξ` rotate by the same `e^{iβ}` with `β = −90°`, and `e^{4iβ} = 1`.
Using `e^{+iα}` throughout keeps all six models sign-comparable with one another.

**E4 · Eq. (8), the overloaded symbol `r` — Documented.**
As printed: `r = (1 − tan(α_s/2))/(1 + tan(α_s/2))`. The symbol `r` carries three different
meanings across the paper: the observed point's radius in Eq. (7), the sun's radius here, and
the factor `(1+r²)²` in Eq. (10). PySkyLumos reads them as `r(φ) = tan(φ/2)`, `y_s = r(φ_s)`,
and `(1+r²)² = (1+abs(ξ)²)²` evaluated at the observed point, following Berry §2 and
Eq. (4.2).

**E5 · Eq. (9) text, the solar radius — Corrected.**
As printed: `y_s = r · cos δ_s`. This mixes a stereographic radius with an orthographic
factor, which destroys the exact half-angle addition of §2.3. The consistent reading is
`y_s = tan((90° − δ_s)/2)`. PySkyLumos uses `t = tan(φ_s/2)`, because only that form makes
`(y_s+a)/(1−a y_s) = tan((φ_s+δ)/2)`, i.e. an exact angular offset rather than an
approximation.

**E6 · Eqs. (9), (11), (12), the second root's denominator — Corrected.**
As printed the denominator is `(1 − a_− y_s)`. This is a sign error; it should be
`(1 + a_− y_s)`. PySkyLumos uses `(t − a)/(1 + t·a)`, which is Berry Eq. (2.6) and follows
from the tangent **subtraction** identity.

**E7 · Eqs. (10), (13), the missing reciprocal — Corrected.**
As printed the fourth factor is `(ξ + ξ*_−)`, or `(µ + µ*_−)` in Eq. (13). That point is not
the antipode of `ξ_−`; the reciprocal is missing, and the factor should read `(ξ + 1/ξ*_−)`.
PySkyLumos builds it with `antipode(ξ) = −1/ξ*`, following Berry Eqs. (2.7) and (4.2). The
correction is required for the antipodal invariance `abs(ω(−1/ξ*)) = abs(ω(ξ))`, which is
verified numerically to `5.6×10⁻¹⁶`.

**E8 · Eq. (11), the Babinet/Brewster pairing — Corrected.**
As printed, `µ_+ = (y_s + a_+)/(1 − a_+ y_s)` is labelled **Babinet**, with
`a_+ = tan(δ_+/2)` and `δ_+` the Babinet distance. But that Möbius form *increases* the
zenith angle, so the point it produces lies **below** the sun. With Pan's own fitted offsets,
the printed pairing puts Brewster about 42° *above* the sun — past the zenith once
`δ_s > 48°`. PySkyLumos places Babinet above the sun at `δ_+` and Brewster below it at `δ_−`,
which is the physically correct placement and matches Berry's explicit labelling of `ζ_+` as
Brewster.

**E9 · Eq. (15), rotation covariance of the AOP — Corrected.**
As printed: `AoP = ln(ω/abs(ω))/(2i)`. The logarithm itself is equivalent modulo π to
`arg(ω)/2`, so that part is fine. The problem is that the bare quartic is not rotation
covariant: rotating the scene by `β` sends `ω → ω e^{4iβ}`, so the AOP moves by `2β` instead
of `β` (see §4). PySkyLumos evaluates the logarithmic phase as `½·arg(ω)` and applies
OpenSky's global convention, giving `wrap[½·arg(ω e^{-2iα_s}) − α_p]`. OpenSky supplies the
phase convention; this package's covariance and Rayleigh-limit tests justify it for arbitrary
solar azimuth. Tests P5 and P6 pin the result.

**E10 · Eq. (14), `DoP = abs(ω)` — As printed.**
Not an error. It is recorded here because `abs(ω)` is Berry's *intensity of polarization*, an
unnormalized quantity, and is easy to mistake for a conventional DoLP. PySkyLumos retains
Pan's published modulus in the `PAN` model; Berry's denominator bounds it to `abs(ω) ≤ 1`.
The `QUEEN` model applies the OpenSky remap instead (§5).

**E11 · Eq. (18), the camera mapping — Replaced.**
As printed: `δ = (π/2)√(x²+y²)`. This silently assumes both a normalized radius and a 180°
field of view, and `δ` behaves as the zenith angle even though Fig. 1 calls it an elevation.
The consistent reading is `φ = ω_half · r/r_max`. PySkyLumos does not build the mapping into
the sky model at all: use `OpticalConjugator('equi_angle')`, since an equidistant fisheye is
exactly the projection that maps zenith angle linearly in radius.

**E12 · Eq. (19), the missing normalization — Replaced.**
As printed: `δ = ω√((x−n/2)² + (y−n/2)²)`. The `/(n/2)` is missing, and "the field of the
camera view" must mean the **half** field of view (70° for the 140° device described) for `δ`
to reach it at the image edge. The consistent reading is `φ = 70° · r/(n/2)`. Handled as in
E11, and asserted by
`test_pan_equidistant_camera_mapping_is_the_equi_angle_conjugation`.

**E13 · Eq. (20) versus Eq. (23), two azimuths 90° apart — Corrected.**
Eq. (20) defines `φ` clockwise from image-up, while Eq. (23) subtracts
`arctan[(y−n/2)/(n/2−x)]`. The two are **complementary**: their tangents multiply to 1, so
they differ by 90°. That matters, because AOP is defined modulo 180°. PySkyLumos subtracts
the point's own azimuth as defined by Eq. (20), i.e. `− α_p`, because Eq. (23)'s stated
purpose — "the angle to the local meridian" — requires the meridian direction, which is
Eq. (20)'s `φ`.

**E14 · Eq. (22), swapped Stokes arguments — Documented.**
As printed: `AoP = ½ tan⁻¹(Q/U)`. The standard convention is `½ tan⁻¹(U/Q)`; as printed the
AOP is 90° off. No change was needed in PySkyLumos: `StokesCalculator` already computes
`0.5 * arctan2(s2, s1)`, which is the standard four-quadrant form.

**E15 · Eq. (17), a single-quadrant arctangent — Replaced.**
As printed: `φ = arctan(y/x)` for `x, y ≥ 0`. This covers one quadrant only, and Eq. (20)
already supersedes it. PySkyLumos uses a four-quadrant azimuth via `numpy.angle`.

**E16 · §3.1 text, a mislabelled offset — Documented.**
The text says "δ+ decreased … in the range of 27° to 64°", but describes `δ_−` — that is the
branch Eq. (25) governs above 27°. Documentation only; no code is affected.

**E17 · Eq. (25) at 27°, the branch discontinuity — As printed.**
The two branches give 50.57° and 50.09° at the breakpoint, a 0.48° jump. PySkyLumos preserves
the published piecewise branches exactly, with `δ_s ≤ 27` taking the lower branch as printed.
Pinned by a test; the magnitude of the effect on the fields is documented in §6. Fidelity to
the publication is preferred over silent smoothing.

**E18 · Eq. (26) above 75.9464°, a negative offset — As printed.**
Above that elevation the fitted `δ_+` goes negative, so the "Babinet" point crosses below the
sun and its label stops being meaningful. PySkyLumos preserves the extrapolation but
identifies the range failure with a distinct warning. The region is far outside the measured
range in any case.

---

## 9. Errata — Berry, Dennis and Lee (2004)

**None found.** Berry Eqs. (2.1)–(2.7), (3.1)–(3.6) and (4.1)–(4.4), together with Berry 2015
Eqs. (37) and (38), were checked against their stated derivations and against numerical
evaluation. Each is internally consistent, and the labelling of `ζ_+` as Brewster (below the
sun) and `ζ_−` as Babinet (above) is correct in both papers.

This section exists so that a reader can tell that Berry was audited, not merely trusted.

Two points that are easy to misread but are correct as printed:

* Berry's `A = tan(δ/4)` uses `δ` for the **pairwise** separation, not the sun-to-point
  distance. Pan's `A_± = tan(δ_±/2)` uses the latter. Both are right; see §2.3.
* Berry Eq. (2.7) omits the `(1+r²)²` normalization present in Eq. (4.2). That is deliberate:
  §2 of the paper is about polarization *directions*, §4 about the *intensity*.

One defect in **this package's** Berry implementation is recorded for completeness:

| ID | Where | Problem | Status |
|---|---|---|---|
| **B1** | `Berry` metadata | The sun and anti-sun azimuth and elevation were transposed in the returned metadata | Fixed in 0.1.0 |

---

## 10. Conventions

### Sensor orientation: North right, East top

`OpticalConjugator` builds the sensor plane with `x` increasing left to right and `y`
increasing bottom to top, then takes `azimuth = atan2(y, x)`. In display order, with row 0
drawn at the top:

| Image position | Azimuth | Cardinal direction |
|---|---|---|
| Centre-right | 0° | **North** |
| Top-centre | +90° | **East** |
| Centre-left | 180° | South |
| Bottom-centre | −90° | West |

Azimuth therefore increases **counter-clockwise** in the rendered image, which is the natural
sense for an upward-looking camera: you are seeing the compass rose from underneath. Pinned
by `tests/test_optical_conjugator.py`.

This is the **mirror image** of Pan Eq. (20), whose azimuth runs clockwise from image-up.
Angles of polarization cannot be carried between the two conventions without a reflection,
which flips their sign. See erratum E3 in §8.

### `altitude_min_clip` clips in the conjugator, masks in the sky models

A deliberate asymmetry, because the two layers need different things from the same number:

* `OpticalConjugator.get_azimuth_altitude(altitude_min_clip=x)` **clips**. Every pixel must
  yield a usable direction, so below-horizon altitudes are clamped up to `x`.
* `SkySimulator.simulate_sky(altitude_min_clip=x)` **masks**. DOP, AOP, radiance and
  scattering angle all become `NaN` where `altitude ≤ x`. Metadata is never masked.

Both behaviours are pinned by tests.

### AOP reference frames

| Model | AOP is measured from | Returned range |
|---|---|---|
| `Rayleigh`, direct constructor | The fixed world chart, i.e. azimuth 0 | `(−π/2, π/2]` |
| `DepolarizedRayleigh`, direct constructor | The fixed world chart, i.e. azimuth 0 | `(−π/2, π/2]` |
| `AsymmetricQuartic`, direct constructor | The fixed world chart, i.e. azimuth 0 | `(−π/2, π/2]` |
| `Berry`, direct constructor | The fixed world chart, i.e. azimuth 0 | `(−π/2, π/2]` |
| `QuEEN`, direct constructor | The fixed world chart, i.e. azimuth 0 | `(−π/2, π/2]` |
| `Pan`, direct constructor | **The local meridian at each pixel** | `[−π/2, π/2)` |

`MicroPolarizer` evaluates `cos(2(AOP − analyzer_angle))` with analyzer angles in the active
sensor plane, so it expects AOP in that same frame. `Engine.simulate_sky_polarization`
converts Pan automatically with `AOP_world = AOP_pan + α_p (mod π)`, then transports that
world-chart angle into the sensor chart when tilt is enabled.

A direct `Pan.simulate_sky` call deliberately retains the published local-meridian quantity
and emits `PanFidelityWarning` once per simulator.
`Engine.convert_local_meridian_aop_to_sensor(aop, observed_azimuths)` performs only the first
of those two conversions, and is therefore sufficient for direct-model measurement **only**
when the sensor and world charts are aligned. Tilted-camera users should use the integrated
Engine path.

`Engine.simulate_sky_polarization(azimuth_rotation_angle=…)` applies a further uniform
in-plane analyzer offset, by subtracting the supplied degree-valued angle from AOP. It does
not rotate sampling directions and is distinct from 3D sensor tilt. For `Pan`, Engine first
converts the local-meridian angle to the world chart, then performs any tilt transport, and
applies this offset last.

### Transporting AOP into a tilted sensor frame

For a tilted camera, let `R` be the right-handed rotation from sensor coordinates to world
coordinates, and let a ray be `d_s` in the sensor chart and `d_w = R d_s` in the world chart.
Both charts are zenith-centred stereographic projections taken from the nadir:

```
(u, v) = (x, y) / (1 + z).
```

Engine lifts the world-chart AOP `χ_w` into the tangent plane, rotates that physical tangent
vector into sensor coordinates, and projects it back into the sensor chart:

```
p_w = J_inverse(d_w) (cos χ_w, sin χ_w)
p_s = R^T p_w
(du_s, dv_s) = J_forward(d_s) p_s
χ_s = atan2(dv_s, du_s)  (mod π).
```

Here `J_inverse` and `J_forward` are the differentials of the inverse and forward
stereographic projections. Because they depend on the sampled ray, `χ_s − χ_w` **varies
across the image** — which is why a tilt correction cannot be a single angle added to the
whole AOP field.

Ordering, for `Pan`: the local-meridian AOP is first converted using the rotated world
azimuth, then the sensor-basis transport is applied, and `azimuth_rotation_angle` is applied
last. The stereographic chart is singular at the exact nadir, where transported AOP is
returned as `NaN`; the zenith is regular.

### Sensor input and reconstruction domains

The sensor path accepts physical linear-polarization states: finite DOP lies in `[0, 1]`, AOP
is finite, and finite radiance is non-negative. `NaN` is reserved for masks. Positive infinity
is also accepted for radiance, as an explicit over-range sentinel; negative infinity is
invalid. After a non-zero analyzer transmission, an over-range value maps to ADC full scale.
Over-range pixels do not define the frame exposure, which is derived from the largest finite
intensity, so finite pixels remain measurable. Exactly zero analyzer transmission still
produces zero, rather than the indeterminate product `infinity × 0`.

Sensor dimensions must be divisible by the mosaic step, so that all four analyzer
orientations sample the same number of pixels; a 5-pixel dimension with a step of 2 would
give 3 rows for one orientation and 2 for another, which cannot be differenced.

`StokesCalculator` forms the Stokes parameters from the four analyzer sub-images:

```
S0 = ½ (I_0 + I_45 + I_90 + I_135)
S1 = I_0 − I_90
S2 = I_45 − I_135
```

and then

```
DOP_raw = sqrt(S1² + S2²) / S0
DOP     = clip(DOP_raw, 0, 1)
AOP     = ½ · atan2(S2, S1)
```

Read noise and channel saturation can put `DOP_raw` outside the physical Stokes cone. The
clip is the reported projection back onto that cone: it scales the linear-polarization
magnitude while preserving the orientation. It guarantees a bounded public DOP, but it does
**not** make the estimate unbiased — at low SNR, values can accumulate at 1. Where `S0` is
zero, or a contributing analyzer pixel is masked, DOP remains `NaN`.

### Normalized sensor response

The sensor pipeline is intentionally relative rather than radiometrically calibrated. In
order, for each pixel:

```
1.  transmission = ½ [1 + extinction_ratio · DOP · cos(2 (AOP − analyzer_angle))]
2.  intensity    = radiance × transmission
3.  noisy        = intensity × (1 + N(0, 1/multiplicative_noise_snr))
4.  full_scale   = auto_exposure_saturation_fraction × max(finite intensity in frame)
5.  counts       = clip(floor((2^bits − 1) / full_scale × noisy), 0, 2^bits − 1)
```

`analyzer_angle` is the nominal orientation of that pixel's wire grid plus its static defect,
drawn once from `[−polarizer_tolerance_radians, +polarizer_tolerance_radians]`.

The constructor names state the implemented quantity and its unit:

| Parameter | Implemented meaning |
|---|---|
| `sensor_pixel_pitch_micrometers` | Linear centre-to-centre pixel pitch, used by the lens projection |
| `extinction_ratio` | Analyzer quality on `[0, 1]`, appearing as a multiplier on DOP in step 1 |
| `polarizer_tolerance_radians` | Bound of one static uniform pixel defect, in `[−t, +t]` radians |
| `auto_exposure_saturation_fraction` | Fraction of the brightest finite pre-noise signal mapped to ADC full scale |
| `adc_resolution_bits` | Unsigned integer ADC bit depth, restricted to 1–24 for exact `float32` counts |
| `multiplicative_noise_snr` | Relative Gaussian SNR in `signal × (1 + N(0, 1/SNR))` |

Exposure is derived independently for each time slice. Positive infinity bypasses that
derivation and saturates explicitly; `NaN` remains a mask. Quantization uses `floor` followed
by clipping to `[0, 2^bits − 1]`. A frame that is entirely dark, or entirely masked, yields
zero counts rather than a division by zero.

There is no calibrated exposure, gain, full-well capacity, shot noise or additive read-noise
floor, so **counts from independently normalized frames are not comparable as absolute
radiances.** All sky fields and metadata are `float64`; ADC counts and reconstructed sensor
DOP and AOP are `float32`.

---

## 11. What is and is not modelled

**Modelled**

* Molecular anisotropy through a configurable depolarization ratio
  (`DEPOLARIZED_RAYLEIGH`).
* Four polarization singularities, as the zeros of an analytic quartic.
* Solar-elevation-dependent neutral-point offsets (`ASYMMETRIC`, `PAN`, `QUEEN`).
* Independently configurable anti-solar neutral-point distances along a covariant signed
  meridian (`ASYMMETRIC`).
* CIE standard sky radiance, all 15 sky types — **a PySkyLumos addition** coupled to each
  polarization law only at the returned-field level, not by full Stokes radiative transfer.
* Solar position from Astropy, using its built-in ephemeris by default or the JPL DE430
  kernel with `accuracy=True`. The JPL path requires the `jpl` extra and a downloaded or
  pre-cached kernel; an explicit `sun_position` bypasses it.
* Rigid sensor tilt, including ray rotation and geometric transport of linear AOP between the
  world and analyzer stereographic bases.
* The sensor chain: lens conjugation, micro-polarizer array, ADC quantization, noise.

**Not modelled**

* **Wavelength dependence.** All models are monochromatic; the offsets in §6 were fitted at a
  single wavelength, and `DEPOLARIZED_RAYLEIGH` holds its configurable molecular ratio
  constant rather than deriving it from wavelength or gas composition.
* **Aerosols, clouds, turbidity, ground albedo and multiple-scattering transport.** The
  quartic is a *phenomenological* description whose only inputs are the singularity
  positions, while `DEPOLARIZED_RAYLEIGH` implements only Wu's molecular phase-matrix term.
* **Circular polarization.** The Stokes `V` component is assumed zero throughout.
* **Optical polarization effects.** Lens-induced polarization, non-conformal lens effects on
  polarization orientation, oblique-incidence micro-polarizer response, and general
  Mueller-matrix optical transport are all absent.
* **Horizon depolarization.** The effect Berry §4 identifies is not corrected by any model;
  see §5. Depolarized Rayleigh likewise has no horizon-dependent term.
* **Any variation of the neutral-point offsets other than with solar elevation.** There is no
  dependence on season, humidity or location, even though Pan's fit was made on a single day
  at a single site.

`ASYMMETRIC` has additional evidential limits. Its default `"peak"` normalization is a
deterministic numerical scan, not a closed form. No published regression gives Arago or
fourth-point distance as a function of solar elevation, so those values remain options with
literature-motivated defaults; specifically, `fourth_offset="brewster"` rests on one
qualitative sentence in Horváth and Varjú rather than a regression. Tests establish analytic
self-consistency, correct symmetric limits, topology, covariance and literature anchors. They
do **not** show that the model fits measured sky imagery better than QuEEN. The local
single-capture quantification is an observational comparison, not the independent clear-sky
measurement campaign required for that claim.

---

## License and citation

PySkyLumos is released under the [MIT License](LICENSE).

If you use it in published work, please cite the package alongside the source papers listed
in [Sources](#sources) — Wu 2014 and Bodhaine 1999 for `DEPOLARIZED_RAYLEIGH`, or Berry 2004,
Pan 2023, OpenSky 2024, Wang 2016 and Horváth and Varjú for the quartic models — as appropriate
to the model you used.
