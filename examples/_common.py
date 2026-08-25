"""Shared utilities for deterministic, untilted model/sensor examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError as exc:  # pragma: no cover - exercised when the optional extra is absent
    raise SystemExit(
        "Matplotlib is required for these examples. Install it with "
        '`python -m pip install -e ".[examples]"`.'
    ) from exc

from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern

IMAGE_SIZE = 128
SENSOR_PIXEL_PITCH_MICROMETERS = 2.2
SUN_AZIMUTH_DEGREES = 137.0
SUN_ALTITUDE_DEGREES = 33.0
CIE_SKY_TYPE = 4

OBSERVATION_TIME = Time(["2026-11-27T13:00:00"])
OBSERVATION_LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
SUN_POSITION = SkyCoord(
    az=[SUN_AZIMUTH_DEGREES] * deg,
    alt=[SUN_ALTITUDE_DEGREES] * deg,
    frame=AltAz(obstime=OBSERVATION_TIME, location=OBSERVATION_LOCATION),
)

WIRE_GRID = {
    0: SlicingPattern(start_row=0, start_column=0, step=2),
    45: SlicingPattern(start_row=0, start_column=1, step=2),
    90: SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}


def _build_engine() -> Engine:
    """Build the deterministic upward-looking fisheye used by every example."""
    sensor_radius_micrometers = SENSOR_PIXEL_PITCH_MICROMETERS * (IMAGE_SIZE - 1) / 2
    focal_length_micrometers = sensor_radius_micrometers / (np.pi / 2)

    return Engine(
        sensor_pixel_pitch_micrometers=SENSOR_PIXEL_PITCH_MICROMETERS,
        lens_conjugation_type="equi_angle",
        number_pixels_vertical=IMAGE_SIZE,
        number_pixels_horizontal=IMAGE_SIZE,
        lens_focal_length_micrometers=focal_length_micrometers,
        polarizer_tolerance_radians=0.01,
        extinction_ratio=0.99,
        auto_exposure_saturation_fraction=0.9,
        adc_resolution_bits=12,
        multiplicative_noise_snr=50,
        wire_grid_orientations_slicing=WIRE_GRID,
        random_seed=0,
    )


def _simulate(model: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return untilted theoretical fields and sensor-frame measurements.

    The optical grid is world aligned in these comparison examples, so no
    integrated tilt parameters are supplied and theoretical AOP already shares
    the analyzer chart.
    """
    engine = _build_engine()
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0.0)

    values, names = engine.simulate_sky_polarization(
        sky_model=model,
        observation_location=OBSERVATION_LOCATION,
        times=OBSERVATION_TIME,
        cie_sky_type=CIE_SKY_TYPE,
        altitudes=altitudes,
        azimuths=azimuths,
        altitude_min_clip=0.0,
        sun_position=SUN_POSITION,
    )
    theory = dict(zip(names, values, strict=True))
    measurement = engine.simulate_measurement(
        degree_of_polarization=theory["degree of polarization"],
        angle_of_polarization=theory["angle of polarization"],
        radiance=theory["radiance"],
    )

    expected_theory_shape = (1, IMAGE_SIZE, IMAGE_SIZE)
    expected_measurement_shape = (1, IMAGE_SIZE // 2, IMAGE_SIZE // 2)
    if theory["degree of polarization"].shape != expected_theory_shape:
        raise RuntimeError(
            "Unexpected theoretical field shape: "
            f"{theory['degree of polarization'].shape}; expected {expected_theory_shape}."
        )
    if measurement["dop"].shape != expected_measurement_shape:
        raise RuntimeError(
            "Unexpected measurement shape: "
            f"{measurement['dop'].shape}; expected {expected_measurement_shape}."
        )
    if not np.isfinite(theory["degree of polarization"]).any():
        raise RuntimeError("The theoretical field contains no finite sky pixels.")
    if not np.isfinite(measurement["dop"]).any():
        raise RuntimeError("The sensor measurement contains no finite sky pixels.")

    return theory, measurement


def _plot_panel(
    figure,
    axis,
    values: np.ndarray,
    *,
    title: str,
    normalization: Normalize,
    colorbar_label: str,
) -> None:
    """Plot one masked sky field with a compact colorbar."""
    color_map = plt.get_cmap("rainbow").copy()
    color_map.set_bad("#e6e6e6")
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap=color_map,
        norm=normalization,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)


def _create_figure(model: str, theory: dict[str, np.ndarray], measurement: dict[str, np.ndarray]):
    """Create the five-panel theory-versus-measurement figure."""
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 6)
    axes = {
        "theory_dop": figure.add_subplot(grid[0, 0:2]),
        "theory_aop": figure.add_subplot(grid[0, 2:4]),
        "theory_radiance": figure.add_subplot(grid[0, 4:6]),
        "measurement_dop": figure.add_subplot(grid[1, 1:3]),
        "measurement_aop": figure.add_subplot(grid[1, 3:5]),
    }

    theory_dop = theory["degree of polarization"][0]
    theory_aop_degrees = np.rad2deg(theory["angle of polarization"][0])
    theory_radiance = theory["radiance"][0]
    measured_dop = measurement["dop"][0]
    measured_aop_degrees = np.rad2deg(measurement["aop"][0])

    dop_normalization = Normalize(vmin=0.0, vmax=1.0)
    aop_normalization = Normalize(vmin=-90.0, vmax=90.0)
    finite_radiance = theory_radiance[np.isfinite(theory_radiance)]
    radiance_normalization = Normalize(
        vmin=float(finite_radiance.min()),
        vmax=float(finite_radiance.max()),
    )

    _plot_panel(
        figure,
        axes["theory_dop"],
        theory_dop,
        title="Theoretical DOP",
        normalization=dop_normalization,
        colorbar_label="DOP",
    )
    _plot_panel(
        figure,
        axes["theory_aop"],
        theory_aop_degrees,
        title="Theoretical AOP",
        normalization=aop_normalization,
        colorbar_label="AOP (degrees)",
    )
    _plot_panel(
        figure,
        axes["theory_radiance"],
        theory_radiance,
        title=f"Theoretical relative radiance (CIE type {CIE_SKY_TYPE})",
        normalization=radiance_normalization,
        colorbar_label="Relative radiance",
    )
    _plot_panel(
        figure,
        axes["measurement_dop"],
        measured_dop,
        title="Sensor-measured DOP",
        normalization=dop_normalization,
        colorbar_label="DOP",
    )
    _plot_panel(
        figure,
        axes["measurement_aop"],
        measured_aop_degrees,
        title="Sensor-measured AOP",
        normalization=aop_normalization,
        colorbar_label="AOP (degrees)",
    )

    figure.suptitle(
        f"{model} sky polarization: theory and simulated measurement\n"
        f"Sun azimuth {SUN_AZIMUTH_DEGREES:.0f}°, altitude {SUN_ALTITUDE_DEGREES:.0f}°; "
        f"theory {IMAGE_SIZE}x{IMAGE_SIZE}, measurement {IMAGE_SIZE // 2}x{IMAGE_SIZE // 2}",
        fontsize=15,
    )
    figure.supxlabel(
        "Upward-looking view: East is at the top and North is at the right. "
        "Gray pixels lie outside the visible sky hemisphere."
    )
    return figure


def _parse_arguments(model: str) -> argparse.Namespace:
    """Parse the command-line options shared by all model wrappers."""
    default_output = (
        Path(__file__).resolve().parent / "output" / (f"{model.lower()}_theory_measurement.png")
    )
    parser = argparse.ArgumentParser(
        description=f"Plot theoretical and sensor-measured fields for the {model} sky model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"PNG output path (default: {default_output}).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive window.",
    )
    return parser.parse_args()


def main(model: str) -> None:
    """Run one model example, save its figure, and optionally display it."""
    arguments = _parse_arguments(model)
    theory, measurement = _simulate(model)
    figure = _create_figure(model, theory, measurement)

    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {model} example to {output_path}")
    print(
        f"Theoretical shape: {theory['degree of polarization'].shape}; "
        f"measurement shape: {measurement['dop'].shape}"
    )

    if not arguments.no_show:
        plt.show()
    plt.close(figure)
