"""Compare an untilted camera with one under a general three-dimensional pose.

This is the only example that points the camera away from vertical. It uses
``sensor_to_world_rotation_matrix``, which accepts an arbitrary sensor-to-world
rotation, rather than the two-parameter tilt arguments that can only describe a
rotation about a horizontal axis.

The third panel is the point of the example: the AOP difference between the two
poses is **not** a single angle added to the whole image. Each ray carries its
own stereographic tangent basis, so the correction varies from pixel to pixel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _common import (
    CIE_SKY_TYPE,
    IMAGE_SIZE,
    OBSERVATION_LOCATION,
    OBSERVATION_TIME,
    SUN_ALTITUDE_DEGREES,
    SUN_AZIMUTH_DEGREES,
    SUN_POSITION,
    _build_engine,
    _plot_panel,
    plt,
)
from matplotlib.colors import Normalize

SKY_MODEL = "QUEEN"
YAW_RADIANS = 0.5
TIP_RADIANS = 0.35


def _rotation_about_up(angle: float) -> np.ndarray:
    """Return a right-handed rotation about the Up axis, in North-East-Up."""
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotation_about_north(angle: float) -> np.ndarray:
    """Return a right-handed rotation about the North axis, in North-East-Up."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )


def _build_pose() -> np.ndarray:
    """Compose the sensor-to-world rotation demonstrated by this example.

    The camera is first twisted about its own optical axis and then tipped away
    from vertical. The twist is the part the tilt arguments cannot express, so
    this pose is genuinely three-dimensional rather than a re-spelling of a
    horizontal-axis tilt. Matrix order matters: these two factors do not
    commute.
    """
    return _rotation_about_north(TIP_RADIANS) @ _rotation_about_up(YAW_RADIANS)


def axial_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the shortest signed difference between two axial angle fields.

    AOP is axial: values separated by ``pi`` describe the same orientation, so a
    plain subtraction is wrong wherever the two fields straddle the wrap.
    """
    delta = first - second
    return 0.5 * np.arctan2(np.sin(2.0 * delta), np.cos(2.0 * delta))


def _simulate() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Simulate the same sky from an untilted pose and a rotated one.

    Returns:
        The untilted theoretical fields, the rotated theoretical fields, and the
        rotated sensor measurement. Both simulations share one engine, one
        sensor-local direction grid, and one horizon clip, so every difference
        between them comes from the pose alone.
    """
    engine = _build_engine()
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0.0)

    common = {
        "sky_model": SKY_MODEL,
        "observation_location": OBSERVATION_LOCATION,
        "times": OBSERVATION_TIME,
        "cie_sky_type": CIE_SKY_TYPE,
        "altitudes": altitudes,
        "azimuths": azimuths,
        "altitude_min_clip": 0.0,
        "sun_position": SUN_POSITION,
    }

    untilted_values, names = engine.simulate_sky_polarization(**common)
    rotated_values, _ = engine.simulate_sky_polarization(
        **common,
        sensor_to_world_rotation_matrix=_build_pose(),
    )

    untilted = dict(zip(names, untilted_values, strict=True))
    rotated = dict(zip(names, rotated_values, strict=True))
    measurement = engine.simulate_measurement(
        degree_of_polarization=rotated["degree of polarization"],
        angle_of_polarization=rotated["angle of polarization"],
        radiance=rotated["radiance"],
    )
    return untilted, rotated, measurement


def _create_figure(untilted: dict[str, np.ndarray], rotated: dict[str, np.ndarray]):
    """Create the three-panel untilted/rotated/difference AOP figure."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 5.0), constrained_layout=True)

    untilted_aop = untilted["angle of polarization"][0]
    rotated_aop = rotated["angle of polarization"][0]
    difference_degrees = np.rad2deg(axial_difference(rotated_aop, untilted_aop))

    aop_normalization = Normalize(vmin=-90.0, vmax=90.0)
    _plot_panel(
        figure,
        axes[0],
        np.rad2deg(untilted_aop),
        title="Untilted AOP",
        normalization=aop_normalization,
        colorbar_label="AOP (degrees)",
    )
    _plot_panel(
        figure,
        axes[1],
        np.rad2deg(rotated_aop),
        title=(
            f"Rotated AOP (yaw {np.rad2deg(YAW_RADIANS):.0f}°, tip {np.rad2deg(TIP_RADIANS):.0f}°)"
        ),
        normalization=aop_normalization,
        colorbar_label="AOP (degrees)",
    )
    _plot_panel(
        figure,
        axes[2],
        difference_degrees,
        title="Axial AOP difference: varies per pixel",
        normalization=Normalize(vmin=-90.0, vmax=90.0),
        colorbar_label="Difference (degrees)",
    )

    figure.suptitle(
        f"{SKY_MODEL} under a general camera pose: sensor_to_world_rotation_matrix\n"
        f"Sun azimuth {SUN_AZIMUTH_DEGREES:.0f}°, altitude {SUN_ALTITUDE_DEGREES:.0f}°; "
        f"{IMAGE_SIZE}x{IMAGE_SIZE} theoretical fields",
        fontsize=15,
    )
    figure.supxlabel(
        "The rotated pose samples a different part of the sky and re-expresses AOP in the "
        "rotated analyzer frame. The right-hand panel is not constant, which is what "
        "distinguishes a real pose from a uniform azimuth_rotation_angle offset. "
        "Gray pixels lie outside the visible sky hemisphere."
    )
    return figure


def _parse_arguments() -> argparse.Namespace:
    """Parse the command-line options, matching the other example scripts."""
    default_output = Path(__file__).resolve().parent / "output" / "rotated_pose.png"
    parser = argparse.ArgumentParser(
        description="Compare untilted and generally rotated camera poses for one sky model."
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


def main() -> None:
    """Run the pose comparison, save its figure, and optionally display it."""
    arguments = _parse_arguments()
    untilted, rotated, measurement = _simulate()
    figure = _create_figure(untilted, rotated)

    output_path = arguments.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")

    difference = np.rad2deg(
        axial_difference(
            rotated["angle of polarization"][0],
            untilted["angle of polarization"][0],
        )
    )
    finite_difference = difference[np.isfinite(difference)]
    untilted_visible = int(np.isfinite(untilted["degree of polarization"]).sum())
    rotated_visible = int(np.isfinite(rotated["degree of polarization"]).sum())

    print(f"Saved rotated-pose example to {output_path}")
    print(
        f"Theoretical shape: {rotated['degree of polarization'].shape}; "
        f"measurement shape: {measurement['dop'].shape}"
    )
    print(f"Visible sky pixels: untilted {untilted_visible}, rotated {rotated_visible}")
    print(
        "Axial AOP difference (degrees): "
        f"min {finite_difference.min():.2f}, "
        f"max {finite_difference.max():.2f}, "
        f"standard deviation {finite_difference.std():.2f}"
    )

    if not arguments.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
