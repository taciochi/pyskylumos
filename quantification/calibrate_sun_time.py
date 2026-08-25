"""Image-anchored UTC calibration for the repository-local sky capture.

The calibration deliberately uses only the raw sensor image and astronomical
ephemerides.  It does not inspect DOP, AOP, or any simulated sky model, so the
subsequent model comparison remains independent of this calibration step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time, TimeDelta
from numpy.typing import NDArray

if TYPE_CHECKING or __package__:
    from quantification.quantify_models import (
        CaptureConfig,
        load_config,
        load_tiff,
        sha256_file,
        verify_checksums,
    )
else:  # Direct ``python quantification/calibrate_sun_time.py`` invocation.
    from quantify_models import (  # type: ignore[import-not-found]
        CaptureConfig,
        load_config,
        load_tiff,
        sha256_file,
        verify_checksums,
    )

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SunTimeCandidate:
    """One projected UTC candidate and its image-aperture score."""

    stage: str
    time_utc: str
    unix_seconds: float
    sun_azimuth_deg: float
    sun_altitude_deg: float
    pixel_x: float
    pixel_y: float
    admissible: bool
    reason: str
    saturation_fraction: float
    mean_raw: float


def format_time_utc(value: Time) -> str:
    """Return a scalar Astropy time as second-resolution ISO UTC."""
    converted = cast(datetime, value.utc.to_datetime(timezone=UTC))
    return converted.strftime("%Y-%m-%dT%H:%M:%S")


def project_altaz_to_sensor(
    config: CaptureConfig, sun_azimuth_deg: float, sun_altitude_deg: float
) -> tuple[float, float]:
    """Project one world AltAz direction onto the configured sensor plane."""
    if config.lens_conjugation_type != "stereographic":
        raise ValueError(
            "Sun-time calibration currently requires camera.lens_conjugation_type = "
            "'stereographic'."
        )
    if not np.isfinite(sun_azimuth_deg) or not np.isfinite(sun_altitude_deg):
        raise ValueError("Sun azimuth and altitude must be finite.")

    sensor_azimuth = np.deg2rad((sun_azimuth_deg - config.yaw_deg + 180.0) % 360.0 - 180.0)
    zenith_angle = np.deg2rad(90.0 - sun_altitude_deg)
    radial_pixels = (
        2.0
        * config.focal_length_micrometers
        * np.tan(zenith_angle / 2.0)
        / config.pixel_pitch_micrometers
    )
    pixel_x = (config.image_width - 1) / 2.0 + radial_pixels * np.cos(sensor_azimuth)
    pixel_y = (config.image_height - 1) / 2.0 - radial_pixels * np.sin(sensor_azimuth)
    return float(pixel_x), float(pixel_y)


def _score_aperture(
    raw: FloatArray,
    pixel_x: float,
    pixel_y: float,
    aperture_radius_pixels: int,
    saturation_threshold: float,
) -> tuple[bool, str, float, float]:
    radius = aperture_radius_pixels
    x_min = int(np.floor(pixel_x)) - radius - 1
    x_max = int(np.floor(pixel_x)) + radius + 2
    y_min = int(np.floor(pixel_y)) - radius - 1
    y_max = int(np.floor(pixel_y)) + radius + 2
    if (
        not np.isfinite(pixel_x)
        or not np.isfinite(pixel_y)
        or x_min < 0
        or y_min < 0
        or x_max > raw.shape[1]
        or y_max > raw.shape[0]
    ):
        return False, "aperture_outside_sensor", 0.0, 0.0

    rows, columns = np.ogrid[y_min:y_max, x_min:x_max]
    disk = (columns - pixel_x) ** 2 + (rows - pixel_y) ** 2 <= radius**2
    values = raw[y_min:y_max, x_min:x_max][disk]
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False, "aperture_has_no_finite_pixels", 0.0, 0.0
    return (
        True,
        "",
        float(np.mean(values >= saturation_threshold)),
        float(np.mean(values)),
    )


def evaluate_candidate_times(
    config: CaptureConfig,
    raw: FloatArray,
    candidate_times: Time,
    *,
    stage: str,
    aperture_radius_pixels: int,
) -> list[SunTimeCandidate]:
    """Project and score every time in an Astropy time array."""
    location = EarthLocation(
        lat=config.latitude_deg * u.deg,
        lon=config.longitude_deg * u.deg,
        height=config.height_m * u.m,
    )
    sun = get_sun(candidate_times).transform_to(AltAz(obstime=candidate_times, location=location))
    azimuths = np.asarray(sun.az.deg, dtype=np.float64).reshape(-1)
    altitudes = np.asarray(sun.alt.deg, dtype=np.float64).reshape(-1)
    unix_seconds = np.asarray(candidate_times.unix, dtype=np.float64).reshape(-1)
    scalar_times = candidate_times.reshape((candidate_times.size,))

    candidates: list[SunTimeCandidate] = []
    for index, (azimuth, altitude, unix_second) in enumerate(
        zip(azimuths, altitudes, unix_seconds, strict=True)
    ):
        pixel_x, pixel_y = project_altaz_to_sensor(config, float(azimuth), float(altitude))
        admissible, reason, saturation_fraction, mean_raw = _score_aperture(
            raw,
            pixel_x,
            pixel_y,
            aperture_radius_pixels,
            config.saturation_threshold,
        )
        candidates.append(
            SunTimeCandidate(
                stage=stage,
                time_utc=format_time_utc(scalar_times[index]),
                unix_seconds=float(unix_second),
                sun_azimuth_deg=float(azimuth),
                sun_altitude_deg=float(altitude),
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                admissible=admissible,
                reason=reason,
                saturation_fraction=saturation_fraction,
                mean_raw=mean_raw,
            )
        )
    return candidates


def choose_best_candidate(
    candidates: list[SunTimeCandidate], minimum_saturation_fraction: float
) -> SunTimeCandidate:
    """Select the lexicographic image winner, preferring earlier UTC on an exact tie."""
    admissible = [candidate for candidate in candidates if candidate.admissible]
    if not admissible:
        raise ValueError("No candidate Sun aperture lies fully inside the configured sensor.")
    winner = min(
        admissible,
        key=lambda candidate: (
            -candidate.saturation_fraction,
            -candidate.mean_raw,
            candidate.unix_seconds,
        ),
    )
    if winner.saturation_fraction < minimum_saturation_fraction:
        raise ValueError(
            "No credible visible Sun was found: best saturation fraction "
            f"{winner.saturation_fraction:.6f} is below the required "
            f"{minimum_saturation_fraction:.6f}."
        )
    return winner


def _candidate_times(center: Time, half_window_seconds: int, step_seconds: int) -> Time:
    offsets = np.arange(
        -half_window_seconds,
        half_window_seconds + 1,
        step_seconds,
        dtype=np.float64,
    )
    if not np.any(offsets == 0.0):
        offsets = np.sort(np.append(offsets, 0.0))
    return center + TimeDelta(offsets, format="sec")


def replace_capture_time_text(document: str, time_utc: str) -> str:
    """Replace only ``[capture].time_utc`` while retaining TOML formatting."""
    Time(time_utc, scale="utc")
    section = re.search(r"(?ms)^\[capture\][ \t]*\r?\n.*?(?=^\[|\Z)", document)
    if section is None:
        raise ValueError("Configuration does not contain a [capture] table.")
    replacement_pattern = re.compile(r'(?m)^([ \t]*time_utc[ \t]*=[ \t]*")[^"\r\n]*("[^\r\n]*)$')
    updated_section, count = replacement_pattern.subn(
        lambda match: f"{match.group(1)}{time_utc}{match.group(2)}",
        section.group(0),
    )
    if count != 1:
        raise ValueError("[capture] must contain exactly one quoted time_utc assignment.")
    return document[: section.start()] + updated_section + document[section.end() :]


def write_capture_time(config_path: Path, time_utc: str) -> None:
    """Atomically update one capture time without reserializing the TOML."""
    original = config_path.read_text(encoding="utf-8")
    updated = replace_capture_time_text(original, time_utc)
    if updated == original:
        return

    mode = config_path.stat().st_mode
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(updated)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, config_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _candidate_record(candidate: SunTimeCandidate) -> dict[str, Any]:
    return asdict(candidate)


def _angular_separation_deg(first: SunTimeCandidate, second: SunTimeCandidate) -> float:
    first_azimuth = np.deg2rad(first.sun_azimuth_deg)
    second_azimuth = np.deg2rad(second.sun_azimuth_deg)
    first_altitude = np.deg2rad(first.sun_altitude_deg)
    second_altitude = np.deg2rad(second.sun_altitude_deg)
    cosine = np.sin(first_altitude) * np.sin(second_altitude) + np.cos(first_altitude) * np.cos(
        second_altitude
    ) * np.cos(first_azimuth - second_azimuth)
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _write_search_csv(path: Path, candidates: list[SunTimeCandidate]) -> None:
    fieldnames = tuple(asdict(candidates[0]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def _plot_calibration(
    output_path: Path,
    raw: FloatArray,
    coarse: list[SunTimeCandidate],
    fine: list[SunTimeCandidate],
    baseline: SunTimeCandidate,
    selected: SunTimeCandidate,
    aperture_radius_pixels: int,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError as error:  # pragma: no cover - dependency contract
        raise RuntimeError(
            'Matplotlib is required; install with `python -m pip install -e ".[quantification]"`.'
        ) from error

    figure, (image_axis, score_axis) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    image_axis.imshow(raw, cmap="gray", vmin=0.0, vmax=float(np.percentile(raw, 99.5)))
    track = [candidate for candidate in coarse if candidate.admissible]
    image_axis.plot(
        [candidate.pixel_x for candidate in track],
        [candidate.pixel_y for candidate in track],
        color="tab:cyan",
        linewidth=1.0,
        label="ephemeris search track",
    )
    image_axis.scatter(
        [baseline.pixel_x],
        [baseline.pixel_y],
        color="tab:orange",
        marker="x",
        s=80,
        label=f"manifest {baseline.time_utc[11:]}",
    )
    image_axis.scatter(
        [selected.pixel_x],
        [selected.pixel_y],
        color="tab:red",
        marker="+",
        s=100,
        label=f"selected {selected.time_utc[11:]}",
    )
    image_axis.add_patch(
        Circle(
            (selected.pixel_x, selected.pixel_y),
            aperture_radius_pixels,
            fill=False,
            edgecolor="tab:red",
            linewidth=1.5,
        )
    )
    image_axis.set_title("Raw capture and projected Sun track")
    image_axis.set_xlim(0, raw.shape[1])
    image_axis.set_ylim(raw.shape[0], 0)
    image_axis.legend(loc="upper right")

    coarse_times = [
        datetime.strptime(candidate.time_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
        for candidate in coarse
    ]
    score_axis.plot(
        coarse_times,
        [candidate.saturation_fraction if candidate.admissible else np.nan for candidate in coarse],
        color="tab:blue",
        label="one-minute search",
    )
    if fine:
        fine_times = [
            datetime.strptime(candidate.time_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
            for candidate in fine
        ]
        score_axis.plot(
            fine_times,
            [candidate.saturation_fraction for candidate in fine],
            color="tab:red",
            alpha=0.8,
            label="one-second diagnostic",
        )
    score_axis.axvline(
        datetime.strptime(selected.time_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC),
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    score_axis.set_title("Sun-aperture saturation score")
    score_axis.set_xlabel("Candidate UTC")
    score_axis.set_ylabel("Saturated-pixel fraction")
    score_axis.grid(alpha=0.25)
    score_axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def run_calibration(
    config: CaptureConfig,
    output_dir: Path,
    *,
    search_half_window_seconds: int,
    coarse_step_seconds: int,
    fine_half_window_seconds: int,
    aperture_radius_pixels: int,
    minimum_saturation_fraction: float,
    run_fine_refinement: bool,
    write_config: bool,
    show: bool,
) -> dict[str, Any]:
    """Execute the checksum-verified image-anchored UTC search."""
    checksums = verify_checksums(config)
    raw = load_tiff(
        config.input_dir / config.raw_file,
        (config.image_height, config.image_width),
    )
    center = Time(config.time_utc, scale="utc")
    coarse_times = _candidate_times(
        center,
        search_half_window_seconds,
        coarse_step_seconds,
    )
    coarse = evaluate_candidate_times(
        config,
        raw,
        coarse_times,
        stage="coarse",
        aperture_radius_pixels=aperture_radius_pixels,
    )
    selected = choose_best_candidate(coarse, minimum_saturation_fraction)
    baseline = min(coarse, key=lambda candidate: abs(candidate.unix_seconds - float(center.unix)))

    fine: list[SunTimeCandidate] = []
    fine_winner: SunTimeCandidate | None = None
    if run_fine_refinement:
        fine_times = _candidate_times(
            Time(selected.time_utc, scale="utc"),
            fine_half_window_seconds,
            1,
        )
        fine = evaluate_candidate_times(
            config,
            raw,
            fine_times,
            stage="fine",
            aperture_radius_pixels=aperture_radius_pixels,
        )
        fine_winner = choose_best_candidate(fine, minimum_saturation_fraction)

    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    all_candidates = [*coarse, *fine]
    _write_search_csv(resolved_output / "sun_time_search.csv", all_candidates)
    _plot_calibration(
        resolved_output / "sun_time_calibration.png",
        raw,
        coarse,
        fine,
        baseline,
        selected,
        aperture_radius_pixels,
        show,
    )

    config_hash_before = sha256_file(config.config_path)
    if write_config:
        write_capture_time(config.config_path, selected.time_utc)
    config_hash_after = sha256_file(config.config_path)

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "image_anchored_ephemeris_time_search",
        "config_path": str(config.config_path),
        "config_sha256_before": config_hash_before,
        "config_sha256_after": config_hash_after,
        "config_updated": write_config and config_hash_before != config_hash_after,
        "input_checksums": checksums,
        "search": {
            "half_window_seconds": search_half_window_seconds,
            "coarse_step_seconds": coarse_step_seconds,
            "fine_half_window_seconds": fine_half_window_seconds,
            "fine_step_seconds": 1,
            "fine_refinement_is_diagnostic_only": True,
            "aperture_radius_pixels": aperture_radius_pixels,
            "saturation_threshold": config.saturation_threshold,
            "minimum_saturation_fraction": minimum_saturation_fraction,
            "ranking": [
                "maximum saturation_fraction",
                "maximum mean_raw",
                "earliest UTC",
            ],
        },
        "baseline": _candidate_record(baseline),
        "selected_minute": _candidate_record(selected),
        "fine_diagnostic_winner": (
            _candidate_record(fine_winner) if fine_winner is not None else None
        ),
        "displacement": {
            "time_seconds": selected.unix_seconds - baseline.unix_seconds,
            "sun_angular_degrees": _angular_separation_deg(baseline, selected),
            "sensor_pixels": float(
                np.hypot(selected.pixel_x - baseline.pixel_x, selected.pixel_y - baseline.pixel_y)
            ),
        },
        "manifest_time_written_utc": selected.time_utc if write_config else None,
        "assumptions": {
            "visible_starburst_is_direct_sun": True,
            "calibration_uses_polarization_measurements": False,
            "calibration_estimates_camera_pose_or_lens_distortion": False,
            "minute_precision_is_authoritative": True,
        },
    }
    with (resolved_output / "sun_time_calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive finite number") from error
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return parsed


def _fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a finite fraction in [0, 1]") from error
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a finite fraction in [0, 1]")
    return parsed


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the image-anchored calibration command line."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calibrate capture UTC by matching the ephemeris Sun to the raw starburst."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "capture.toml",
        help="Capture TOML manifest (default: quantification/capture.toml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output",
        help="Directory for calibration CSV, JSON and PNG artifacts.",
    )
    parser.add_argument(
        "--search-half-window-hours",
        type=_positive_float,
        default=3.0,
        help="Hours searched on each side of the manifest time (default: 3).",
    )
    parser.add_argument(
        "--coarse-step-seconds",
        type=_positive_int,
        default=60,
        help="Coarse search interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--fine-half-window-seconds",
        type=_positive_int,
        default=60,
        help="Diagnostic one-second search radius around the winning minute (default: 60).",
    )
    parser.add_argument(
        "--aperture-radius-pixels",
        type=_positive_int,
        default=45,
        help="Raw-image scoring aperture radius (default: 45 pixels).",
    )
    parser.add_argument(
        "--minimum-saturation-fraction",
        type=_fraction,
        default=0.1,
        help="Minimum credible saturated fraction in the winning aperture (default: 0.1).",
    )
    parser.add_argument(
        "--skip-fine-refinement",
        action="store_true",
        help="Skip the diagnostic one-second search around the winning minute.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Atomically replace [capture].time_utc with the selected minute.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the calibration figure interactively after saving it.",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run the calibration CLI and return a process status."""
    options = parse_arguments(arguments)
    half_window_seconds = round(options.search_half_window_hours * 3600.0)
    config = load_config(options.config)
    report = run_calibration(
        config,
        options.output_dir,
        search_half_window_seconds=half_window_seconds,
        coarse_step_seconds=options.coarse_step_seconds,
        fine_half_window_seconds=options.fine_half_window_seconds,
        aperture_radius_pixels=options.aperture_radius_pixels,
        minimum_saturation_fraction=options.minimum_saturation_fraction,
        run_fine_refinement=not options.skip_fine_refinement,
        write_config=options.write_config,
        show=options.show,
    )
    selected = report["selected_minute"]
    print(
        f"Selected capture UTC: {selected['time_utc']} "
        f"(Sun pixel x={selected['pixel_x']:.2f}, y={selected['pixel_y']:.2f})"
    )
    fine = report["fine_diagnostic_winner"]
    if fine is not None:
        print(f"One-second diagnostic winner: {fine['time_utc']}")
    if options.write_config:
        print(f"Updated {config.config_path}")
    else:
        print("Dry run: capture manifest was not changed.")
    print(f"Wrote calibration artifacts to {Path(options.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
