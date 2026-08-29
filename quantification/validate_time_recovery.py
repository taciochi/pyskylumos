"""Closed-loop characterization of the image-anchored UTC recovery method.

``calibrate_sun_time`` recovers a capture's acquisition time from the raw sensor
frame alone.  On the repository capture it corrected a one-hour error, but a
single observation cannot establish an operating envelope.  This script renders
frames whose acquisition time is known exactly, runs the shipped recovery on
them, and reports the error against that ground truth.

The recovery search itself is imported from :mod:`calibrate_sun_time` rather
than reimplemented, so the characterized procedure is provably the shipped one.
The renderer perturbs the frame away from an exact inverse of the recovery
model: analyzer noise, exposure, camera yaw error and occlusion are all swept
independently.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from numpy.typing import NDArray

from pyskylumos import __version__
from pyskylumos.sensor import SensorChip

if TYPE_CHECKING or __package__:
    from quantification.calibrate_sun_time import (
        SunTimeCandidate,
        _candidate_times,
        choose_best_candidate,
        containment_statistics,
        evaluate_candidate_times,
        format_time_utc,
        project_altaz_to_sensor,
    )
    from quantification.quantify_models import (
        POSITIONS,
        CaptureConfig,
        _build_engine,
        analyzer_response,
        load_config,
        load_tiff,
    )
else:  # Direct ``python quantification/validate_time_recovery.py`` invocation.
    from calibrate_sun_time import (  # type: ignore[import-not-found]
        SunTimeCandidate,
        _candidate_times,
        choose_best_candidate,
        containment_statistics,
        evaluate_candidate_times,
        format_time_utc,
        project_altaz_to_sensor,
    )
    from quantify_models import (  # type: ignore[import-not-found]
        POSITIONS,
        CaptureConfig,
        _build_engine,
        analyzer_response,
        load_config,
        load_tiff,
    )

type FloatArray = NDArray[np.float64]

DEFAULT_HALF_WINDOW_SECONDS = 10800
DEFAULT_COARSE_STEP_SECONDS = 60
DEFAULT_APERTURE_RADIUS_PIXELS = 45
DEFAULT_MINIMUM_SATURATION_FRACTION = 0.1
DEFAULT_EXPOSURE_FRACTION = 0.9
DEFAULT_NOISE_SNR = 50.0
DEFAULT_MODEL = "RAYLEIGH"
DEFAULT_CIE_SKY_TYPE = 8
DEFAULT_SEARCH_DISPLACEMENT_SECONDS = 3600.0
PLATEAU_SCORE_TOLERANCE = 1.0e-9
SILENT_FAILURE_SECONDS = 600.0
DEFAULT_CONTAINMENT_THRESHOLD = 0.90


@dataclass(frozen=True)
class RenderSettings:
    """Everything that distinguishes a rendered frame from the base capture."""

    exposure_fraction: float = DEFAULT_EXPOSURE_FRACTION
    noise_snr: float = DEFAULT_NOISE_SNR
    yaw_error_deg: float = 0.0
    occlusion_fraction: float = 0.0
    occlude_sun: bool = False
    seed: int = 0


@dataclass(frozen=True)
class RecoveryTrial:
    """One closed-loop trial: render at a known time, recover, compare."""

    sweep: str
    setting: str
    value: float
    repeat: int
    seed: int
    true_time_utc: str
    true_sun_azimuth_deg: float
    true_sun_altitude_deg: float
    search_centre_time_utc: str
    aperture_radius_pixels: int
    exposure_fraction: float
    noise_snr: float
    yaw_error_deg: float
    occlusion_fraction: float
    occlude_sun: bool
    global_saturation_fraction: float
    aperture_saturation_fraction: float
    succeeded: bool
    failure_reason: str
    recovered_time_utc: str
    time_error_seconds: float
    absolute_time_error_seconds: float
    sun_angular_error_deg: float
    pixel_error: float
    plateau_width_seconds: float
    admissible_candidates: int
    winner_aperture_saturation: float
    winner_annulus_saturation: float
    winner_containment: float
    guard_passed: bool


def _sun_altaz(location: EarthLocation, times: Time) -> tuple[FloatArray, FloatArray]:
    """Return the astronomical Sun azimuth and altitude in degrees."""
    sun = get_sun(times).transform_to(AltAz(obstime=times, location=location))
    return (
        np.asarray(sun.az.deg, dtype=np.float64).reshape(-1),
        np.asarray(sun.alt.deg, dtype=np.float64).reshape(-1),
    )


def _location(config: CaptureConfig) -> EarthLocation:
    return EarthLocation(
        lat=config.latitude_deg * u.deg,
        lon=config.longitude_deg * u.deg,
        height=config.height_m * u.m,
    )


def angular_separation_deg(
    first_azimuth_deg: float,
    first_altitude_deg: float,
    second_azimuth_deg: float,
    second_altitude_deg: float,
) -> float:
    """Return the great-circle separation between two horizontal directions."""
    first_azimuth = np.deg2rad(first_azimuth_deg)
    first_altitude = np.deg2rad(first_altitude_deg)
    second_azimuth = np.deg2rad(second_azimuth_deg)
    second_altitude = np.deg2rad(second_altitude_deg)
    cosine = np.sin(first_altitude) * np.sin(second_altitude) + np.cos(first_altitude) * np.cos(
        second_altitude
    ) * np.cos(first_azimuth - second_azimuth)
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _occlusion_transmission(
    config: CaptureConfig,
    settings: RenderSettings,
    sun_pixel_x: float,
    sun_pixel_y: float,
) -> FloatArray | None:
    """Return a multiplicative cloud-transmission field, or ``None`` if clear.

    Clouds are opaque discs of random radius placed by a seeded generator until
    the requested fraction of the frame is covered.  ``occlude_sun`` forces one
    disc over the projected solar position, which is the case that must make the
    recovery fail rather than return a confident wrong answer.
    """
    if settings.occlusion_fraction <= 0.0 and not settings.occlude_sun:
        return None

    generator = np.random.default_rng(settings.seed + 9_973)
    transmission = np.ones((config.image_height, config.image_width), dtype=np.float64)
    rows = np.arange(config.image_height, dtype=np.float64)[:, None]
    columns = np.arange(config.image_width, dtype=np.float64)[None, :]

    if settings.occlude_sun:
        radius = 2.0 * DEFAULT_APERTURE_RADIUS_PIXELS
        disc = (columns - sun_pixel_x) ** 2 + (rows - sun_pixel_y) ** 2 <= radius**2
        transmission[disc] = 0.0

    target = float(settings.occlusion_fraction)
    guard = 0
    while transmission.mean() > 1.0 - target and guard < 4096:
        guard += 1
        centre_x = generator.uniform(0.0, config.image_width)
        centre_y = generator.uniform(0.0, config.image_height)
        radius = generator.uniform(60.0, 260.0)
        disc = (columns - centre_x) ** 2 + (rows - centre_y) ** 2 <= radius**2
        transmission[disc] = 0.0
    return transmission


def render_raw_counts(
    config: CaptureConfig,
    time_utc: Time,
    settings: RenderSettings,
    *,
    model: str = DEFAULT_MODEL,
    cie_sky_type: int = DEFAULT_CIE_SKY_TYPE,
) -> tuple[FloatArray, float, float]:
    """Render one division-of-focal-plane frame in ADC counts at a known time.

    Args:
        config: Capture manifest describing the camera and site.
        time_utc: The frame's true acquisition time.
        settings: Rendering perturbations applied to this frame.
        model: Sky model supplying degree and angle of polarization.
        cie_sky_type: CIE relative-radiance sky type.

    Returns:
        The ADC-count frame, the true Sun azimuth and the true Sun altitude.
    """
    location = _location(config)
    times = Time([format_time_utc(time_utc)], scale="utc")
    azimuths_deg, altitudes_deg = _sun_altaz(location, times)
    sun_azimuth = float(azimuths_deg[0])
    sun_altitude = float(altitudes_deg[0])

    # The frame is rendered through the camera's true yaw.  ``yaw_error_deg``
    # is the error the recovery will later be given, so it is subtracted here.
    render_config = replace(config, yaw_deg=config.yaw_deg - settings.yaw_error_deg)
    engine = _build_engine(render_config)
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=None)
    world_azimuths = np.asarray(
        engine.rotate_sensor(azimuths, render_config.yaw_deg), dtype=np.float64
    )
    simulation_altitudes = np.maximum(altitudes, render_config.altitude_min_deg)

    intensity = np.full((config.image_height, config.image_width), np.nan, dtype=np.float64)
    for row, column in POSITIONS:
        values, names = engine.simulate_sky_polarization(
            sky_model=model,
            observation_location=location,
            times=times,
            cie_sky_type=cie_sky_type,
            altitudes=simulation_altitudes[row::2, column::2],
            azimuths=world_azimuths[row::2, column::2],
            altitude_min_clip=None,
            azimuth_rotation_angle=render_config.yaw_deg,
            accuracy=False,
        )
        fields = dict(zip(names, values, strict=True))
        degree = np.asarray(fields["degree of polarization"][0], dtype=np.float64)
        angle = np.asarray(fields["angle of polarization"][0], dtype=np.float64)
        radiance = np.asarray(fields["radiance"][0], dtype=np.float64)
        intensity[row::2, column::2] = analyzer_response(
            config.dop_scale * degree,
            angle,
            radiance,
            config.analyzer_tile[row][column],
            config.extinction_ratio,
        )

    sun_pixel_x, sun_pixel_y = project_altaz_to_sensor(config, sun_azimuth, sun_altitude)
    transmission = _occlusion_transmission(config, settings, sun_pixel_x, sun_pixel_y)
    if transmission is not None:
        intensity = np.asarray(intensity * transmission, dtype=np.float64)

    chip = SensorChip(
        auto_exposure_saturation_fraction=settings.exposure_fraction,
        adc_resolution_bits=12,
        multiplicative_noise_snr=settings.noise_snr,
        random_seed=settings.seed,
    )
    counts = np.asarray(chip.get_bits_intensity(intensity[None, :, :])[0], dtype=np.float64)
    return counts, sun_azimuth, sun_altitude


def _plateau_width_seconds(
    candidates: list[SunTimeCandidate], winner: SunTimeCandidate, step_seconds: int
) -> float:
    """Return the span of candidates tying the winner's saturation fraction.

    The recovery's time resolution is set by the saturated blob's angular size
    and the solar rate, not by the search step: every candidate whose aperture
    covers the same saturated pixels scores identically.
    """
    tied = [
        candidate
        for candidate in candidates
        if candidate.admissible
        and abs(candidate.saturation_fraction - winner.saturation_fraction)
        <= PLATEAU_SCORE_TOLERANCE
    ]
    if len(tied) <= 1:
        return float(step_seconds)
    span = max(candidate.unix_seconds for candidate in tied) - min(
        candidate.unix_seconds for candidate in tied
    )
    return float(span)


def run_trial(
    config: CaptureConfig,
    true_time: Time,
    settings: RenderSettings,
    *,
    sweep: str,
    setting: str,
    value: float,
    repeat: int,
    aperture_radius_pixels: int = DEFAULT_APERTURE_RADIUS_PIXELS,
    half_window_seconds: int = DEFAULT_HALF_WINDOW_SECONDS,
    step_seconds: int = DEFAULT_COARSE_STEP_SECONDS,
    minimum_saturation_fraction: float = DEFAULT_MINIMUM_SATURATION_FRACTION,
    search_displacement_seconds: float = DEFAULT_SEARCH_DISPLACEMENT_SECONDS,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> RecoveryTrial:
    """Render a frame at a known time, recover that time, and score the result."""
    counts, true_azimuth, true_altitude = render_raw_counts(config, true_time, settings)
    global_saturation = float(np.mean(counts >= config.saturation_threshold))

    sun_pixel_x, sun_pixel_y = project_altaz_to_sensor(config, true_azimuth, true_altitude)
    rows, columns = np.ogrid[: config.image_height, : config.image_width]
    aperture = (columns - sun_pixel_x) ** 2 + (rows - sun_pixel_y) ** 2 <= aperture_radius_pixels**2
    aperture_saturation = float(np.mean(counts[aperture] >= config.saturation_threshold))

    # The search starts from a deliberately wrong centre, mirroring the local-time
    # metadata error the method was written to detect.
    centre = true_time + search_displacement_seconds * u.s
    candidate_times = _candidate_times(centre, half_window_seconds, step_seconds)
    candidates = evaluate_candidate_times(
        config,
        counts,
        candidate_times,
        stage="coarse",
        aperture_radius_pixels=aperture_radius_pixels,
    )
    admissible = sum(1 for candidate in candidates if candidate.admissible)

    base: dict[str, Any] = {
        "sweep": sweep,
        "setting": setting,
        "value": value,
        "repeat": repeat,
        "seed": settings.seed,
        "true_time_utc": format_time_utc(true_time),
        "true_sun_azimuth_deg": true_azimuth,
        "true_sun_altitude_deg": true_altitude,
        "search_centre_time_utc": format_time_utc(centre),
        "aperture_radius_pixels": aperture_radius_pixels,
        "exposure_fraction": settings.exposure_fraction,
        "noise_snr": settings.noise_snr,
        "yaw_error_deg": settings.yaw_error_deg,
        "occlusion_fraction": settings.occlusion_fraction,
        "occlude_sun": settings.occlude_sun,
        "global_saturation_fraction": global_saturation,
        "aperture_saturation_fraction": aperture_saturation,
        "admissible_candidates": admissible,
    }

    try:
        winner = choose_best_candidate(candidates, minimum_saturation_fraction)
    except ValueError as error:
        return RecoveryTrial(
            **base,
            succeeded=False,
            failure_reason=str(error).split(":")[0],
            recovered_time_utc="",
            time_error_seconds=float("nan"),
            absolute_time_error_seconds=float("nan"),
            sun_angular_error_deg=float("nan"),
            pixel_error=float("nan"),
            plateau_width_seconds=float("nan"),
            winner_aperture_saturation=float("nan"),
            winner_annulus_saturation=float("nan"),
            winner_containment=float("nan"),
            guard_passed=False,
        )

    winner_aperture, winner_annulus, containment = containment_statistics(
        counts,
        winner.pixel_x,
        winner.pixel_y,
        aperture_radius_pixels,
        config.saturation_threshold,
    )
    time_error = float(winner.unix_seconds - float(np.asarray(true_time.unix).reshape(-1)[0]))
    return RecoveryTrial(
        **base,
        succeeded=True,
        failure_reason="",
        recovered_time_utc=winner.time_utc,
        time_error_seconds=time_error,
        absolute_time_error_seconds=abs(time_error),
        sun_angular_error_deg=angular_separation_deg(
            winner.sun_azimuth_deg, winner.sun_altitude_deg, true_azimuth, true_altitude
        ),
        pixel_error=float(np.hypot(winner.pixel_x - sun_pixel_x, winner.pixel_y - sun_pixel_y)),
        plateau_width_seconds=_plateau_width_seconds(candidates, winner, step_seconds),
        winner_aperture_saturation=winner_aperture,
        winner_annulus_saturation=winner_annulus,
        winner_containment=containment,
        guard_passed=containment >= containment_threshold,
    )


def times_at_solar_elevations(
    config: CaptureConfig, elevations_deg: list[float]
) -> dict[float, Time]:
    """Find an afternoon UTC minute reaching each target solar elevation.

    The capture site is at latitude 53.6 degrees, so its equinox culmination is
    near 36 degrees and a single day cannot supply the whole range.  Candidate
    afternoons are therefore drawn from across the year at the same site, which
    keeps the camera geometry fixed while varying the solar elevation.

    Args:
        config: Capture manifest supplying the site and the reference date.
        elevations_deg: Target solar elevations in degrees.

    Returns:
        A mapping from each reachable elevation to a time achieving it.
    """
    location = _location(config)
    year = format_time_utc(Time(config.time_utc, scale="utc"))[:4]
    # Every fourth day of the year, sampled each minute of the afternoon.
    days = Time(f"{year}-01-01T12:00:00", scale="utc") + np.arange(0, 365, 4) * 86400.0 * u.s
    offsets = np.arange(0, 480, 1) * 60.0 * u.s
    grid = Time(
        (days.unix[:, None] + np.asarray(offsets.to_value(u.s))[None, :]).reshape(-1),
        format="unix",
        scale="utc",
    )
    _, altitudes = _sun_altaz(location, grid)

    selected: dict[float, Time] = {}
    for elevation in elevations_deg:
        index = int(np.argmin(np.abs(altitudes - elevation)))
        if abs(float(altitudes[index]) - elevation) > 0.5:
            continue
        selected[elevation] = Time(format_time_utc(grid[index]), scale="utc")
    return selected


def build_sweeps(config: CaptureConfig, *, repeats: int, quick: bool) -> list[dict[str, Any]]:
    """Enumerate every trial specification, one dictionary per trial."""
    base_time = Time(config.time_utc, scale="utc")
    # The site culminates near 60 degrees at the solstice, so the sweep stops there.
    elevations = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
    if quick:
        elevations = [10.0, 25.0, 50.0]
    elevation_times = times_at_solar_elevations(config, elevations)

    specifications: list[dict[str, Any]] = []

    for elevation, time in elevation_times.items():
        specifications.append(
            {
                "sweep": "solar_elevation",
                "setting": f"{elevation:.0f} deg",
                "value": elevation,
                "true_time": time,
                "settings": RenderSettings(),
                "aperture_radius_pixels": DEFAULT_APERTURE_RADIUS_PIXELS,
                "repeat": 0,
            }
        )

    radii = [15, 30, 45, 60, 90] if not quick else [30, 45, 90]
    for radius in radii:
        specifications.append(
            {
                "sweep": "aperture_radius",
                "setting": f"{radius} px",
                "value": float(radius),
                "true_time": base_time,
                "settings": RenderSettings(),
                "aperture_radius_pixels": radius,
                "repeat": 0,
            }
        )

    exposures = [0.5, 0.7, 0.8, 0.9, 0.95] if not quick else [0.7, 0.9]
    for exposure in exposures:
        specifications.append(
            {
                "sweep": "exposure",
                "setting": f"{exposure:.2f}",
                "value": exposure,
                "true_time": base_time,
                "settings": RenderSettings(exposure_fraction=exposure),
                "aperture_radius_pixels": DEFAULT_APERTURE_RADIUS_PIXELS,
                "repeat": 0,
            }
        )

    snrs = [5.0, 10.0, 25.0, 50.0, 200.0] if not quick else [10.0, 50.0]
    for snr in snrs:
        for repeat in range(repeats):
            specifications.append(
                {
                    "sweep": "noise_snr",
                    "setting": f"{snr:.0f}",
                    "value": snr,
                    "true_time": base_time,
                    "settings": RenderSettings(noise_snr=snr, seed=repeat),
                    "aperture_radius_pixels": DEFAULT_APERTURE_RADIUS_PIXELS,
                    "repeat": repeat,
                }
            )

    yaw_errors = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0] if not quick else [-3.0, 0.0, 3.0]
    for yaw_error in yaw_errors:
        specifications.append(
            {
                "sweep": "yaw_error",
                "setting": f"{yaw_error:+.0f} deg",
                "value": yaw_error,
                "true_time": base_time,
                "settings": RenderSettings(yaw_error_deg=yaw_error),
                "aperture_radius_pixels": DEFAULT_APERTURE_RADIUS_PIXELS,
                "repeat": 0,
            }
        )

    occlusions = [0.0, 0.25, 0.5, 0.75] if not quick else [0.0, 0.5]
    for occlusion in occlusions:
        for repeat in range(repeats):
            specifications.append(
                {
                    "sweep": "occlusion",
                    "setting": f"{occlusion:.0%}",
                    "value": occlusion,
                    "true_time": base_time,
                    "settings": RenderSettings(occlusion_fraction=occlusion, seed=repeat),
                    "aperture_radius_pixels": DEFAULT_APERTURE_RADIUS_PIXELS,
                    "repeat": repeat,
                }
            )
    for repeat in range(repeats):
        specifications.append(
            {
                "sweep": "occlusion",
                "setting": "sun occluded",
                "value": 1.0,
                "true_time": base_time,
                "settings": RenderSettings(occlude_sun=True, seed=repeat),
                "aperture_radius_pixels": DEFAULT_APERTURE_RADIUS_PIXELS,
                "repeat": repeat,
            }
        )

    return specifications


def _summarize(trials: list[RecoveryTrial]) -> list[dict[str, Any]]:
    """Aggregate trials by sweep and setting."""
    summary: list[dict[str, Any]] = []
    seen: list[tuple[str, str]] = []
    for trial in trials:
        key = (trial.sweep, trial.setting)
        if key in seen:
            continue
        seen.append(key)
        group = [
            item for item in trials if item.sweep == trial.sweep and item.setting == trial.setting
        ]
        succeeded = [item for item in group if item.succeeded]
        errors = np.asarray(
            [item.absolute_time_error_seconds for item in succeeded], dtype=np.float64
        )
        angular = np.asarray([item.sun_angular_error_deg for item in succeeded], dtype=np.float64)
        summary.append(
            {
                "sweep": trial.sweep,
                "setting": trial.setting,
                "value": trial.value,
                "trials": len(group),
                "successes": len(succeeded),
                "failure_rate": 1.0 - len(succeeded) / len(group),
                "median_absolute_time_error_seconds": (
                    float(np.median(errors)) if errors.size else float("nan")
                ),
                "max_absolute_time_error_seconds": (
                    float(np.max(errors)) if errors.size else float("nan")
                ),
                "median_sun_angular_error_deg": (
                    float(np.median(angular)) if angular.size else float("nan")
                ),
                "max_sun_angular_error_deg": (
                    float(np.max(angular)) if angular.size else float("nan")
                ),
                "median_plateau_width_seconds": (
                    float(np.median([item.plateau_width_seconds for item in succeeded]))
                    if succeeded
                    else float("nan")
                ),
                # A silent failure is worse than a refusal: the method returned a
                # confident time that is wrong by more than its resolution limit.
                "silent_failure_rate": (
                    float(
                        np.mean(
                            [
                                item.absolute_time_error_seconds > SILENT_FAILURE_SECONDS
                                for item in succeeded
                            ]
                        )
                    )
                    if succeeded
                    else 0.0
                ),
            }
        )
    return summary


def measure_real_capture_containment(
    config: CaptureConfig,
    *,
    aperture_radius_pixels: int = DEFAULT_APERTURE_RADIUS_PIXELS,
    minimum_saturation_fraction: float = DEFAULT_MINIMUM_SATURATION_FRACTION,
) -> dict[str, Any]:
    """Score the real capture with the same statistic used on rendered frames.

    This is the control the synthetic sweeps cannot provide.  The rendered Sun
    is a CIE radiance peak with no lens flare, glare or blooming, so its
    starburst is far more compact than the instrument's.  Measuring the real
    capture on the same axis is what shows whether a threshold fitted to
    rendered frames means anything on real data.
    """
    raw = load_tiff(config.input_dir / config.raw_file, (config.image_height, config.image_width))
    candidates = evaluate_candidate_times(
        config,
        raw,
        _candidate_times(
            Time(config.time_utc, scale="utc"),
            DEFAULT_HALF_WINDOW_SECONDS,
            DEFAULT_COARSE_STEP_SECONDS,
        ),
        stage="coarse",
        aperture_radius_pixels=aperture_radius_pixels,
    )
    winner = choose_best_candidate(candidates, minimum_saturation_fraction)
    aperture, annulus, containment = containment_statistics(
        raw,
        winner.pixel_x,
        winner.pixel_y,
        aperture_radius_pixels,
        config.saturation_threshold,
    )
    return {
        "time_utc": winner.time_utc,
        "aperture_radius_pixels": aperture_radius_pixels,
        "aperture_saturation_fraction": aperture,
        "annulus_saturation_fraction": annulus,
        "containment": containment,
        "passes_synthetic_threshold": containment >= DEFAULT_CONTAINMENT_THRESHOLD,
    }


def _guard_report(
    trials: list[RecoveryTrial], real_capture: dict[str, Any] | None
) -> dict[str, Any]:
    """Score the containment guard, then test whether it transfers to real data.

    Two limits bound what this scorecard means.  A yaw error produces a genuine,
    compact detection of the real Sun whose inferred time is nevertheless wrong;
    no single-frame image statistic can detect it, so those trials are counted
    separately rather than held against the guard.  More importantly the whole
    scorecard is computed on rendered frames, and ``real_capture`` records what
    the same statistic does on the instrument.
    """
    accepted = [trial for trial in trials if trial.succeeded]
    silent = [
        trial for trial in accepted if trial.absolute_time_error_seconds > SILENT_FAILURE_SECONDS
    ]
    detectable = [trial for trial in silent if trial.sweep != "yaw_error"]
    correct = [
        trial for trial in accepted if trial.absolute_time_error_seconds <= SILENT_FAILURE_SECONDS
    ]
    caught = [trial for trial in detectable if not trial.guard_passed]
    false_rejections = [trial for trial in correct if not trial.guard_passed]
    return {
        "threshold": DEFAULT_CONTAINMENT_THRESHOLD,
        "statistic": "1 - eroded annulus saturated fraction / aperture saturated fraction",
        "annulus_scale": 3.0,
        "accepted_before_guard": len(accepted),
        "correct_recoveries": len(correct),
        "silent_failures_before_guard": len(silent),
        "image_detectable_silent_failures": len(detectable),
        "silent_failures_caught": len(caught),
        "silent_failures_missed": len(detectable) - len(caught),
        "false_rejections": len(false_rejections),
        "false_rejection_rate": (len(false_rejections) / len(correct) if correct else 0.0),
        "minimum_containment_among_correct": (
            float(min(trial.winner_containment for trial in correct)) if correct else float("nan")
        ),
        "maximum_containment_among_detectable_failures": (
            float(max(trial.winner_containment for trial in detectable))
            if detectable
            else float("nan")
        ),
        "pose_aliased_silent_failures": len(silent) - len(detectable),
        "real_capture_control": real_capture,
        # The scorecard above is fitted to rendered frames. If the real capture's
        # correct detection scores below the worst rendered failure, no threshold
        # on this statistic can separate the two, and the guard must not ship as
        # an accept/reject test.
        "transfers_to_real_capture": (
            bool(
                real_capture is not None
                and detectable
                and real_capture["containment"]
                > max(trial.winner_containment for trial in detectable)
            )
        ),
    }


def _yaw_alias_fit(trials: list[RecoveryTrial]) -> dict[str, float]:
    """Fit recovered time error against the yaw error given to the recovery.

    A camera yaw error rotates the projected solar position, and the search
    absorbs that rotation by moving along the solar track.  The fitted slope is
    the exchange rate between the two, and it bounds how well the acquisition
    time can be known when the pose is uncertain.
    """
    group = [trial for trial in trials if trial.sweep == "yaw_error" and trial.succeeded]
    if len(group) < 2:
        return {"slope_seconds_per_degree": float("nan"), "r_squared": float("nan")}
    yaw = np.asarray([trial.yaw_error_deg for trial in group], dtype=np.float64)
    error = np.asarray([trial.time_error_seconds for trial in group], dtype=np.float64)
    slope, intercept = np.polyfit(yaw, error, 1)
    predicted = slope * yaw + intercept
    total = float(np.sum((error - error.mean()) ** 2))
    residual = float(np.sum((error - predicted) ** 2))
    return {
        "slope_seconds_per_degree": float(slope),
        "intercept_seconds": float(intercept),
        "r_squared": 1.0 - residual / total if total > 0.0 else 0.0,
        "degrees_per_minute_of_time": float(60.0 / slope) if slope else float("nan"),
    }


def _plot_recovery(output_path: Path, trials: list[RecoveryTrial]) -> None:
    """Render the six-panel characterization figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sweep(name: str) -> list[RecoveryTrial]:
        return [trial for trial in trials if trial.sweep == name]

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))

    elevation = [trial for trial in sweep("solar_elevation") if trial.succeeded]
    axes[0, 0].plot(
        [trial.value for trial in elevation],
        [trial.absolute_time_error_seconds for trial in elevation],
        "o-",
        color="#1f77b4",
    )
    axes[0, 0].set_xlabel("solar elevation (deg)")
    axes[0, 0].set_ylabel("|time error| (s)")
    axes[0, 0].set_title("(a) solar elevation")

    aperture = [trial for trial in sweep("aperture_radius") if trial.succeeded]
    axes[0, 1].plot(
        [trial.value for trial in aperture],
        [trial.absolute_time_error_seconds for trial in aperture],
        "s-",
        color="#d62728",
    )
    axes[0, 1].set_xlabel("aperture radius (px)")
    axes[0, 1].set_ylabel("|time error| (s)")
    axes[0, 1].set_title("(b) aperture radius")

    exposure = [trial for trial in sweep("exposure") if trial.succeeded]
    axes[0, 2].semilogy(
        [trial.value for trial in exposure],
        [max(trial.absolute_time_error_seconds, 1.0) for trial in exposure],
        "^-",
        color="#2ca02c",
    )
    axes[0, 2].set_xlabel("auto-exposure saturation fraction")
    axes[0, 2].set_ylabel("|time error| (s), floored at 1")
    axes[0, 2].set_title("(c) exposure")

    yaw = [trial for trial in sweep("yaw_error") if trial.succeeded]
    fit = _yaw_alias_fit(trials)
    axes[1, 0].plot(
        [trial.yaw_error_deg for trial in yaw],
        [trial.time_error_seconds for trial in yaw],
        "o",
        color="#9467bd",
    )
    if np.isfinite(fit["slope_seconds_per_degree"]):
        span = np.linspace(-5.5, 5.5, 32)
        axes[1, 0].plot(
            span,
            fit["slope_seconds_per_degree"] * span + fit["intercept_seconds"],
            "--",
            color="#9467bd",
            label=f"{fit['slope_seconds_per_degree']:.0f} s/deg",
        )
        axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].set_xlabel("camera yaw error (deg)")
    axes[1, 0].set_ylabel("signed time error (s)")
    axes[1, 0].set_title("(d) pose aliasing")

    occlusion = sweep("occlusion")
    fractions = sorted({trial.value for trial in occlusion})
    correct_counts, refused_counts, silent_counts = [], [], []
    for value in fractions:
        group = [trial for trial in occlusion if trial.value == value]
        refused_counts.append(sum(1 for trial in group if not trial.succeeded))
        silent_counts.append(
            sum(
                1
                for trial in group
                if trial.succeeded and trial.absolute_time_error_seconds > SILENT_FAILURE_SECONDS
            )
        )
        correct_counts.append(len(group) - refused_counts[-1] - silent_counts[-1])
    labels = [f"{value:.0%}" if value < 1.0 else "sun hidden" for value in fractions]
    bottom = np.zeros(len(fractions))
    for counts, colour, label in (
        (correct_counts, "#2ca02c", "correct"),
        (refused_counts, "#7f7f7f", "refused"),
        (silent_counts, "#d62728", "silent failure"),
    ):
        axes[1, 1].bar(labels, counts, bottom=bottom, color=colour, label=label)
        bottom = bottom + np.asarray(counts, dtype=np.float64)
    axes[1, 1].set_ylabel("trials")
    axes[1, 1].set_ylim(0.0, float(bottom.max()) * 1.45)
    axes[1, 1].set_title("(e) occlusion")
    axes[1, 1].legend(frameon=False, fontsize=9, ncol=3, loc="upper center")

    default = [trial for trial in trials if trial.succeeded and trial.aperture_radius_pixels == 45]
    correct = [
        trial.winner_containment
        for trial in default
        if trial.absolute_time_error_seconds <= SILENT_FAILURE_SECONDS
    ]
    failed = [
        trial.winner_containment
        for trial in default
        if trial.absolute_time_error_seconds > SILENT_FAILURE_SECONDS and trial.sweep != "yaw_error"
    ]
    # Correct recoveries pile up at containment 1.0; jitter so the count is visible.
    jitter = np.random.default_rng(0)
    axes[1, 2].plot(
        jitter.uniform(-0.18, 0.18, len(correct)),
        correct,
        "o",
        color="#2ca02c",
        alpha=0.65,
        label=f"correct (n={len(correct)})",
    )
    axes[1, 2].plot(
        1.0 + jitter.uniform(-0.18, 0.18, len(failed)),
        failed,
        "o",
        color="#d62728",
        alpha=0.85,
        label=f"silent failure (n={len(failed)})",
    )
    axes[1, 2].axhline(
        DEFAULT_CONTAINMENT_THRESHOLD,
        linestyle="--",
        color="black",
        label=f"guard {DEFAULT_CONTAINMENT_THRESHOLD:.2f}",
    )
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_xticklabels(["correct", "silent"])
    axes[1, 2].set_ylabel("containment")
    axes[1, 2].set_title("(f) containment guard, 45 px")
    axes[1, 2].legend(frameon=False, fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_validation(
    config_path: Path,
    output_dir: Path,
    *,
    repeats: int,
    quick: bool,
    verbose: bool,
) -> int:
    """Run every sweep and write the trial table, summary and report."""
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = build_sweeps(config, repeats=repeats, quick=quick)

    trials: list[RecoveryTrial] = []
    for index, specification in enumerate(specifications, start=1):
        trial = run_trial(
            config,
            specification["true_time"],
            specification["settings"],
            sweep=specification["sweep"],
            setting=specification["setting"],
            value=specification["value"],
            repeat=specification["repeat"],
            aperture_radius_pixels=specification["aperture_radius_pixels"],
        )
        trials.append(trial)
        if verbose:
            status = (
                f"{trial.time_error_seconds:+.0f} s" if trial.succeeded else trial.failure_reason
            )
            print(
                f"[{index:3d}/{len(specifications)}] {trial.sweep:16s} "
                f"{trial.setting:14s} -> {status}",
                flush=True,
            )

    # The control: the same statistic measured on the instrument rather than on a
    # rendered frame. Absent inputs must not fail the sweep, which needs no capture.
    real_capture: dict[str, Any] | None
    try:
        real_capture = measure_real_capture_containment(config)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        real_capture = None
        if verbose:
            print(f"real-capture control unavailable: {error}", flush=True)
    else:
        if verbose:
            print(
                f"real-capture control: containment {real_capture['containment']:.4f} "
                f"at {real_capture['time_utc']}",
                flush=True,
            )

    summary = _summarize(trials)
    fieldnames = list(asdict(trials[0]).keys())
    with (output_dir / "time_recovery_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow(asdict(trial))

    with (output_dir / "time_recovery_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    successes = [trial for trial in trials if trial.succeeded]
    clear = [
        trial for trial in successes if trial.occlusion_fraction == 0.0 and not trial.occlude_sun
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pyskylumos_version": __version__,
        "method": "closed_loop_synthetic_time_recovery",
        "config_path": str(config.config_path),
        "search": {
            "half_window_seconds": DEFAULT_HALF_WINDOW_SECONDS,
            "coarse_step_seconds": DEFAULT_COARSE_STEP_SECONDS,
            "search_displacement_seconds": DEFAULT_SEARCH_DISPLACEMENT_SECONDS,
            "minimum_saturation_fraction": DEFAULT_MINIMUM_SATURATION_FRACTION,
        },
        "rendering": {
            "sky_model": DEFAULT_MODEL,
            "cie_sky_type": DEFAULT_CIE_SKY_TYPE,
            "default_exposure_fraction": DEFAULT_EXPOSURE_FRACTION,
            "default_multiplicative_noise_snr": DEFAULT_NOISE_SNR,
            "dop_scale": config.dop_scale,
        },
        "assumptions": {
            "renderer_and_recovery_share_the_ephemeris_and_lens_model": True,
            "recovery_never_reads_dop_aop_or_a_sky_model": True,
            "clouds_are_opaque_discs_not_a_radiative_transfer_model": True,
            "yaw_error_is_the_only_pose_error_swept": True,
        },
        "totals": {
            "trials": len(trials),
            "successes": len(successes),
            "failure_rate": 1.0 - len(successes) / len(trials),
            "clear_sky_trials": len(clear),
            "clear_sky_exact_minute_rate": (
                float(np.mean([trial.absolute_time_error_seconds <= 60.0 for trial in clear]))
                if clear
                else float("nan")
            ),
            "clear_sky_median_absolute_time_error_seconds": (
                float(np.median([trial.absolute_time_error_seconds for trial in clear]))
                if clear
                else float("nan")
            ),
            "clear_sky_max_absolute_time_error_seconds": (
                float(np.max([trial.absolute_time_error_seconds for trial in clear]))
                if clear
                else float("nan")
            ),
            "silent_failure_rate": (
                float(
                    np.mean(
                        [
                            trial.absolute_time_error_seconds > SILENT_FAILURE_SECONDS
                            for trial in successes
                        ]
                    )
                )
                if successes
                else 0.0
            ),
            "silent_failure_threshold_seconds": SILENT_FAILURE_SECONDS,
        },
        "containment_guard": _guard_report(trials, real_capture),
        "pose_aliasing": _yaw_alias_fit(trials),
        "summary": summary,
        "trials": [asdict(trial) for trial in trials],
    }
    with (output_dir / "time_recovery.json").open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    _plot_recovery(output_dir / "time_recovery.png", trials)

    print(
        f"{len(successes)}/{len(trials)} trials recovered a time; "
        f"clear-sky median |error| "
        f"{report['totals']['clear_sky_median_absolute_time_error_seconds']:.0f} s"
    )
    return 0


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "capture.toml",
        help="Capture manifest supplying the camera and site.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory receiving the trial table, summary and JSON report.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Seeded repeats for the stochastic sweeps (noise and occlusion).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced grid for a smoke test.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-trial progress line.",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Entry point."""
    parsed = parse_arguments(arguments)
    return run_validation(
        parsed.config.expanduser().resolve(),
        parsed.output_dir.expanduser().resolve(),
        repeats=parsed.repeats,
        quick=parsed.quick,
        verbose=not parsed.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
