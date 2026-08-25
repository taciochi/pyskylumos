"""Quantify every canonical PySkyLumos model against a polarimetric capture.

This is deliberately a repository script rather than part of the installed
``pyskylumos`` API.  Capture-specific assumptions live in a TOML manifest, the
large inputs remain local, and every generated report records those assumptions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from numpy.typing import NDArray

from pyskylumos import __version__
from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type Numeric = int | float

CANONICAL_MODELS = (
    "RAYLEIGH",
    "DEPOLARIZED_RAYLEIGH",
    "ASYMMETRIC",
    "BERRY",
    "PAN",
    "QUEEN",
)
POSITIONS = ((0, 0), (0, 1), (1, 0), (1, 1))
FIELD_DOP = "degree of polarization"
FIELD_AOP = "angle of polarization"
FIELD_RADIANCE = "radiance"


@dataclass(frozen=True)
class CaptureConfig:
    """Validated capture and quantification configuration."""

    config_path: Path
    input_dir: Path
    time_utc: str
    latitude_deg: float
    longitude_deg: float
    height_m: float
    image_height: int
    image_width: int
    focal_length_micrometers: float
    pixel_pitch_micrometers: float
    lens_conjugation_type: str
    yaw_deg: float
    altitude_min_deg: float
    usable_image_radius_pixels: float
    sun_exclusion_deg: float
    saturation_threshold: float
    aop_min_measured_dop: float
    analyzer_tile: tuple[tuple[int, int], tuple[int, int]]
    extinction_ratio: float
    adc_max: float
    vendor_aop_offset_deg: float
    dop_scale: float
    raw_file: str
    intensity_file: str
    dop_file: str
    aop_file: str
    checksums: dict[str, str]


@dataclass(frozen=True)
class ErrorMetrics:
    """Scalar residual metrics in native and normalized units."""

    mae: float
    rmse: float
    nmae: float
    nrmse: float
    count: int


@dataclass(frozen=True)
class AffineMetrics:
    """Residual metrics after fitting ``measured = gain * simulated + offset``."""

    mae: float
    rmse: float
    nrmse: float
    r_squared: float
    gain: float
    offset: float
    count: int


@dataclass(frozen=True)
class ModelMetrics:
    """Complete metrics for one canonical model."""

    model: str
    dop: ErrorMetrics
    aop_degrees: ErrorMetrics
    polarization_score: float
    raw: AffineMetrics


@dataclass(frozen=True)
class MeasuredData:
    """Decoded native and analyzer-tile measurements."""

    raw: FloatArray
    intensity: FloatArray
    dop: FloatArray
    aop: FloatArray
    tile_intensity: FloatArray
    tile_dop: FloatArray
    tile_aop: FloatArray


@dataclass(frozen=True)
class EvaluationMasks:
    """Shared native-pixel and analyzer-tile evaluation masks."""

    native: BoolArray
    tile: BoolArray
    aop_tile: BoolArray


@dataclass(frozen=True)
class ModelArrays:
    """Arrays retained long enough to render one model's residual plot."""

    dop: FloatArray
    aop: FloatArray
    raw: FloatArray
    raw_aligned: FloatArray


def _table(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration table [{name}] is required.")
    return cast(dict[str, Any], value)


def _string(table: dict[str, Any], name: str) -> str:
    value = table.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration value {name!r} must be a non-empty string.")
    return value


def _real(table: dict[str, Any], name: str) -> float:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration value {name!r} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Configuration value {name!r} must be finite.")
    return result


def _positive_int(table: dict[str, Any], name: str) -> int:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Configuration value {name!r} must be a positive integer.")
    return value


def _parse_analyzer_tile(value: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("camera.analyzer_tile must be a 2x2 array.")
    rows: list[tuple[int, int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("camera.analyzer_tile must be a 2x2 array.")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in row):
            raise ValueError("camera.analyzer_tile entries must be integer degrees.")
        rows.append((int(row[0]), int(row[1])))
    result = (rows[0], rows[1])
    if {angle for row in result for angle in row} != {0, 45, 90, 135}:
        raise ValueError("camera.analyzer_tile must contain 0, 45, 90 and 135 exactly once.")
    return result


def load_config(path: Path) -> CaptureConfig:
    """Load and validate a capture manifest."""
    config_path = path.expanduser().resolve()
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)

    capture = _table(document, "capture")
    camera = _table(document, "camera")
    measurement = _table(document, "measurement")
    quantification = _table(document, "quantification")
    checksum_values = _table(document, "checksums")

    input_dir_value = _string(measurement, "input_dir")
    input_dir = (config_path.parent / input_dir_value).resolve()
    image_height = _positive_int(camera, "image_height")
    image_width = _positive_int(camera, "image_width")
    if image_height % 2 or image_width % 2:
        raise ValueError("camera image dimensions must both be even for a 2x2 analyzer tile.")

    checksums: dict[str, str] = {}
    for filename, digest in checksum_values.items():
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ValueError("Every checksum entry must map a filename to a SHA-256 string.")
        normalized = digest.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(f"Invalid SHA-256 digest for {filename!r}.")
        checksums[filename] = normalized

    required_files = {
        _string(measurement, "raw_file"),
        _string(measurement, "intensity_file"),
        _string(measurement, "dop_file"),
        _string(measurement, "aop_file"),
    }
    missing_checksums = sorted(required_files - set(checksums))
    if missing_checksums:
        raise ValueError(f"Required measurement files lack checksums: {missing_checksums}.")

    extinction_ratio = _real(camera, "extinction_ratio")
    adc_max = _real(camera, "adc_max")
    altitude_min_deg = _real(camera, "altitude_min_deg")
    usable_image_radius_pixels = _real(camera, "usable_image_radius_pixels")
    sun_exclusion_deg = _real(camera, "sun_exclusion_deg")
    saturation_threshold = _real(camera, "saturation_threshold")
    aop_min_measured_dop = _real(quantification, "aop_min_measured_dop")
    dop_scale = _real(quantification, "dop_scale")
    if not 0.0 <= extinction_ratio <= 1.0:
        raise ValueError("camera.extinction_ratio must lie in [0, 1].")
    if adc_max <= 0.0:
        raise ValueError("camera.adc_max must be positive.")
    if not -90.0 <= altitude_min_deg <= 90.0:
        raise ValueError("camera.altitude_min_deg must lie in [-90, 90].")
    if usable_image_radius_pixels <= 0.0:
        raise ValueError("camera.usable_image_radius_pixels must be positive.")
    if not 0.0 <= sun_exclusion_deg <= 180.0:
        raise ValueError("camera.sun_exclusion_deg must lie in [0, 180].")
    if not 0.0 < saturation_threshold <= adc_max:
        raise ValueError("camera.saturation_threshold must lie in (0, adc_max].")
    if not 0.0 <= aop_min_measured_dop <= 1.0:
        raise ValueError("quantification.aop_min_measured_dop must lie in [0, 1].")
    if not 0.0 < dop_scale <= 1.0:
        raise ValueError("quantification.dop_scale must lie in (0, 1].")

    time_utc = _string(capture, "time_utc")
    try:
        Time(time_utc, scale="utc")
    except ValueError as error:
        raise ValueError(f"capture.time_utc is not a valid UTC time: {time_utc!r}.") from error

    return CaptureConfig(
        config_path=config_path,
        input_dir=input_dir,
        time_utc=time_utc,
        latitude_deg=_real(capture, "latitude_deg"),
        longitude_deg=_real(capture, "longitude_deg"),
        height_m=_real(capture, "height_m"),
        image_height=image_height,
        image_width=image_width,
        focal_length_micrometers=_real(camera, "focal_length_micrometers"),
        pixel_pitch_micrometers=_real(camera, "pixel_pitch_micrometers"),
        lens_conjugation_type=_string(camera, "lens_conjugation_type"),
        yaw_deg=_real(camera, "yaw_deg"),
        altitude_min_deg=altitude_min_deg,
        usable_image_radius_pixels=usable_image_radius_pixels,
        sun_exclusion_deg=sun_exclusion_deg,
        saturation_threshold=saturation_threshold,
        aop_min_measured_dop=aop_min_measured_dop,
        analyzer_tile=_parse_analyzer_tile(camera.get("analyzer_tile")),
        extinction_ratio=extinction_ratio,
        adc_max=adc_max,
        vendor_aop_offset_deg=_real(measurement, "vendor_aop_offset_deg"),
        dop_scale=dop_scale,
        raw_file=_string(measurement, "raw_file"),
        intensity_file=_string(measurement, "intensity_file"),
        dop_file=_string(measurement, "dop_file"),
        aop_file=_string(measurement, "aop_file"),
        checksums=checksums,
    )


def sha256_file(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(config: CaptureConfig) -> dict[str, str]:
    """Verify every input named in the manifest and return actual hashes."""
    if not config.input_dir.is_dir():
        raise FileNotFoundError(
            f"Capture input directory does not exist: {config.input_dir}. "
            "Extract pyskylumos_quantification/input there first."
        )
    actual: dict[str, str] = {}
    for filename, expected in sorted(config.checksums.items()):
        path = config.input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected capture input is missing: {path}.")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {path.name}: expected {expected}, observed {observed}."
            )
        actual[filename] = observed
    return actual


def load_tiff(path: Path, expected_shape: tuple[int, int]) -> FloatArray:
    """Load one single-channel TIFF as float64 after shape validation."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency contract
        raise RuntimeError(
            'Pillow is required; install with `python -m pip install -e ".[quantification]"`.'
        ) from error

    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count != 1:
            raise ValueError(f"Expected a single-frame TIFF, got {frame_count}: {path}.")
        values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"Expected a single-channel TIFF, got shape {values.shape}: {path}.")
    if values.shape != expected_shape:
        raise ValueError(
            f"TIFF shape mismatch for {path.name}: expected {expected_shape}, got {values.shape}."
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Expected integer TIFF samples, got {values.dtype}: {path}.")
    return np.asarray(values, dtype=np.float64)


def pool_scalar_2x2(values: FloatArray) -> FloatArray:
    """Return arithmetic means for non-overlapping 2x2 sensor tiles."""
    if values.ndim != 2 or values.shape[0] % 2 or values.shape[1] % 2:
        raise ValueError(f"Expected an even two-dimensional array, got {values.shape}.")
    height, width = values.shape
    return np.asarray(values.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3)))


def pool_axial_2x2(angles_radians: FloatArray) -> FloatArray:
    """Return 2x2 axial means, respecting AOP's 180-degree periodicity."""
    doubled = np.exp(2j * angles_radians)
    pooled = pool_scalar_2x2(np.asarray(doubled.real, dtype=np.float64)) + 1j * pool_scalar_2x2(
        np.asarray(doubled.imag, dtype=np.float64)
    )
    return np.asarray(0.5 * np.angle(pooled), dtype=np.float64)


def pool_mask_2x2(mask: BoolArray) -> BoolArray:
    """Require all four sensor pixels to pass before accepting a tile."""
    if mask.ndim != 2 or mask.shape[0] % 2 or mask.shape[1] % 2:
        raise ValueError(f"Expected an even two-dimensional mask, got {mask.shape}.")
    height, width = mask.shape
    return np.asarray(mask.reshape(height // 2, 2, width // 2, 2).all(axis=(1, 3)))


def axial_difference_radians(simulated: FloatArray, measured: FloatArray) -> FloatArray:
    """Return signed AOP residuals in ``[-pi/2, pi/2)``."""
    return np.asarray(0.5 * np.angle(np.exp(2j * (simulated - measured))), dtype=np.float64)


def analyzer_response(
    dop: FloatArray,
    aop: FloatArray,
    radiance: FloatArray,
    analyzer_angle_deg: int,
    extinction_ratio: float,
) -> FloatArray:
    """Return the deterministic ideal-analyzer response for one pixel class."""
    angle_radians = np.deg2rad(analyzer_angle_deg)
    return np.asarray(
        0.5 * radiance * (1.0 + extinction_ratio * dop * np.cos(2.0 * (aop - angle_radians))),
        dtype=np.float64,
    )


def scalar_error_metrics(
    simulated: FloatArray,
    measured: FloatArray,
    mask: BoolArray,
    normalization_range: float,
) -> ErrorMetrics:
    """Compute scalar MAE/RMSE and their normalized counterparts."""
    valid = mask & np.isfinite(simulated) & np.isfinite(measured)
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("No finite pixels remain for scalar metrics.")
    differences = simulated[valid] - measured[valid]
    mae = float(np.mean(np.abs(differences)))
    rmse = float(np.sqrt(np.mean(np.square(differences))))
    return ErrorMetrics(
        mae=mae,
        rmse=rmse,
        nmae=mae / normalization_range,
        nrmse=rmse / normalization_range,
        count=count,
    )


def axial_error_metrics(
    simulated: FloatArray,
    measured: FloatArray,
    mask: BoolArray,
) -> ErrorMetrics:
    """Compute 180-degree-periodic AOP metrics in degrees."""
    valid = mask & np.isfinite(simulated) & np.isfinite(measured)
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("No finite pixels remain for AOP metrics.")
    differences_deg = np.rad2deg(axial_difference_radians(simulated, measured)[valid])
    mae = float(np.mean(np.abs(differences_deg)))
    rmse = float(np.sqrt(np.mean(np.square(differences_deg))))
    return ErrorMetrics(mae=mae, rmse=rmse, nmae=mae / 90.0, nrmse=rmse / 90.0, count=count)


def fit_affine(
    simulated: FloatArray,
    measured: FloatArray,
    mask: BoolArray,
    normalization_range: float,
) -> tuple[AffineMetrics, FloatArray]:
    """Fit and evaluate a memory-efficient least-squares affine alignment."""
    valid = mask & np.isfinite(simulated) & np.isfinite(measured)
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise ValueError("No finite pixels remain for affine metrics.")
    x = simulated[valid]
    y = measured[valid]
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    centered_x = x - mean_x
    variance_x = float(np.dot(centered_x, centered_x))
    gain = 0.0 if variance_x == 0.0 else float(np.dot(centered_x, y - mean_y) / variance_x)
    offset = mean_y - gain * mean_x
    aligned = np.asarray(gain * simulated + offset, dtype=np.float64)
    residuals = aligned[valid] - y
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    total = float(np.sum(np.square(y - mean_y)))
    residual_sum = float(np.sum(np.square(residuals)))
    r_squared = 1.0 - residual_sum / total if total > 0.0 else 0.0
    return (
        AffineMetrics(
            mae=mae,
            rmse=rmse,
            nrmse=rmse / normalization_range,
            r_squared=r_squared,
            gain=gain,
            offset=offset,
            count=count,
        ),
        aligned,
    )


def choose_cie_type(candidate_metrics: list[dict[str, Numeric]]) -> int:
    """Choose the lowest-RMSE CIE type, preferring the lower index on ties."""
    if not candidate_metrics:
        raise ValueError("At least one CIE candidate metric is required.")
    winner = min(candidate_metrics, key=lambda row: (row["rmse"], row["cie_sky_type"]))
    return int(winner["cie_sky_type"])


def _load_measurements(config: CaptureConfig) -> MeasuredData:
    shape = (config.image_height, config.image_width)
    raw = load_tiff(config.input_dir / config.raw_file, shape)
    intensity = load_tiff(config.input_dir / config.intensity_file, shape)
    dop_codes = load_tiff(config.input_dir / config.dop_file, shape)
    aop_codes = load_tiff(config.input_dir / config.aop_file, shape)
    dop = np.asarray(dop_codes / config.adc_max, dtype=np.float64)
    aop = np.asarray(
        aop_codes / config.adc_max * np.pi + np.deg2rad(config.vendor_aop_offset_deg),
        dtype=np.float64,
    )
    aop = np.asarray((aop + np.pi / 2.0) % np.pi - np.pi / 2.0, dtype=np.float64)
    return MeasuredData(
        raw=raw,
        intensity=intensity,
        dop=dop,
        aop=aop,
        tile_intensity=pool_scalar_2x2(intensity),
        tile_dop=pool_scalar_2x2(dop),
        tile_aop=pool_axial_2x2(aop),
    )


def _wire_grid(config: CaptureConfig) -> dict[int, SlicingPattern]:
    return {
        config.analyzer_tile[row][column]: SlicingPattern(
            start_row=row, start_column=column, step=2
        )
        for row, column in POSITIONS
    }


def _build_engine(config: CaptureConfig, *, tile_centers: bool = False) -> Engine:
    factor = 2 if tile_centers else 1
    return Engine(
        sensor_pixel_pitch_micrometers=config.pixel_pitch_micrometers * factor,
        lens_conjugation_type=config.lens_conjugation_type,
        number_pixels_vertical=config.image_height // factor,
        number_pixels_horizontal=config.image_width // factor,
        lens_focal_length_micrometers=config.focal_length_micrometers,
        polarizer_tolerance_radians=0.0,
        extinction_ratio=config.extinction_ratio,
        auto_exposure_saturation_fraction=1.0,
        adc_resolution_bits=12,
        multiplicative_noise_snr=1.0e12,
        wire_grid_orientations_slicing=_wire_grid(config),
        random_seed=0,
    )


def _sun_position(config: CaptureConfig) -> tuple[Time, EarthLocation, float, float]:
    times = Time([config.time_utc], scale="utc")
    location = EarthLocation(
        lat=config.latitude_deg * u.deg,
        lon=config.longitude_deg * u.deg,
        height=config.height_m * u.m,
    )
    frame = AltAz(obstime=times[0], location=location)
    sun = get_sun(times[0]).transform_to(frame)
    return times, location, float(sun.az.deg), float(sun.alt.deg)


def _angular_separation_degrees(
    azimuths_deg: FloatArray,
    altitudes_deg: FloatArray,
    sun_azimuth_deg: float,
    sun_altitude_deg: float,
) -> FloatArray:
    azimuths = np.deg2rad(azimuths_deg)
    altitudes = np.deg2rad(altitudes_deg)
    sun_azimuth = np.deg2rad(sun_azimuth_deg)
    sun_altitude = np.deg2rad(sun_altitude_deg)
    cosine = np.sin(altitudes) * np.sin(sun_altitude) + np.cos(altitudes) * np.cos(
        sun_altitude
    ) * np.cos(azimuths - sun_azimuth)
    return np.asarray(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))), dtype=np.float64)


def sensor_pixel_radii(config: CaptureConfig) -> FloatArray:
    """Return each pixel's radial distance from the assumed optical centre.

    The centring matches ``OpticalConjugator``, which places the optical centre
    at the geometric centre of the pixel grid.  This radius is therefore a
    strictly decreasing function of the altitudes that class produces, whatever
    ``lens_conjugation_type`` is configured.
    """
    rows = np.arange(config.image_height, dtype=np.float64) - (config.image_height - 1) / 2.0
    columns = np.arange(config.image_width, dtype=np.float64) - (config.image_width - 1) / 2.0
    return np.asarray(np.hypot(rows[:, None], columns[None, :]), dtype=np.float64)


def build_masks(
    config: CaptureConfig,
    measured: MeasuredData,
    world_azimuths: FloatArray,
    altitudes: FloatArray,
    sun_azimuth_deg: float,
    sun_altitude_deg: float,
) -> EvaluationMasks:
    """Build model-independent physical evaluation masks.

    The outer bound is ``usable_image_radius_pixels``, the lens's usable image
    circle, which keeps the camera rim out of every score.  The remaining
    clauses drop below-horizon geometry, saturated measurements and a region
    around the Sun.  None of them depends on the model being scored.
    """
    separation = _angular_separation_degrees(
        world_azimuths, altitudes, sun_azimuth_deg, sun_altitude_deg
    )
    native = (
        np.isfinite(world_azimuths)
        & np.isfinite(altitudes)
        & np.isfinite(measured.raw)
        & np.isfinite(measured.intensity)
        & np.isfinite(measured.dop)
        & np.isfinite(measured.aop)
        & (altitudes >= config.altitude_min_deg)
        & (sensor_pixel_radii(config) <= config.usable_image_radius_pixels)
        & (separation > config.sun_exclusion_deg)
        & (measured.raw < config.saturation_threshold)
        & (measured.intensity < config.saturation_threshold)
    )
    tile = pool_mask_2x2(np.asarray(native, dtype=np.bool_))
    aop_tile = tile & (measured.tile_dop >= config.aop_min_measured_dop)
    if not np.any(native) or not np.any(tile) or not np.any(aop_tile):
        raise ValueError(
            "The configured physical masks leave no usable capture pixels; "
            "camera.usable_image_radius_pixels may be too small."
        )
    return EvaluationMasks(native=np.asarray(native), tile=tile, aop_tile=aop_tile)


def validate_vendor_polarization(
    config: CaptureConfig,
    measured: MeasuredData,
    masks: EvaluationMasks,
) -> dict[str, Any]:
    """Cross-check the vendor DoLP/Azimuth convention against the raw mosaic."""
    channel: dict[int, FloatArray] = {}
    for row, column in POSITIONS:
        angle = config.analyzer_tile[row][column]
        channel[angle] = measured.raw[row::2, column::2]
    s0 = 0.5 * (channel[0] + channel[45] + channel[90] + channel[135])
    raw_dop = np.divide(
        np.hypot(channel[0] - channel[90], channel[45] - channel[135]),
        s0,
        out=np.full_like(s0, np.nan),
        where=s0 != 0.0,
    )
    raw_aop = np.asarray(
        0.5 * np.arctan2(channel[45] - channel[135], channel[0] - channel[90]),
        dtype=np.float64,
    )
    dop_metrics = scalar_error_metrics(raw_dop, measured.tile_dop, masks.tile, 1.0)
    aop_metrics = axial_error_metrics(raw_aop, measured.tile_aop, masks.aop_tile)
    return {
        "raw_reconstruction_vs_vendor_dop": asdict(dop_metrics),
        "raw_reconstruction_vs_vendor_aop_degrees": asdict(aop_metrics),
    }


def _extract_fields(values: Any, names: Any) -> tuple[FloatArray, FloatArray, FloatArray]:
    fields = dict(zip(names, values, strict=True))
    try:
        dop = fields[FIELD_DOP]
        aop = fields[FIELD_AOP]
        radiance = fields[FIELD_RADIANCE]
    except KeyError as error:
        raise RuntimeError(f"Sky model omitted required field {error.args[0]!r}.") from error
    return (
        np.asarray(dop[0], dtype=np.float64),
        np.asarray(aop[0], dtype=np.float64),
        np.asarray(radiance[0], dtype=np.float64),
    )


def _simulate(
    engine: Engine,
    config: CaptureConfig,
    times: Time,
    location: EarthLocation,
    model: str,
    cie_sky_type: int,
    world_azimuths: FloatArray,
    altitudes: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    # The CIE luminance formula is undefined below the horizon and can overflow
    # before SkySimulator applies its output mask.  Evaluation masks retain the
    # original geometry, so clipping only these ignored samples is lossless for
    # every reported metric and keeps warnings-as-errors runs clean.
    simulation_altitudes = np.maximum(altitudes, config.altitude_min_deg)
    values, names = engine.simulate_sky_polarization(
        sky_model=model,
        observation_location=location,
        times=times,
        cie_sky_type=cie_sky_type,
        altitudes=simulation_altitudes,
        azimuths=world_azimuths,
        altitude_min_clip=None,
        azimuth_rotation_angle=config.yaw_deg,
        accuracy=False,
    )
    return _extract_fields(values, names)


def _select_cie_sky_type(
    config: CaptureConfig,
    measured: MeasuredData,
    masks: EvaluationMasks,
    times: Time,
    location: EarthLocation,
) -> tuple[int, list[dict[str, Numeric]]]:
    engine = _build_engine(config, tile_centers=True)
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=None)
    world_azimuths = np.asarray(engine.rotate_sensor(azimuths, config.yaw_deg), dtype=np.float64)
    rows: list[dict[str, Numeric]] = []
    for cie_sky_type in range(1, 16):
        _, _, radiance = _simulate(
            engine,
            config,
            times,
            location,
            "RAYLEIGH",
            cie_sky_type,
            world_azimuths,
            altitudes,
        )
        metrics, _ = fit_affine(
            radiance,
            measured.tile_intensity,
            masks.tile,
            config.adc_max,
        )
        rows.append(
            {
                "cie_sky_type": cie_sky_type,
                "rmse": metrics.rmse,
                "nrmse": metrics.nrmse,
                "r_squared": metrics.r_squared,
                "gain": metrics.gain,
                "offset": metrics.offset,
            }
        )
    return choose_cie_type(rows), rows


def _quantify_model(
    config: CaptureConfig,
    measured: MeasuredData,
    masks: EvaluationMasks,
    engine: Engine,
    times: Time,
    location: EarthLocation,
    model: str,
    cie_sky_type: int,
    world_azimuths: FloatArray,
    altitudes: FloatArray,
) -> tuple[ModelMetrics, ModelArrays]:
    tile_shape = (config.image_height // 2, config.image_width // 2)
    dop_sum = np.zeros(tile_shape, dtype=np.float64)
    axial_sum = np.zeros(tile_shape, dtype=np.complex128)
    raw_prediction = np.full((config.image_height, config.image_width), np.nan, dtype=np.float64)

    for row, column in POSITIONS:
        dop, aop, radiance = _simulate(
            engine,
            config,
            times,
            location,
            model,
            cie_sky_type,
            world_azimuths[row::2, column::2],
            altitudes[row::2, column::2],
        )
        scaled_dop = np.asarray(config.dop_scale * dop, dtype=np.float64)
        dop_sum += scaled_dop
        axial_sum += np.exp(2j * aop)
        analyzer_angle = config.analyzer_tile[row][column]
        raw_prediction[row::2, column::2] = analyzer_response(
            scaled_dop,
            aop,
            radiance,
            analyzer_angle,
            config.extinction_ratio,
        )

    model_dop = np.asarray(dop_sum / 4.0, dtype=np.float64)
    model_aop = np.asarray(0.5 * np.angle(axial_sum), dtype=np.float64)
    if not np.all(np.isfinite(model_dop[masks.tile])):
        raise RuntimeError(f"{model} produced non-finite DOP inside the common mask.")
    if not np.all(np.isfinite(model_aop[masks.aop_tile])):
        raise RuntimeError(f"{model} produced non-finite AOP inside the common mask.")

    dop_metrics = scalar_error_metrics(
        model_dop,
        measured.tile_dop,
        masks.tile,
        config.dop_scale,
    )
    aop_metrics = axial_error_metrics(model_aop, measured.tile_aop, masks.aop_tile)
    raw_metrics, raw_aligned = fit_affine(
        raw_prediction,
        measured.raw,
        masks.native,
        config.adc_max,
    )
    score = float(np.sqrt((dop_metrics.nrmse**2 + aop_metrics.nrmse**2) / 2.0))
    return (
        ModelMetrics(
            model=model,
            dop=dop_metrics,
            aop_degrees=aop_metrics,
            polarization_score=score,
            raw=raw_metrics,
        ),
        ModelArrays(
            dop=model_dop,
            aop=model_aop,
            raw=raw_prediction,
            raw_aligned=raw_aligned,
        ),
    )


def _masked(values: FloatArray, mask: BoolArray) -> FloatArray:
    return np.where(mask, values, np.nan)


def _plot_model(
    output_path: Path,
    model: str,
    measured: MeasuredData,
    masks: EvaluationMasks,
    arrays: ModelArrays,
    field_label: str,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - dependency contract
        raise RuntimeError(
            'Matplotlib is required; install with `python -m pip install -e ".[quantification]"`.'
        ) from error

    dop_residual = arrays.dop - measured.tile_dop
    aop_residual = np.rad2deg(axial_difference_radians(arrays.aop, measured.tile_aop))
    raw_residual = arrays.raw_aligned - measured.raw
    raw_masked = _masked(measured.raw, masks.native)[::2, ::2]
    raw_aligned_masked = _masked(arrays.raw_aligned, masks.native)[::2, ::2]
    raw_residual_masked = _masked(raw_residual, masks.native)[::2, ::2]
    finite_raw_residual = raw_residual_masked[np.isfinite(raw_residual_masked)]
    raw_limit = float(np.percentile(np.abs(finite_raw_residual), 99.0))
    raw_limit = max(raw_limit, 1.0)

    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    panels = (
        (_masked(measured.tile_dop, masks.tile), "Measured DOP", "viridis", 0.0, 1.0),
        (_masked(arrays.dop, masks.tile), "Simulated DOP (scaled)", "viridis", 0.0, 1.0),
        (_masked(dop_residual, masks.tile), "DOP residual", "coolwarm", -0.7, 0.7),
        (
            _masked(np.rad2deg(measured.tile_aop), masks.aop_tile),
            "Measured AOP (deg)",
            "twilight",
            -90.0,
            90.0,
        ),
        (
            _masked(np.rad2deg(arrays.aop), masks.aop_tile),
            "Simulated AOP (deg)",
            "twilight",
            -90.0,
            90.0,
        ),
        (
            _masked(aop_residual, masks.aop_tile),
            "Axial AOP residual (deg)",
            "coolwarm",
            -90.0,
            90.0,
        ),
        (raw_masked, "Measured raw", "gray", 0.0, 4095.0),
        (raw_aligned_masked, "Affine-aligned simulated raw", "gray", 0.0, 4095.0),
        (raw_residual_masked, "Raw residual", "coolwarm", -raw_limit, raw_limit),
    )
    for axis, (image, title, color_map, minimum, maximum) in zip(axes.flat, panels, strict=True):
        handle = axis.imshow(image, cmap=color_map, vmin=minimum, vmax=maximum)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{model}: capture quantification ({field_label})")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def _plot_cie_candidates(output_path: Path, rows: list[dict[str, Numeric]], selected: int) -> None:
    import matplotlib.pyplot as plt

    types = [int(row["cie_sky_type"]) for row in rows]
    errors = [row["nrmse"] for row in rows]
    colors = ["#d62728" if value == selected else "#4c78a8" for value in types]
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(types, errors, color=colors)
    axis.set_xticks(types)
    axis.set_xlabel("CIE sky type")
    axis.set_ylabel("Affine intensity NRMSE")
    axis.set_title(f"Shared radiance calibration (selected CIE type {selected})")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_ranking(output_path: Path, results: list[ModelMetrics]) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(results, key=lambda result: result.polarization_score)
    labels = [result.model for result in ordered]
    scores = [result.polarization_score for result in ordered]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.bar(labels, scores, color="#4c78a8")
    axis.set_ylabel("Equal-weight normalized DOP/AOP RMSE")
    axis.set_title("PySkyLumos model ranking (lower is better)")
    axis.tick_params(axis="x", rotation=25)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, results: list[ModelMetrics]) -> None:
    fieldnames = (
        "rank",
        "model",
        "polarization_score",
        "dop_mae",
        "dop_rmse",
        "dop_nmae",
        "dop_nrmse",
        "dop_count",
        "aop_mae_deg",
        "aop_rmse_deg",
        "aop_nmae",
        "aop_nrmse",
        "aop_count",
        "raw_mae_counts",
        "raw_rmse_counts",
        "raw_nrmse",
        "raw_r_squared",
        "raw_gain",
        "raw_offset",
        "raw_count",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(
            sorted(results, key=lambda item: item.polarization_score), start=1
        ):
            writer.writerow(
                {
                    "rank": rank,
                    "model": result.model,
                    "polarization_score": result.polarization_score,
                    "dop_mae": result.dop.mae,
                    "dop_rmse": result.dop.rmse,
                    "dop_nmae": result.dop.nmae,
                    "dop_nrmse": result.dop.nrmse,
                    "dop_count": result.dop.count,
                    "aop_mae_deg": result.aop_degrees.mae,
                    "aop_rmse_deg": result.aop_degrees.rmse,
                    "aop_nmae": result.aop_degrees.nmae,
                    "aop_nrmse": result.aop_degrees.nrmse,
                    "aop_count": result.aop_degrees.count,
                    "raw_mae_counts": result.raw.mae,
                    "raw_rmse_counts": result.raw.rmse,
                    "raw_nrmse": result.raw.nrmse,
                    "raw_r_squared": result.raw.r_squared,
                    "raw_gain": result.raw.gain,
                    "raw_offset": result.raw.offset,
                    "raw_count": result.raw.count,
                }
            )


def run_quantification(
    config: CaptureConfig,
    output_dir: Path,
    models: tuple[str, ...],
    *,
    show: bool,
) -> dict[str, Any]:
    """Execute the complete manifest-driven quantification workflow."""
    checksums = verify_checksums(config)
    measured = _load_measurements(config)
    times, location, sun_azimuth_deg, sun_altitude_deg = _sun_position(config)
    engine = _build_engine(config)
    sensor_azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=None)
    world_azimuths = np.asarray(
        engine.rotate_sensor(sensor_azimuths, config.yaw_deg), dtype=np.float64
    )
    masks = build_masks(
        config,
        measured,
        world_azimuths,
        altitudes,
        sun_azimuth_deg,
        sun_altitude_deg,
    )
    vendor_validation = validate_vendor_polarization(config, measured, masks)
    # Describe the retained field through the configured projection rather than
    # re-deriving it, so the reported altitude follows lens_conjugation_type.
    inside_field = sensor_pixel_radii(config) <= config.usable_image_radius_pixels
    field_altitude_min_deg = float(np.nanmin(altitudes[inside_field]))
    field_sky_fraction = float(1.0 - np.cos(np.deg2rad(90.0 - field_altitude_min_deg)))
    field_label = (
        f"r ≤ {config.usable_image_radius_pixels:.0f} px, altitude ≥ {field_altitude_min_deg:.1f}°"
    )
    cie_sky_type, cie_rows = _select_cie_sky_type(config, measured, masks, times, location)

    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    results: list[ModelMetrics] = []
    for model in models:
        metrics, arrays = _quantify_model(
            config,
            measured,
            masks,
            engine,
            times,
            location,
            model,
            cie_sky_type,
            world_azimuths,
            altitudes,
        )
        results.append(metrics)
        _plot_model(
            resolved_output / f"{model.lower()}_residuals.png",
            model,
            measured,
            masks,
            arrays,
            field_label,
            show,
        )
        print(
            f"{model}: score={metrics.polarization_score:.6f}, "
            f"DOP RMSE={metrics.dop.rmse:.6f}, "
            f"AOP RMSE={metrics.aop_degrees.rmse:.3f} deg, "
            f"raw NRMSE={metrics.raw.nrmse:.6f}"
        )

    _write_csv(resolved_output / "metrics.csv", results)
    _plot_cie_candidates(resolved_output / "cie_selection.png", cie_rows, cie_sky_type)
    _plot_ranking(resolved_output / "model_ranking.png", results)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pyskylumos_version": __version__,
        "config_path": str(config.config_path),
        "input_checksums": checksums,
        "capture": {
            "time_utc": config.time_utc,
            "latitude_deg": config.latitude_deg,
            "longitude_deg": config.longitude_deg,
            "height_m": config.height_m,
            "sun_azimuth_deg": sun_azimuth_deg,
            "sun_altitude_deg": sun_altitude_deg,
        },
        "camera": {
            "image_shape": [config.image_height, config.image_width],
            "focal_length_micrometers": config.focal_length_micrometers,
            "pixel_pitch_micrometers": config.pixel_pitch_micrometers,
            "lens_conjugation_type": config.lens_conjugation_type,
            "yaw_deg": config.yaw_deg,
            "analyzer_tile": config.analyzer_tile,
            "extinction_ratio": config.extinction_ratio,
            "vendor_aop_offset_deg": config.vendor_aop_offset_deg,
        },
        "quantification": {
            "dop_scale": config.dop_scale,
            "dop_normalization_range": config.dop_scale,
            "aop_period_deg": 180.0,
            "aop_normalization_range_deg": 90.0,
            "altitude_min_deg": config.altitude_min_deg,
            "usable_image_radius_pixels": config.usable_image_radius_pixels,
            "usable_image_radius_altitude_deg": field_altitude_min_deg,
            "usable_image_sky_fraction": field_sky_fraction,
            "sun_exclusion_deg": config.sun_exclusion_deg,
            "saturation_threshold": config.saturation_threshold,
            "aop_min_measured_dop": config.aop_min_measured_dop,
            "selected_cie_sky_type": cie_sky_type,
            "raw_affine_fit_is_diagnostic_only": True,
            "partly_cloudy_single_capture": True,
        },
        "mask_counts": {
            "native": int(np.count_nonzero(masks.native)),
            "dop_tiles": int(np.count_nonzero(masks.tile)),
            "aop_tiles": int(np.count_nonzero(masks.aop_tile)),
        },
        "vendor_convention_validation": vendor_validation,
        "cie_candidates": cie_rows,
        "models": [
            asdict(result) for result in sorted(results, key=lambda item: item.polarization_score)
        ],
    }
    with (resolved_output / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _model_name(value: str) -> str:
    normalized = value.upper()
    if normalized not in CANONICAL_MODELS:
        raise argparse.ArgumentTypeError(
            f"unknown model {value!r}; choose from {', '.join(CANONICAL_MODELS)}"
        )
    return normalized


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the repository script's command-line interface."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Quantify canonical PySkyLumos models against a polarimetric capture."
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
        help="Directory for CSV, JSON and PNG artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=_model_name,
        default=list(CANONICAL_MODELS),
        help="Canonical models to evaluate (default: all six).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each residual figure interactively after saving it.",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Run the command-line workflow and return a process status."""
    options = parse_arguments(arguments)
    models = tuple(dict.fromkeys(options.models))
    config = load_config(options.config)
    report = run_quantification(config, options.output_dir, models, show=options.show)
    print(f"Selected CIE sky type: {report['quantification']['selected_cie_sky_type']}")
    print(f"Wrote quantification artifacts to {Path(options.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
