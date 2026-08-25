"""Tests for the repository-local capture quantification workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyskylumos.sensor import MicroPolarizer, SlicingPattern
from quantification.quantify_models import (
    MeasuredData,
    analyzer_response,
    axial_difference_radians,
    build_masks,
    choose_cie_type,
    fit_affine,
    load_config,
    load_tiff,
    main,
    pool_axial_2x2,
    pool_mask_2x2,
    pool_scalar_2x2,
    scalar_error_metrics,
    verify_checksums,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tiff(path: Path, values: np.ndarray) -> None:
    Image.fromarray(np.asarray(values, dtype=np.uint16)).save(path)


def _write_capture(
    tmp_path: Path, *, height: int = 8, width: int = 8, radius: float = 1000.0
) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    rows, columns = np.indices((height, width))
    raw = 500 + 17 * rows + 11 * columns
    intensity = 700 + 13 * rows + 19 * columns
    dop = 900 + 20 * rows + 15 * columns
    aop = (500 + 73 * rows + 41 * columns) % 3500
    files = {
        "raw.tif": raw,
        "intensity.tif": intensity,
        "dop.tif": dop,
        "aop.tif": aop,
    }
    for filename, values in files.items():
        _write_tiff(input_dir / filename, values)
    checksums = "\n".join(f'"{filename}" = "{_digest(input_dir / filename)}"' for filename in files)
    manifest = tmp_path / "capture.toml"
    manifest.write_text(
        f"""
[capture]
time_utc = "2025-09-22T16:05:00"
latitude_deg = 53.6358
longitude_deg = -2.52229
height_m = 50.0

[camera]
image_height = {height}
image_width = {width}
focal_length_micrometers = 20.0
pixel_pitch_micrometers = 1.0
lens_conjugation_type = "stereographic"
yaw_deg = 15.0
altitude_min_deg = 1.0
usable_image_radius_pixels = {radius}
sun_exclusion_deg = 0.0
saturation_threshold = 4090.0
analyzer_tile = [[90, 45], [135, 0]]
extinction_ratio = 0.99
adc_max = 4095.0

[measurement]
input_dir = "input"
raw_file = "raw.tif"
intensity_file = "intensity.tif"
dop_file = "dop.tif"
aop_file = "aop.tif"
vendor_aop_offset_deg = 90.0

[quantification]
dop_scale = 0.7
aop_min_measured_dop = 0.05

[checksums]
{checksums}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_scalar_and_axial_pooling() -> None:
    scalar = np.arange(16, dtype=np.float64).reshape(4, 4)
    np.testing.assert_allclose(
        pool_scalar_2x2(scalar),
        np.array([[2.5, 4.5], [10.5, 12.5]]),
    )
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    np.testing.assert_array_equal(
        pool_mask_2x2(mask),
        np.array([[False, True], [True, True]]),
    )

    angles = np.deg2rad(np.array([[89.0, -89.0], [89.0, -89.0]]))
    pooled = float(np.rad2deg(pool_axial_2x2(angles)[0, 0]))
    assert abs(abs(pooled) - 90.0) < 1e-12


def test_aop_difference_is_axial_not_directional() -> None:
    simulated = np.deg2rad(np.array([[-89.0]]))
    measured = np.deg2rad(np.array([[89.0]]))
    difference = float(np.rad2deg(axial_difference_radians(simulated, measured)[0, 0]))
    assert difference == pytest.approx(2.0)


def test_dop_scale_is_applied_once_and_analyzer_matches_sensor() -> None:
    unscaled = np.full((2, 2), 0.8)
    scaled = 0.7 * unscaled
    assert scaled[0, 0] == pytest.approx(0.56)
    assert scaled[0, 0] != pytest.approx(0.7 * scaled[0, 0])
    aop = np.full((2, 2), 0.2)
    radiance = np.full((2, 2), 3.0)
    tile = ((90, 45), (135, 0))
    expected = np.empty((2, 2), dtype=np.float64)
    mapping: dict[int, SlicingPattern] = {}
    for row in range(2):
        for column in range(2):
            angle = tile[row][column]
            expected[row, column] = analyzer_response(
                scaled[row : row + 1, column : column + 1],
                aop[row : row + 1, column : column + 1],
                radiance[row : row + 1, column : column + 1],
                angle,
                0.99,
            )[0, 0]
            mapping[angle] = SlicingPattern(row, column, 2)

    sensor = MicroPolarizer(
        polarizer_tolerance_radians=0.0,
        extinction_ratio=0.99,
        wire_grid_orientations_slicing=mapping,
        random_seed=0,
    )
    actual = sensor.get_intensity_on_pixel(
        scaled[None, :, :], aop[None, :, :], radiance[None, :, :]
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)


def test_error_normalization_and_affine_fit() -> None:
    simulated = np.array([[0.0, 0.35], [0.7, 0.2]])
    measured = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=bool)
    metrics = scalar_error_metrics(simulated, measured, mask, 0.7)
    assert metrics.nrmse == pytest.approx(metrics.rmse / 0.7)

    source = np.arange(9, dtype=np.float64).reshape(3, 3)
    target = 2.5 * source + 7.0
    affine, aligned = fit_affine(source, target, np.ones((3, 3), dtype=bool), 4095.0)
    assert affine.gain == pytest.approx(2.5)
    assert affine.offset == pytest.approx(7.0)
    assert affine.rmse < 1e-12
    assert affine.r_squared == pytest.approx(1.0)
    np.testing.assert_allclose(aligned, target, atol=1e-12)


def test_cie_selection_uses_rmse_then_lower_type() -> None:
    candidates = [
        {"cie_sky_type": 4.0, "rmse": 2.0},
        {"cie_sky_type": 3.0, "rmse": 1.0},
        {"cie_sky_type": 2.0, "rmse": 1.0},
    ]
    assert choose_cie_type(candidates) == 2
    with pytest.raises(ValueError, match="At least one"):
        choose_cie_type([])


def test_config_tiff_and_checksum_validation(tmp_path: Path) -> None:
    manifest = _write_capture(tmp_path)
    config = load_config(manifest)
    assert config.dop_scale == 0.7
    assert config.analyzer_tile == ((90, 45), (135, 0))
    assert set(verify_checksums(config)) == {"raw.tif", "intensity.tif", "dop.tif", "aop.tif"}
    assert load_tiff(config.input_dir / config.raw_file, (8, 8)).shape == (8, 8)

    (config.input_dir / config.raw_file).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(config)


def test_physical_mask_is_common_and_low_dop_only_affects_aop(tmp_path: Path) -> None:
    config = load_config(_write_capture(tmp_path, height=4, width=4))
    shape = (4, 4)
    raw = np.full(shape, 1000.0)
    intensity = np.full(shape, 1000.0)
    dop = np.full(shape, 0.4)
    aop = np.zeros(shape)
    raw[0, 0] = config.saturation_threshold
    dop[2:, 2:] = 0.01
    measured = MeasuredData(
        raw=raw,
        intensity=intensity,
        dop=dop,
        aop=aop,
        tile_intensity=pool_scalar_2x2(intensity),
        tile_dop=pool_scalar_2x2(dop),
        tile_aop=pool_axial_2x2(aop),
    )
    masks = build_masks(
        config,
        measured,
        np.zeros(shape),
        np.full(shape, 30.0),
        180.0,
        30.0,
    )
    assert not masks.tile[0, 0]
    assert masks.tile[0, 1]
    assert masks.tile[1, 0]
    assert masks.tile[1, 1]
    assert not masks.aop_tile[1, 1]


def test_usable_image_radius_trims_the_outer_field(tmp_path: Path) -> None:
    config = load_config(_write_capture(tmp_path, height=6, width=6, radius=1.0))
    shape = (6, 6)
    measured = MeasuredData(
        raw=np.full(shape, 1000.0),
        intensity=np.full(shape, 1000.0),
        dop=np.full(shape, 0.4),
        aop=np.zeros(shape),
        tile_intensity=pool_scalar_2x2(np.full(shape, 1000.0)),
        tile_dop=pool_scalar_2x2(np.full(shape, 0.4)),
        tile_aop=pool_axial_2x2(np.zeros(shape)),
    )
    masks = build_masks(
        config,
        measured,
        np.zeros(shape),
        np.full(shape, 30.0),
        180.0,
        30.0,
    )
    # Only the central tile lies within one pixel of the optical centre; every
    # measurement is otherwise acceptable, so the radius alone decides the mask.
    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(masks.tile, expected)


def test_non_positive_usable_image_radius_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="usable_image_radius_pixels"):
        load_config(_write_capture(tmp_path, radius=0.0))


def test_synthetic_end_to_end_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _write_capture(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    assert (
        main(
            [
                "--config",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--models",
                "rayleigh",
            ]
        )
        == 0
    )
    expected = {
        "metrics.csv",
        "results.json",
        "cie_selection.png",
        "model_ranking.png",
        "rayleigh_residuals.png",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    report = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert report["quantification"]["dop_scale"] == 0.7
    assert report["models"][0]["model"] == "RAYLEIGH"
    assert report["models"][0]["dop"]["count"] == report["mask_counts"]["dop_tiles"]
