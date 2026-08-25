"""Tests for image-anchored Sun-time calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from PIL import Image

from quantification.calibrate_sun_time import (
    SunTimeCandidate,
    _score_aperture,
    choose_best_candidate,
    main,
    parse_arguments,
    project_altaz_to_sensor,
    replace_capture_time_text,
    write_capture_time,
)
from quantification.quantify_models import CaptureConfig, _build_engine, load_config


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera_config(tmp_path: Path, *, height: int = 128, width: int = 128) -> CaptureConfig:
    return CaptureConfig(
        config_path=tmp_path / "capture.toml",
        input_dir=tmp_path / "input",
        time_utc="2025-09-22T12:00:00",
        latitude_deg=0.0,
        longitude_deg=0.0,
        height_m=0.0,
        image_height=height,
        image_width=width,
        focal_length_micrometers=20.0,
        pixel_pitch_micrometers=1.0,
        lens_conjugation_type="stereographic",
        yaw_deg=15.0,
        altitude_min_deg=1.0,
        sun_exclusion_deg=10.0,
        saturation_threshold=4090.0,
        aop_min_measured_dop=0.05,
        analyzer_tile=((90, 45), (135, 0)),
        extinction_ratio=0.99,
        adc_max=4095.0,
        vendor_aop_offset_deg=90.0,
        dop_scale=0.7,
        raw_file="raw.tif",
        intensity_file="intensity.tif",
        dop_file="dop.tif",
        aop_file="aop.tif",
        checksums={},
    )


def _candidate(
    time_utc: str,
    unix_seconds: float,
    saturation_fraction: float,
    mean_raw: float,
    *,
    admissible: bool = True,
) -> SunTimeCandidate:
    return SunTimeCandidate(
        stage="test",
        time_utc=time_utc,
        unix_seconds=unix_seconds,
        sun_azimuth_deg=180.0,
        sun_altitude_deg=30.0,
        pixel_x=50.0,
        pixel_y=50.0,
        admissible=admissible,
        reason="" if admissible else "outside",
        saturation_fraction=saturation_fraction,
        mean_raw=mean_raw,
    )


def _write_synthetic_capture(tmp_path: Path) -> tuple[Path, str]:
    height = width = 256
    target_time = "2025-09-22T12:00:00"
    manifest_time = "2025-09-22T12:02:00"
    config = replace(
        _camera_config(tmp_path, height=height, width=width),
        time_utc=manifest_time,
        focal_length_micrometers=500.0,
        yaw_deg=0.0,
    )
    target = Time(target_time, scale="utc")
    location = EarthLocation(lat=0.0 * u.deg, lon=0.0 * u.deg, height=0.0 * u.m)
    sun = get_sun(target).transform_to(AltAz(obstime=target, location=location))
    target_x, target_y = project_altaz_to_sensor(config, float(sun.az.deg), float(sun.alt.deg))

    rows, columns = np.ogrid[:height, :width]
    raw = np.full((height, width), 1000, dtype=np.uint16)
    raw[(columns - target_x) ** 2 + (rows - target_y) ** 2 <= 3.0**2] = 4090
    raw[(columns - 20.0) ** 2 + (rows - 20.0) ** 2 <= 6.0**2] = 4095
    files = {
        "raw.tif": raw,
        "intensity.tif": np.full((height, width), 1000, dtype=np.uint16),
        "dop.tif": np.full((height, width), 1000, dtype=np.uint16),
        "aop.tif": np.full((height, width), 1000, dtype=np.uint16),
    }
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for filename, values in files.items():
        Image.fromarray(values).save(input_dir / filename)
    checksums = "\n".join(f'"{filename}" = "{_digest(input_dir / filename)}"' for filename in files)
    manifest = tmp_path / "capture.toml"
    manifest.write_text(
        f"""
# Preserve this comment and all unrelated values.
[capture]
time_utc = "{manifest_time}"
latitude_deg = 0.0
longitude_deg = 0.0
height_m = 0.0

[camera]
image_height = {height}
image_width = {width}
focal_length_micrometers = 500.0
pixel_pitch_micrometers = 1.0
lens_conjugation_type = "stereographic"
yaw_deg = 0.0
altitude_min_deg = 1.0
sun_exclusion_deg = 10.0
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
    return manifest, target_time


def test_projection_matches_engine_direction_grid(tmp_path: Path) -> None:
    config = _camera_config(tmp_path)
    engine = _build_engine(config)
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=None)
    world_azimuths = engine.rotate_sensor(azimuths, config.yaw_deg)
    for row, column in ((20, 80), (64, 64), (100, 40)):
        pixel_x, pixel_y = project_altaz_to_sensor(
            config,
            float(world_azimuths[row, column]),
            float(altitudes[row, column]),
        )
        assert pixel_x == pytest.approx(column, abs=1e-12)
        assert pixel_y == pytest.approx(row, abs=1e-12)

    with pytest.raises(ValueError, match="stereographic"):
        project_altaz_to_sensor(replace(config, lens_conjugation_type="thin"), 180.0, 30.0)
    with pytest.raises(ValueError, match="finite"):
        project_altaz_to_sensor(config, np.nan, 30.0)


def test_candidate_ranking_thresholds_and_aperture_rejection() -> None:
    earlier = _candidate("2025-09-22T15:04:00", 1.0, 0.5, 3000.0)
    later = _candidate("2025-09-22T15:05:00", 2.0, 0.5, 3000.0)
    brighter = _candidate("2025-09-22T15:06:00", 3.0, 0.5, 3100.0)
    assert choose_best_candidate([later, earlier], 0.1) == earlier
    assert choose_best_candidate([earlier, brighter], 0.1) == brighter

    with pytest.raises(ValueError, match="fully inside"):
        choose_best_candidate(
            [_candidate("2025-09-22T15:00:00", 0.0, 1.0, 4095.0, admissible=False)], 0.1
        )
    with pytest.raises(ValueError, match="No credible visible Sun"):
        choose_best_candidate([_candidate("2025-09-22T15:00:00", 0.0, 0.05, 2000.0)], 0.1)

    admissible, reason, saturation, mean = _score_aperture(
        np.ones((20, 20), dtype=np.float64),
        1.0,
        1.0,
        4,
        4090.0,
    )
    assert not admissible
    assert reason == "aperture_outside_sensor"
    assert saturation == mean == 0.0


def test_capture_time_replacement_is_scoped_and_atomic(tmp_path: Path) -> None:
    original = (
        '# keep\n[capture]\ntime_utc = "2025-09-22T16:05:00" # source\nvalue = 1\n'
        '[other]\ntime_utc = "unchanged"\n'
    )
    expected = original.replace("2025-09-22T16:05:00", "2025-09-22T15:05:00")
    assert replace_capture_time_text(original, "2025-09-22T15:05:00") == expected
    path = tmp_path / "capture.toml"
    path.write_text(original, encoding="utf-8")
    write_capture_time(path, "2025-09-22T15:05:00")
    assert path.read_text(encoding="utf-8") == expected
    write_capture_time(path, "2025-09-22T15:05:00")
    assert path.read_text(encoding="utf-8") == expected

    with pytest.raises(ValueError, match=r"\[capture\]"):
        replace_capture_time_text("[other]\nvalue = 1\n", "2025-09-22T15:05:00")
    with pytest.raises(ValueError, match="exactly one"):
        replace_capture_time_text("[capture]\nvalue = 1\n", "2025-09-22T15:05:00")


def test_synthetic_cli_dry_run_and_manifest_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, target_time = _write_synthetic_capture(tmp_path)
    original = manifest.read_text(encoding="utf-8")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    dry_output = tmp_path / "dry-output"
    base_arguments = [
        "--config",
        str(manifest),
        "--search-half-window-hours",
        "0.05",
        "--coarse-step-seconds",
        "60",
        "--aperture-radius-pixels",
        "2",
        "--minimum-saturation-fraction",
        "0.5",
        "--skip-fine-refinement",
    ]
    assert main([*base_arguments, "--output-dir", str(dry_output)]) == 0
    assert manifest.read_text(encoding="utf-8") == original
    dry_report = json.loads((dry_output / "sun_time_calibration.json").read_text(encoding="utf-8"))
    assert dry_report["selected_minute"]["time_utc"] == target_time
    assert not dry_report["config_updated"]

    write_output = tmp_path / "write-output"
    assert (
        main(
            [
                *base_arguments,
                "--output-dir",
                str(write_output),
                "--write-config",
            ]
        )
        == 0
    )
    updated = manifest.read_text(encoding="utf-8")
    assert f'time_utc = "{target_time}"' in updated
    assert "# Preserve this comment and all unrelated values." in updated
    assert "dop_scale = 0.7" in updated
    assert load_config(manifest).time_utc == target_time
    assert {path.name for path in write_output.iterdir()} == {
        "sun_time_search.csv",
        "sun_time_calibration.json",
        "sun_time_calibration.png",
    }
    write_report = json.loads(
        (write_output / "sun_time_calibration.json").read_text(encoding="utf-8")
    )
    assert write_report["config_updated"]
    assert write_report["manifest_time_written_utc"] == target_time


def test_checksum_and_argument_failures(tmp_path: Path) -> None:
    manifest, _ = _write_synthetic_capture(tmp_path)
    config = load_config(manifest)
    (config.input_dir / config.raw_file).write_bytes(b"altered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        main(
            [
                "--config",
                str(manifest),
                "--output-dir",
                str(tmp_path / "output"),
                "--skip-fine-refinement",
            ]
        )

    with pytest.raises(SystemExit):
        parse_arguments(["--aperture-radius-pixels", "0"])
    with pytest.raises(SystemExit):
        parse_arguments(["--minimum-saturation-fraction", "1.1"])
    with pytest.raises(SystemExit):
        parse_arguments(["--search-half-window-hours", "nan"])
