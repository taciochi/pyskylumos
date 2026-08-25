"""Tests for the optional, reproducible JPL solar ephemeris path."""

import os
import warnings
from importlib import import_module
from urllib.error import URLError

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern
from pyskylumos.sky_models import NeutralPointRangeWarning, PanFidelityWarning, Rayleigh

SKY_SIMULATOR_MODULE = import_module("pyskylumos.sky_models.SkySimulator")
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
AZIMUTHS = np.array([[0.0, 90.0, 180.0], [45.0, 135.0, 225.0]])
ALTITUDES = np.array([[10.0, 30.0, 60.0], [20.0, 45.0, 75.0]])
WIRE_GRID = {
    0: SlicingPattern(start_row=0, start_column=0, step=2),
    45: SlicingPattern(start_row=0, start_column=1, step=2),
    90: SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}


def make_rayleigh(times):
    """Return a small Rayleigh simulator suitable for ephemeris tests."""
    return Rayleigh(
        times=times,
        observation_location=LOCATION,
        altitudes=ALTITUDES,
        azimuths=AZIMUTHS,
    )


def make_engine():
    """Return an engine with the standard 2x2 micro-polarizer mosaic."""
    return Engine(
        sensor_pixel_pitch_micrometers=2.2,
        lens_conjugation_type="thin",
        number_pixels_vertical=8,
        number_pixels_horizontal=8,
        lens_focal_length_micrometers=3500,
        polarizer_tolerance_radians=0.05,
        extinction_ratio=0.99,
        auto_exposure_saturation_fraction=0.9,
        adc_resolution_bits=12,
        multiplicative_noise_snr=50,
        wire_grid_orientations_slicing=WIRE_GRID,
    )


def install_fake_get_body(monkeypatch, calls):
    """Install a deterministic AltAz body lookup and record its arguments."""

    def fake_get_body(name, obstime, ephemeris):
        calls.append((name, obstime, ephemeris))
        shape = obstime.shape
        return SkyCoord(
            az=np.full(shape, 137.0) * deg,
            alt=np.full(shape, 33.0) * deg,
            frame=AltAz(obstime=obstime, location=LOCATION),
        )

    monkeypatch.setattr(SKY_SIMULATOR_MODULE, "get_body", fake_get_body)


@pytest.mark.parametrize(
    "times, expected_length",
    [
        (Time("2026-06-21T12:00:00"), 1),
        (Time(["2026-06-21T12:00:00", "2026-06-21T13:00:00"]), 2),
    ],
)
def test_accuracy_true_requests_de430_and_preserves_time_shape(monkeypatch, times, expected_length):
    calls = []
    install_fake_get_body(monkeypatch, calls)

    sun = make_rayleigh(times)._get_sun(accuracy=True)

    assert sun.shape == (expected_length, 1, 1)
    assert len(calls) == 1
    name, requested_times, ephemeris = calls[0]
    assert name == "sun"
    assert ephemeris == "de430"
    np.testing.assert_allclose(requested_times.jd, np.atleast_1d(times.jd))


@pytest.mark.parametrize("sky_model", ["RAYLEIGH", "BERRY", "PAN", "QUEEN"])
def test_engine_forwards_accuracy_true_for_every_model(monkeypatch, sky_model):
    calls = []
    install_fake_get_body(monkeypatch, calls)
    model_options = {"out_of_range": "ignore"} if sky_model in {"PAN", "QUEEN"} else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        values, _ = make_engine().simulate_sky_polarization(
            sky_model=sky_model,
            observation_location=LOCATION,
            times=Time(["2026-06-21T12:00:00"]),
            cie_sky_type=4,
            altitudes=ALTITUDES,
            azimuths=AZIMUTHS,
            accuracy=True,
            model_options=model_options,
        )

    assert values[0].shape == (1, *ALTITUDES.shape)
    assert [(call[0], call[2]) for call in calls] == [("sun", "de430")]


def test_accuracy_false_uses_builtin_ephemeris_only(monkeypatch):
    times = Time(["2026-06-21T12:00:00"])
    calls = []

    def fail_get_body(*args, **kwargs):
        raise AssertionError("accuracy=False must not enter the JPL path")

    def fake_get_sun(obstime):
        calls.append(obstime)
        return SkyCoord(
            az=np.full(obstime.shape, 137.0) * deg,
            alt=np.full(obstime.shape, 33.0) * deg,
            frame=AltAz(obstime=obstime, location=LOCATION),
        )

    monkeypatch.setattr(SKY_SIMULATOR_MODULE, "get_body", fail_get_body)
    monkeypatch.setattr(SKY_SIMULATOR_MODULE, "get_sun", fake_get_sun)

    sun = make_rayleigh(times)._get_sun(accuracy=False)

    assert sun.shape == (1, 1, 1)
    assert len(calls) == 1


def test_explicit_sun_position_bypasses_all_ephemeris_lookups(monkeypatch):
    times = Time(["2026-06-21T12:00:00"])
    explicit_sun = SkyCoord(
        az=[137.0] * deg,
        alt=[33.0] * deg,
        frame=AltAz(obstime=times, location=LOCATION),
    )

    def fail_lookup(*args, **kwargs):
        raise AssertionError("an explicit sun_position must bypass ephemeris lookup")

    monkeypatch.setattr(SKY_SIMULATOR_MODULE, "get_body", fail_lookup)
    monkeypatch.setattr(SKY_SIMULATOR_MODULE, "get_sun", fail_lookup)

    sun = make_rayleigh(times)._get_sun(accuracy=True, sun_position=explicit_sun)

    assert sun.shape == (1, 1, 1)
    np.testing.assert_allclose(sun.az.deg[:, 0, 0], [137.0])
    np.testing.assert_allclose(sun.alt.deg[:, 0, 0], [33.0])


def test_missing_jplephem_has_targeted_installation_error(monkeypatch):
    original_error = ModuleNotFoundError("Solar system JPL ephemeris calculations require jplephem")

    def missing_dependency(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(SKY_SIMULATOR_MODULE, "get_body", missing_dependency)

    with pytest.raises(ModuleNotFoundError, match=r"pyskylumos\[jpl\]") as caught:
        make_rayleigh(Time(["2026-06-21T12:00:00"]))._get_sun(accuracy=True)

    assert caught.value.__cause__ is original_error


@pytest.mark.parametrize("original_error", [URLError("offline"), TimeoutError("timed out")])
def test_unavailable_de430_has_network_and_cache_guidance(monkeypatch, original_error):
    def unavailable_kernel(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(SKY_SIMULATOR_MODULE, "get_body", unavailable_kernel)

    with pytest.raises(RuntimeError, match=r"DE430.*network access.*cache") as caught:
        make_rayleigh(Time(["2026-06-21T12:00:00"]))._get_sun(accuracy=True)

    assert caught.value.__cause__ is original_error


@pytest.mark.remote_data
@pytest.mark.skipif(
    os.getenv("PYSKYLUMOS_RUN_REMOTE_JPL") != "1",
    reason="set PYSKYLUMOS_RUN_REMOTE_JPL=1 to permit a real DE430 lookup",
)
def test_real_de430_lookup():
    pytest.importorskip("jplephem")
    times = Time(["2026-06-21T12:00:00", "2026-06-21T13:00:00"])

    sun = make_rayleigh(times)._get_sun(accuracy=True)

    assert sun.shape == (2, 1, 1)
    assert np.all(np.isfinite(sun.az.deg))
    assert np.all(np.isfinite(sun.alt.deg))
