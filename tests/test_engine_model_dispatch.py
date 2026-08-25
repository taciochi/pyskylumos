"""Tests for sky model dispatch and option forwarding through the Engine."""

import warnings

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern
from pyskylumos.sky_models import (
    AsymmetricQuartic,
    Berry,
    DepolarizedRayleigh,
    NeutralPointRangeWarning,
    Pan,
    PanFidelityWarning,
    QuEEN,
    Rayleigh,
)

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
SUN = SkyCoord(az=[137.0] * deg, alt=[33.0] * deg, frame=AltAz(obstime=TIMES, location=LOCATION))

WIRE_GRID = {
    0: SlicingPattern(start_row=0, start_column=0, step=2),
    45: SlicingPattern(start_row=0, start_column=1, step=2),
    90: SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}


@pytest.fixture
def engine():
    """Return a small engine suitable for dispatch tests."""
    return Engine(
        sensor_pixel_pitch_micrometers=2.2,
        lens_conjugation_type="thin",
        number_pixels_vertical=16,
        number_pixels_horizontal=16,
        lens_focal_length_micrometers=3500,
        polarizer_tolerance_radians=0.0,
        extinction_ratio=0.99,
        auto_exposure_saturation_fraction=0.9,
        adc_resolution_bits=12,
        multiplicative_noise_snr=50,
        wire_grid_orientations_slicing=WIRE_GRID,
    )


def run(engine, sky_model, **kwargs):
    """Run a sky simulation through the engine, suppressing advisory warnings."""
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        return engine.simulate_sky_polarization(
            sky_model=sky_model,
            observation_location=LOCATION,
            times=TIMES,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=SUN,
            **kwargs,
        )


def build_simulator(engine, sky_model, model_options=None):
    """Return the simulator instance the engine would construct."""
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    return Engine._Engine__get_sky_simulator(
        times=TIMES,
        sky_model=sky_model,
        azimuths=azimuths,
        altitudes=altitudes,
        observation_location=LOCATION,
        model_options=model_options,
    )


# --------------------------------------------------------------------------- #
# name resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sky_model, expected",
    [
        ("rayleigh", Rayleigh),
        ("RAYLEIGH", Rayleigh),
        ("Rayleigh", Rayleigh),
        ("depolarized_rayleigh", DepolarizedRayleigh),
        ("DEPOLARIZED_RAYLEIGH", DepolarizedRayleigh),
        ("Depolarized_Rayleigh", DepolarizedRayleigh),
        ("asymmetric", AsymmetricQuartic),
        ("ASYMMETRIC", AsymmetricQuartic),
        ("AsYmMeTrIc", AsymmetricQuartic),
        ("asq", AsymmetricQuartic),
        ("ASQ", AsymmetricQuartic),
        ("berry", Berry),
        ("BERRY", Berry),
        ("pan", Pan),
        ("PAN", Pan),
        ("queen", QuEEN),
        ("QUEEN", QuEEN),
        ("QuEEN", QuEEN),
        ("qen", QuEEN),
        ("QEN", QuEEN),
    ],
)
def test_sky_model_names_resolve_to_the_right_class(engine, sky_model, expected):
    assert isinstance(build_simulator(engine, sky_model), expected)


def test_unknown_sky_model_names_the_alternatives(engine):
    with pytest.raises(
        ValueError,
        match="RAYLEIGH, DEPOLARIZED_RAYLEIGH, ASYMMETRIC, BERRY, PAN, QUEEN",
    ):
        build_simulator(engine, "singularity")


def test_unknown_sky_model_explains_the_pan_rename(engine):
    with pytest.raises(ValueError, match=r"shipped as PAN up to version 0\.0\.6 is now QUEEN"):
        build_simulator(engine, "nope")


def test_every_model_declares_an_aop_reference():
    for model_class in (Rayleigh, DepolarizedRayleigh, AsymmetricQuartic, Berry, Pan, QuEEN):
        assert model_class.AOP_REFERENCE in {"fixed_world", "local_meridian"}


def test_only_pan_is_local_meridian_referenced():
    assert Pan.AOP_REFERENCE == "local_meridian"
    for model_class in (Rayleigh, DepolarizedRayleigh, AsymmetricQuartic, Berry, QuEEN):
        assert model_class.AOP_REFERENCE == "fixed_world"


# --------------------------------------------------------------------------- #
# PAN and QUEEN are distinct through the public API
# --------------------------------------------------------------------------- #


def test_pan_and_queen_produce_different_results_through_the_engine(engine):
    pan_values, names = run(engine, "PAN")
    queen_values, _ = run(engine, "QUEEN")

    pan = dict(zip(names, pan_values, strict=False))
    queen = dict(zip(names, queen_values, strict=False))

    assert np.nanmax(np.abs(pan["degree of polarization"] - queen["degree of polarization"])) > 1e-3
    np.testing.assert_allclose(
        pan["angle of polarization"],
        queen["angle of polarization"],
        atol=1e-12,
        equal_nan=True,
    )


def test_qen_alias_matches_queen(engine):
    alias_values, _ = run(engine, "QEN")
    queen_values, _ = run(engine, "QUEEN")

    for alias, queen in zip(alias_values, queen_values, strict=False):
        np.testing.assert_allclose(alias, queen, atol=1e-12, equal_nan=True)


def test_engine_and_direct_berry_construction_return_the_same_published_field(engine):
    engine_values, _ = run(engine, "BERRY")
    direct = build_simulator(engine, "BERRY").simulate_sky(
        cie_sky_type=4,
        sun_position=SUN,
    )

    np.testing.assert_allclose(engine_values[0], direct[0], rtol=1e-6, atol=1e-8)


def test_engine_converts_direct_pan_aop_to_the_fixed_sensor_frame(engine):
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    direct_model = build_simulator(engine, "PAN")
    with pytest.warns(PanFidelityWarning, match="referenced to the local meridian"):
        direct_values = direct_model.simulate_sky(cie_sky_type=4, sun_position=SUN)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine_values, _ = engine.simulate_sky_polarization(
            sky_model="PAN",
            observation_location=LOCATION,
            times=TIMES,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=SUN,
        )

    expected_aop = (direct_values[1] + np.deg2rad(azimuths) + np.pi / 2) % np.pi - np.pi / 2
    valid = np.isfinite(expected_aop) & (direct_values[0] >= 1e-8)
    axial_residual = 0.5 * np.arctan2(
        np.sin(2 * (engine_values[1] - expected_aop)),
        np.cos(2 * (engine_values[1] - expected_aop)),
    )

    assert np.nanmax(np.abs(axial_residual[valid])) <= 1e-12
    assert not any(isinstance(item.message, PanFidelityWarning) for item in caught)
    for index in range(len(engine_values)):
        if index != 1:
            np.testing.assert_allclose(
                engine_values[index], direct_values[index], atol=1e-12, equal_nan=True
            )


def test_engine_converts_pan_aop_across_times_azimuths_and_elevation_regimes(engine):
    times = Time(
        [
            "2026-06-21T09:00:00",
            "2026-06-21T10:00:00",
            "2026-06-21T11:00:00",
            "2026-06-21T12:00:00",
        ]
    )
    sun = SkyCoord(
        az=np.array([0.0, 37.0, 137.0, 275.0]) * deg,
        alt=np.array([0.0, 27.0 - 1e-7, 27.0 + 1e-7, 64.0]) * deg,
        frame=AltAz(obstime=times, location=LOCATION),
    )
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    direct_model = Pan(
        times=times,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        out_of_range="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        direct_values = direct_model.simulate_sky(cie_sky_type=4, sun_position=sun)

    engine_values, _ = engine.simulate_sky_polarization(
        sky_model="PAN",
        observation_location=LOCATION,
        times=times,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=sun,
        model_options={"out_of_range": "raise"},
    )
    expected = (direct_values[1] + np.deg2rad(azimuths) + np.pi / 2) % np.pi - np.pi / 2
    valid = np.isfinite(expected) & (direct_values[0] >= 1e-8)
    residual = 0.5 * np.arctan2(
        np.sin(2 * (engine_values[1] - expected)),
        np.cos(2 * (engine_values[1] - expected)),
    )

    assert engine_values[1].shape == (4, 16, 16)
    assert np.nanmax(np.abs(residual[valid])) <= 1e-12


def test_pan_azimuth_rotation_is_applied_after_sensor_frame_conversion(engine):
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    direct_model = build_simulator(engine, "PAN")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        direct_values = direct_model.simulate_sky(cie_sky_type=4, sun_position=SUN)

    rotated_values, _ = engine.simulate_sky_polarization(
        sky_model="PAN",
        observation_location=LOCATION,
        times=TIMES,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=SUN,
        azimuth_rotation_angle=25.0,
    )
    expected = (
        direct_values[1] + np.deg2rad(azimuths) - np.deg2rad(25.0) + np.pi / 2
    ) % np.pi - np.pi / 2

    np.testing.assert_allclose(rotated_values[1], expected, atol=1e-12, equal_nan=True)


def test_engine_pan_aop_is_covariant_under_rigid_scene_rotation(engine):
    rotation_deg = 53.0
    azimuths = np.array([[-120.0, -20.0, 40.0], [85.0, 140.0, 175.0]])
    altitudes = np.array([[5.0, 30.0, 70.0], [15.0, 45.0, 85.0]])
    rotated_azimuths = (azimuths + rotation_deg + 180.0) % 360.0 - 180.0
    base_sun = SkyCoord(
        az=[37.0] * deg,
        alt=[33.0] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )
    rotated_sun = SkyCoord(
        az=[37.0 + rotation_deg] * deg,
        alt=[33.0] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )

    base_values, _ = engine.simulate_sky_polarization(
        sky_model="PAN",
        observation_location=LOCATION,
        times=TIMES,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=base_sun,
    )
    rotated_values, _ = engine.simulate_sky_polarization(
        sky_model="PAN",
        observation_location=LOCATION,
        times=TIMES,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=rotated_azimuths,
        sun_position=rotated_sun,
    )
    expected_rotated_aop = (
        base_values[1] + np.deg2rad(rotation_deg) + np.pi / 2
    ) % np.pi - np.pi / 2

    np.testing.assert_allclose(rotated_values[0], base_values[0], atol=1e-12)
    np.testing.assert_allclose(rotated_values[1], expected_rotated_aop, atol=1e-12)


def test_engine_suppresses_only_pan_fidelity_warning(engine):
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    out_of_range_sun = SkyCoord(
        az=[137.0] * deg,
        alt=[70.0] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )

    with pytest.warns(NeutralPointRangeWarning, match="outside the range measured") as caught:
        engine.simulate_sky_polarization(
            sky_model="PAN",
            observation_location=LOCATION,
            times=TIMES,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=out_of_range_sun,
        )

    assert not any(isinstance(item.message, PanFidelityWarning) for item in caught)


# --------------------------------------------------------------------------- #
# model_options
# --------------------------------------------------------------------------- #


def test_dop_max_is_forwarded_to_queen(engine):
    reference_values, names = run(engine, "QUEEN")
    scaled_values, _ = run(engine, "QUEEN", model_options={"dop_max": 0.8})

    reference = dict(zip(names, reference_values, strict=False))["degree of polarization"]
    scaled = dict(zip(names, scaled_values, strict=False))["degree of polarization"]

    np.testing.assert_allclose(scaled, 0.8 * reference, atol=1e-12, equal_nan=True)


def test_out_of_range_is_forwarded_to_pan(engine):
    assert build_simulator(engine, "PAN", {"out_of_range": "ignore"}) is not None


def test_depolarization_ratio_is_forwarded_to_depolarized_rayleigh(engine):
    model = build_simulator(
        engine,
        "DEPOLARIZED_RAYLEIGH",
        {"depolarization_ratio": 0.1},
    )

    assert isinstance(model, DepolarizedRayleigh)
    assert model.depolarization_ratio == 0.1


def test_all_asymmetric_options_are_forwarded(engine):
    model = build_simulator(
        engine,
        "ASYMMETRIC",
        {
            "arago_offset": 31.0,
            "fourth_offset": "babinet",
            "normalisation": "berry",
            "dop_max": 0.8,
            "out_of_range": "ignore",
        },
    )

    assert isinstance(model, AsymmetricQuartic)


@pytest.mark.parametrize("sky_model", ["BERRY", "RAYLEIGH"])
def test_models_without_options_reject_model_options(engine, sky_model):
    with pytest.raises(TypeError, match="does not accept model_options"):
        build_simulator(engine, sky_model, {"dop_max": 0.8})


def test_pan_rejects_dop_max_and_names_what_it_accepts(engine):
    with pytest.raises(TypeError, match=r"does not accept model_options \['dop_max'\]"):
        build_simulator(engine, "PAN", {"dop_max": 0.8})

    with pytest.raises(TypeError, match="out_of_range"):
        build_simulator(engine, "PAN", {"dop_max": 0.8})


def test_model_options_must_be_a_dict(engine):
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    with pytest.raises(TypeError, match="model_options must be a dict or None"):
        engine.simulate_sky_polarization(
            sky_model="QUEEN",
            observation_location=LOCATION,
            times=TIMES,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=SUN,
            model_options=["dop_max", 0.8],
        )


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sky_model", ["RAYLEIGH", "DEPOLARIZED_RAYLEIGH", "ASYMMETRIC", "BERRY", "PAN", "QUEEN"]
)
def test_every_model_produces_a_usable_sky(engine, sky_model):
    values, names = run(engine, sky_model)
    sky = dict(zip(names, values, strict=False))

    for key in ("degree of polarization", "angle of polarization", "radiance"):
        assert sky[key].shape == (1, 16, 16)

    finite = np.isfinite(sky["degree of polarization"])
    assert finite.any()
    assert np.all(sky["degree of polarization"][finite] >= 0.0)
    assert np.all(sky["degree of polarization"][finite] <= 1.0 + 1e-9)
    assert np.all(np.isfinite(sky["radiance"][finite]))


@pytest.mark.parametrize(
    "sky_model", ["RAYLEIGH", "DEPOLARIZED_RAYLEIGH", "ASYMMETRIC", "BERRY", "PAN", "QUEEN"]
)
def test_every_model_runs_end_to_end_through_a_measurement(engine, sky_model):
    values, names = run(engine, sky_model)
    sky = dict(zip(names, values, strict=False))

    measurement = engine.simulate_measurement(
        degree_of_polarization=sky["degree of polarization"],
        angle_of_polarization=sky["angle of polarization"],
        radiance=sky["radiance"],
    )

    assert measurement["dop"].shape == (1, 8, 8)
    assert measurement["aop"].shape == (1, 8, 8)
    assert np.isfinite(measurement["dop"]).any()


@pytest.mark.parametrize("sky_model", ["ASYMMETRIC", "BERRY", "PAN", "QUEEN"])
def test_quartic_models_share_the_same_parameter_schema(engine, sky_model):
    _, names = run(engine, sky_model)

    assert len(names) == 16
    assert names[:4] == (
        "degree of polarization",
        "angle of polarization",
        "radiance",
        "scattering angle",
    )
    assert names[4] == "sun azimuth"
    assert names[5] == "sun elevation"
