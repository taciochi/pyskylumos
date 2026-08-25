"""Tilt-aware AOP transport tests using independent geometric oracles."""

import warnings

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.engine import Engine
from pyskylumos.engine._geometry import (
    directions_from_degrees,
    sensor_tilt_rotation,
    transport_stereographic_aop_to_sensor,
)
from pyskylumos.sensor import SlicingPattern
from pyskylumos.sky_models import NeutralPointRangeWarning, Pan, PanFidelityWarning

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
SUN = SkyCoord(az=[137.0] * deg, alt=[33.0] * deg, frame=AltAz(obstime=TIMES, location=LOCATION))

WIRE_GRID = {
    0: SlicingPattern(0, 0, 2),
    45: SlicingPattern(0, 1, 2),
    90: SlicingPattern(1, 0, 2),
    135: SlicingPattern(1, 1, 2),
}


@pytest.fixture
def engine():
    """Return a deterministic engine for tilt-aware simulations."""
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
        random_seed=7,
    )


def axial_residual(actual, expected):
    """Return the signed shortest difference between axial angles."""
    return 0.5 * np.arctan2(
        np.sin(2.0 * (actual - expected)),
        np.cos(2.0 * (actual - expected)),
    )


def rodrigues_rotation_oracle(azimuthal_tilt, tilt_angle):
    """Construct the sensor-to-world rotation independently with Rodrigues' formula."""
    axis = np.array([np.cos(azimuthal_tilt), -np.sin(azimuthal_tilt), 0.0])
    cross_matrix = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    identity = np.eye(3)
    return (
        identity * np.cos(tilt_angle)
        + (1.0 - np.cos(tilt_angle)) * np.outer(axis, axis)
        + np.sin(tilt_angle) * cross_matrix
    )


def directions_to_degrees_oracle(directions):
    """Return azimuth and altitude from independent Cartesian vectors."""
    azimuths = np.rad2deg(np.arctan2(directions[..., 1], directions[..., 0]))
    altitudes = np.rad2deg(
        np.arctan2(
            directions[..., 2],
            np.hypot(directions[..., 0], directions[..., 1]),
        )
    )
    return azimuths, altitudes


def inverse_stereographic(u, v):
    """Map north-pole stereographic coordinates to unit directions."""
    denominator = 1.0 + u**2 + v**2
    return np.stack(
        (
            2.0 * u / denominator,
            2.0 * v / denominator,
            (1.0 - u**2 - v**2) / denominator,
        ),
        axis=-1,
    )


def numerical_stereographic_basis(directions):
    """Return chart tangent vectors using finite differences, not production formulas."""
    u = directions[..., 0] / (1.0 + directions[..., 2])
    v = directions[..., 1] / (1.0 + directions[..., 2])
    step = 1e-6
    basis_u = (inverse_stereographic(u + step, v) - inverse_stereographic(u - step, v)) / (
        2.0 * step
    )
    basis_v = (inverse_stereographic(u, v + step) - inverse_stereographic(u, v - step)) / (
        2.0 * step
    )
    return basis_u, basis_v


def transport_aop_oracle(world_aop, world_directions, sensor_directions, rotation):
    """Transport AOP using numerical chart bases and independent rotation algebra."""
    world_u, world_v = numerical_stereographic_basis(world_directions)
    sensor_u, sensor_v = numerical_stereographic_basis(sensor_directions)
    world_u = world_u[np.newaxis, ...]
    world_v = world_v[np.newaxis, ...]
    sensor_u = sensor_u[np.newaxis, ...]
    sensor_v = sensor_v[np.newaxis, ...]
    tangent_world = (
        np.cos(world_aop)[..., np.newaxis] * world_u + np.sin(world_aop)[..., np.newaxis] * world_v
    )
    tangent_sensor = np.einsum("ij,...j->...i", rotation.T, tangent_world)
    component_u = np.sum(tangent_sensor * sensor_u, axis=-1)
    component_v = np.sum(tangent_sensor * sensor_v, axis=-1)
    sensor_aop = np.arctan2(component_v, component_u)
    return (sensor_aop + np.pi / 2.0) % np.pi - np.pi / 2.0


def simulate(engine, model, azimuths, altitudes, **kwargs):
    """Run an Engine simulation while suppressing model-domain advisories."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        return engine.simulate_sky_polarization(
            sky_model=model,
            observation_location=LOCATION,
            times=TIMES,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=SUN,
            **kwargs,
        )


@pytest.mark.parametrize("model", ["RAYLEIGH", "BERRY", "PAN", "QUEEN"])
@pytest.mark.parametrize("azimuthal_tilt", [0.0, 0.83])
def test_explicit_zero_tilt_is_exactly_the_legacy_path(engine, model, azimuthal_tilt):
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    legacy, legacy_names = simulate(engine, model, azimuths, altitudes)
    explicit, explicit_names = simulate(
        engine,
        model,
        azimuths,
        altitudes,
        sensor_azimuthal_tilt_radians=azimuthal_tilt,
        sensor_tilt_angle_radians=0.0,
    )

    assert explicit_names == legacy_names
    for actual, expected in zip(explicit, legacy, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("model", ["RAYLEIGH", "BERRY", "PAN", "QUEEN"])
def test_tilted_engine_matches_independent_direction_and_aop_oracles(engine, model):
    sensor_azimuths = np.array([[-120.0, -25.0, 35.0], [80.0, 135.0, 175.0]])
    sensor_altitudes = np.array([[12.0, 38.0, 72.0], [18.0, 48.0, 82.0]])
    azimuthal_tilt = 0.63
    tilt_angle = -0.31
    rotation = rodrigues_rotation_oracle(azimuthal_tilt, tilt_angle)
    sensor_directions = directions_from_degrees(sensor_azimuths, sensor_altitudes)
    world_directions = np.einsum("ij,...j->...i", rotation, sensor_directions)
    world_azimuths, world_altitudes = directions_to_degrees_oracle(world_directions)

    actual, names = simulate(
        engine,
        model,
        sensor_azimuths,
        sensor_altitudes,
        sensor_azimuthal_tilt_radians=azimuthal_tilt,
        sensor_tilt_angle_radians=tilt_angle,
    )
    simulator = Engine._Engine__get_sky_simulator(
        times=TIMES,
        sky_model=model,
        azimuths=world_azimuths,
        altitudes=world_altitudes,
        observation_location=LOCATION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        direct = simulator.simulate_sky(cie_sky_type=4, sun_position=SUN)
    world_aop = direct[1]
    if isinstance(simulator, Pan):
        world_aop = (world_aop + np.deg2rad(world_azimuths) + np.pi / 2.0) % np.pi - np.pi / 2.0
    expected_aop = transport_aop_oracle(
        world_aop,
        world_directions,
        sensor_directions,
        rotation,
    )

    assert names == simulator.parameters_simulated
    np.testing.assert_allclose(actual[0], direct[0], atol=2e-12, equal_nan=True)
    np.testing.assert_allclose(actual[2], direct[2], atol=2e-12, equal_nan=True)
    valid = np.isfinite(actual[1]) & np.isfinite(expected_aop)
    assert valid.any()
    assert np.max(np.abs(axial_residual(actual[1][valid], expected_aop[valid]))) < 2e-9


def test_aop_basis_correction_is_spatially_varying_and_orientation_independent():
    sensor_azimuths = np.array([[-130.0, -40.0], [55.0, 145.0]])
    sensor_altitudes = np.array([[10.0, 35.0], [60.0, 82.0]])
    sensor_directions = directions_from_degrees(sensor_azimuths, sensor_altitudes)
    rotation = sensor_tilt_rotation(0.48, 0.37)
    world_directions = np.einsum("ij,...j->...i", rotation, sensor_directions)
    first = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    second = np.array([[[0.31, -0.72], [1.04, -1.2]]])

    first_sensor = transport_stereographic_aop_to_sensor(
        first, world_directions, sensor_directions, rotation
    )
    second_sensor = transport_stereographic_aop_to_sensor(
        second, world_directions, sensor_directions, rotation
    )
    first_correction = axial_residual(first_sensor, first)
    second_correction = axial_residual(second_sensor, second)

    assert np.ptp(first_correction) > np.deg2rad(1.0)
    np.testing.assert_allclose(second_correction, first_correction, atol=2e-15)


def test_aop_transport_followed_by_inverse_pose_recovers_axial_angles():
    sensor_azimuths = np.array([[-110.0, -10.0], [70.0, 155.0]])
    sensor_altitudes = np.array([[15.0, 45.0], [65.0, 80.0]])
    sensor_directions = directions_from_degrees(sensor_azimuths, sensor_altitudes)
    rotation = sensor_tilt_rotation(-0.71, 0.43)
    world_directions = np.einsum("ij,...j->...i", rotation, sensor_directions)
    world_aop = np.array([[[0.12, -0.8], [1.1, -1.45]]])

    sensor_aop = transport_stereographic_aop_to_sensor(
        world_aop, world_directions, sensor_directions, rotation
    )
    recovered = transport_stereographic_aop_to_sensor(
        sensor_aop,
        sensor_directions,
        world_directions,
        rotation.T,
    )

    np.testing.assert_allclose(axial_residual(recovered, world_aop), 0.0, atol=2e-15)


def test_aop_transport_is_stable_at_zenith_and_masks_chart_nadir_without_warnings():
    identity = np.eye(3)
    directions = np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    aop = np.array([[[0.37, -0.52]]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transported = transport_stereographic_aop_to_sensor(aop, directions, directions, identity)

    assert caught == []
    assert transported[0, 0, 0] == pytest.approx(0.37)
    assert np.isnan(transported[0, 0, 1])


def test_tilted_simulation_preserves_inputs_and_accepts_float32(engine):
    azimuths = np.array([[0.0, 45.0], [90.0, 135.0]], dtype=np.float32)
    altitudes = np.array([[15.0, 35.0], [55.0, 75.0]], dtype=np.float32)
    original_azimuths = azimuths.copy()
    original_altitudes = altitudes.copy()

    values, _ = simulate(
        engine,
        "RAYLEIGH",
        azimuths,
        altitudes,
        sensor_azimuthal_tilt_radians=np.float32(0.4),
        sensor_tilt_angle_radians=np.float32(-0.2),
    )

    np.testing.assert_array_equal(azimuths, original_azimuths)
    np.testing.assert_array_equal(altitudes, original_altitudes)
    assert all(value.dtype == np.float64 for value in values)


def test_altitude_mask_is_applied_to_rotated_world_directions(engine):
    azimuths = np.array([[90.0, 0.0], [180.0, -90.0]])
    altitudes = np.full((2, 2), 5.0)
    tilt_angle = np.deg2rad(-20.0)
    _, world_altitudes = engine.tilt_sensor(
        azimuths,
        altitudes,
        azimuthal_tilt=0.0,
        tilt_angle=tilt_angle,
    )

    values, _ = simulate(
        engine,
        "RAYLEIGH",
        azimuths,
        altitudes,
        altitude_min_clip=0.0,
        sensor_azimuthal_tilt_radians=0.0,
        sensor_tilt_angle_radians=tilt_angle,
    )

    expected_mask = world_altitudes <= 0.0
    assert expected_mask.any()
    assert (~expected_mask).any()
    for field in values[:4]:
        np.testing.assert_array_equal(np.isnan(field[0]), expected_mask)


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        (
            {"sensor_azimuthal_tilt_radians": 0.2},
            ValueError,
            "must be supplied together",
        ),
        (
            {"sensor_tilt_angle_radians": 0.2},
            ValueError,
            "must be supplied together",
        ),
        (
            {
                "sensor_azimuthal_tilt_radians": True,
                "sensor_tilt_angle_radians": 0.2,
            },
            TypeError,
            "sensor_azimuthal_tilt_radians",
        ),
        (
            {
                "sensor_azimuthal_tilt_radians": 0.2,
                "sensor_tilt_angle_radians": "tilted",
            },
            TypeError,
            "sensor_tilt_angle_radians",
        ),
        (
            {
                "sensor_azimuthal_tilt_radians": np.nan,
                "sensor_tilt_angle_radians": 0.2,
            },
            ValueError,
            "sensor_azimuthal_tilt_radians must be finite",
        ),
        (
            {
                "sensor_azimuthal_tilt_radians": 0.2,
                "sensor_tilt_angle_radians": np.inf,
            },
            ValueError,
            "sensor_tilt_angle_radians must be finite",
        ),
    ],
)
def test_integrated_tilt_arguments_are_validated(engine, kwargs, error_type, message):
    azimuths = np.zeros((2, 2))
    altitudes = np.full((2, 2), 45.0)

    with pytest.raises(error_type, match=message):
        simulate(engine, "RAYLEIGH", azimuths, altitudes, **kwargs)


def test_azimuth_rotation_is_applied_after_spatial_tilt_correction(engine):
    azimuths = np.array([[-100.0, -20.0], [65.0, 150.0]])
    altitudes = np.array([[15.0, 40.0], [65.0, 80.0]])
    tilt = {
        "sensor_azimuthal_tilt_radians": 0.52,
        "sensor_tilt_angle_radians": -0.28,
    }
    base, _ = simulate(engine, "PAN", azimuths, altitudes, **tilt)
    rotated, _ = simulate(
        engine,
        "PAN",
        azimuths,
        altitudes,
        azimuth_rotation_angle=25.0,
        **tilt,
    )

    expected = (base[1] - np.deg2rad(25.0) + np.pi / 2.0) % np.pi - np.pi / 2.0
    np.testing.assert_allclose(rotated[1], expected, atol=2e-15, equal_nan=True)


@pytest.mark.parametrize("model", ["RAYLEIGH", "BERRY", "PAN", "QUEEN"])
def test_every_tilted_model_runs_through_measurement(engine, model):
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    values, names = simulate(
        engine,
        model,
        azimuths,
        altitudes,
        altitude_min_clip=0,
        sensor_azimuthal_tilt_radians=0.37,
        sensor_tilt_angle_radians=0.22,
    )
    sky = dict(zip(names, values, strict=True))
    measurement = engine.simulate_measurement(
        sky["degree of polarization"],
        sky["angle of polarization"],
        sky["radiance"],
    )

    assert np.isfinite(measurement["dop"]).any()
    assert np.isfinite(measurement["aop"]).any()
