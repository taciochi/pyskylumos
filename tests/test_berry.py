"""Tests for the Berry sky polarization model."""

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg, rad

from pyskylumos.sky_models.Berry import Berry

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)

DOP, AOP, RADIANCE, SCATTERING = 0, 1, 2, 3
SUN_AZ, SUN_ALT = 4, 5
ABOVE_SUN_AZ, ABOVE_SUN_ALT = 6, 7
BELOW_SUN_AZ, BELOW_SUN_ALT = 8, 9
ANTI_SUN_AZ, ANTI_SUN_ALT = 10, 11
ABOVE_ANTI_AZ, ABOVE_ANTI_ALT = 12, 13
BELOW_ANTI_AZ, BELOW_ANTI_ALT = 14, 15


def make_sun(altitude_deg, azimuth_deg=137.0):
    """Return an explicit sun position in the simulator's AltAz frame."""
    return SkyCoord(
        az=[azimuth_deg] * deg,
        alt=[altitude_deg] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )


def simulate(sun_altitude_deg, sun_azimuth_deg=137.0, rows=13, columns=19):
    """Run Berry for an explicit sun position."""
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(2.0, 88.0, rows)[:, None], (1, columns))
    model = Berry(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )

    return model.simulate_sky(
        cie_sky_type=4, sun_position=make_sun(sun_altitude_deg, sun_azimuth_deg)
    )


def published_modulus_oracle(azimuths_deg, altitudes_deg, sun_altitude_deg, sun_azimuth_deg):
    """Evaluate Berry Eq. (4.2) without importing production geometry helpers."""
    observed_zenith = np.deg2rad(90.0 - altitudes_deg)
    observed_azimuth = np.deg2rad(azimuths_deg)
    sun_zenith = np.deg2rad(90.0 - sun_altitude_deg)
    sun_azimuth = np.deg2rad(sun_azimuth_deg)
    offset = np.deg2rad(15.0)

    observed = np.tan(observed_zenith / 2) * np.exp(1j * observed_azimuth)
    below_sun = np.tan((sun_zenith + offset) / 2) * np.exp(1j * sun_azimuth)
    above_sun = np.tan((sun_zenith - offset) / 2) * np.exp(1j * sun_azimuth)
    above_anti_sun = -1 / np.conjugate(below_sun)
    below_anti_sun = -1 / np.conjugate(above_sun)

    numerator = -4 * (
        (observed - below_sun)
        * (observed - above_sun)
        * (observed - above_anti_sun)
        * (observed - below_anti_sun)
    )
    denominator = (
        (1 + np.abs(observed) ** 2) ** 2
        * np.abs(below_sun - above_anti_sun)
        * np.abs(above_sun - below_anti_sun)
    )
    return np.abs(numerator / denominator)


# --------------------------------------------------------------------------- #
# B1 -- the sun and anti-sun metadata were transposed before 0.1.0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg,sun_azimuth_deg", [(33.0, 137.0), (5.0, 275.0)])
def test_sun_and_anti_sun_metadata_are_not_transposed(sun_altitude_deg, sun_azimuth_deg):
    result = simulate(sun_altitude_deg, sun_azimuth_deg)
    sun = make_sun(sun_altitude_deg, sun_azimuth_deg)
    anti_sun = sun.directional_offset_by(position_angle=0 * deg, separation=180 * deg)

    np.testing.assert_allclose(np.rad2deg(result[SUN_AZ]).ravel(), sun.az.deg, atol=1e-9)
    np.testing.assert_allclose(np.rad2deg(result[SUN_ALT]).ravel(), sun.alt.deg, atol=1e-9)
    np.testing.assert_allclose(np.rad2deg(result[ANTI_SUN_AZ]).ravel(), anti_sun.az.deg, atol=1e-9)
    np.testing.assert_allclose(
        np.rad2deg(result[ANTI_SUN_ALT]).ravel(), anti_sun.alt.deg, atol=1e-9
    )


def test_metadata_names_line_up_with_returned_values():
    azimuths = np.tile(np.linspace(-180.0, 180.0, 5), (3, 1))
    altitudes = np.tile(np.linspace(10.0, 80.0, 3)[:, None], (1, 5))
    model = Berry(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )
    result = model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0))

    named = dict(zip(model.parameters_simulated, result, strict=False))

    assert np.rad2deg(named["sun elevation"]).ravel()[0] == pytest.approx(33.0, abs=1e-9)
    assert np.rad2deg(named["sun azimuth"]).ravel()[0] == pytest.approx(137.0, abs=1e-9)
    assert np.rad2deg(named["anti-sun elevation"]).ravel()[0] == pytest.approx(-33.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# fixed 15 degree splitting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [0.0, 20.0, 45.0, 70.0])
def test_neutral_points_sit_fifteen_degrees_from_the_sun(sun_altitude_deg):
    result = simulate(sun_altitude_deg)
    sun = make_sun(sun_altitude_deg)
    anti_sun = sun.directional_offset_by(position_angle=0 * deg, separation=180 * deg)
    half_separation = Berry.NEUTRAL_POINT_SEPARATION_DEG / 2

    for azimuth_index, altitude_index, reference in (
        (ABOVE_SUN_AZ, ABOVE_SUN_ALT, sun),
        (BELOW_SUN_AZ, BELOW_SUN_ALT, sun),
        (ABOVE_ANTI_AZ, ABOVE_ANTI_ALT, anti_sun),
        (BELOW_ANTI_AZ, BELOW_ANTI_ALT, anti_sun),
    ):
        point = SkyCoord(
            az=result[azimuth_index].ravel() * rad,
            alt=result[altitude_index].ravel() * rad,
            frame=sun.frame,
        )
        assert reference.separation(point).deg[0] == pytest.approx(half_separation, abs=1e-6)


def test_splitting_does_not_depend_on_solar_elevation():
    low = simulate(5.0)
    high = simulate(65.0)

    low_offset = np.rad2deg(low[ABOVE_SUN_ALT] - low[SUN_ALT])
    high_offset = np.rad2deg(high[ABOVE_SUN_ALT] - high[SUN_ALT])

    np.testing.assert_allclose(low_offset, high_offset, atol=1e-9)
    np.testing.assert_allclose(low_offset, Berry.NEUTRAL_POINT_SEPARATION_DEG / 2, atol=1e-9)


# --------------------------------------------------------------------------- #
# field behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sun_altitude_deg,sun_azimuth_deg",
    [(0.0, 0.0), (33.0, 137.0), (64.0, 275.0)],
)
def test_dop_is_the_published_quartic_field_modulus(sun_altitude_deg, sun_azimuth_deg):
    rows, columns = 17, 29
    azimuths = np.tile(np.linspace(-179.0, 179.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(1.0, 89.0, rows)[:, None], (1, columns))
    model = Berry(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
    )

    actual = model.simulate_sky(
        cie_sky_type=4,
        sun_position=make_sun(sun_altitude_deg, sun_azimuth_deg),
    )[DOP]
    expected = published_modulus_oracle(
        azimuths,
        altitudes,
        sun_altitude_deg,
        sun_azimuth_deg,
    )

    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("sun_altitude_deg", [-10.0, 0.0, 30.0, 60.0, 90.0])
def test_degree_of_polarization_stays_within_bounds(sun_altitude_deg):
    azimuths = np.tile(np.linspace(-180.0, 180.0, 121), (61, 1))
    altitudes = np.tile(np.linspace(0.0, 90.0, 61)[:, None], (1, 121))
    model = Berry(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )

    dop = model.simulate_sky(cie_sky_type=4, sun_position=make_sun(sun_altitude_deg))[DOP]

    assert np.nanmin(dop) >= 0.0
    assert np.nanmax(dop) <= 1.0 + 1e-9
    assert np.nanmax(dop) >= 0.998


def test_neutral_points_have_vanishing_degree_of_polarization():
    result = simulate(33.0)
    neutral_azimuths = np.rad2deg(
        np.array(
            [
                result[i].ravel()[0]
                for i in (ABOVE_SUN_AZ, BELOW_SUN_AZ, ABOVE_ANTI_AZ, BELOW_ANTI_AZ)
            ]
        )
    )
    neutral_altitudes = np.rad2deg(
        np.array(
            [
                result[i].ravel()[0]
                for i in (ABOVE_SUN_ALT, BELOW_SUN_ALT, ABOVE_ANTI_ALT, BELOW_ANTI_ALT)
            ]
        )
    )

    model = Berry(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=neutral_azimuths[None, :],
        altitudes=neutral_altitudes[None, :],
    )
    at_neutral_points = model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0))

    np.testing.assert_allclose(at_neutral_points[DOP], 0.0, atol=1e-10)


@pytest.mark.parametrize("rotation_deg", [37.0, 270.0])
def test_rotation_covariance(rotation_deg):
    rows, columns = 9, 15
    relative_azimuths = np.tile(np.linspace(-170.0, 170.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(5.0, 85.0, rows)[:, None], (1, columns))

    def evaluate(sun_azimuth_deg):
        model = Berry(
            times=TIMES,
            observation_location=LOCATION,
            azimuths=relative_azimuths + sun_azimuth_deg,
            altitudes=altitudes,
        )
        return model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0, sun_azimuth_deg))

    base = evaluate(100.0)
    rotated = evaluate(100.0 + rotation_deg)

    difference = rotated[AOP] - base[AOP] - np.deg2rad(rotation_deg)
    np.testing.assert_allclose((difference + np.pi / 2) % np.pi - np.pi / 2, 0.0, atol=1e-9)
    np.testing.assert_allclose(rotated[DOP], base[DOP], atol=1e-12)


def test_altitude_min_clip_masks_only_below_the_horizon():
    rows, columns = 15, 9
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(-20.0, 80.0, rows)[:, None], (1, columns))
    model = Berry(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )

    result = model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0), altitude_min_clip=0.0)

    mask = model.sky_map.alt.deg <= 0.0
    for index in (DOP, AOP, RADIANCE, SCATTERING):
        assert np.all(np.isnan(result[index][mask]))
        assert np.all(np.isfinite(result[index][~mask]))
