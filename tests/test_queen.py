"""Tests for the QuEEN sky polarization model."""

import warnings

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.sky_models.NeutralPointOffsets import (
    NeutralPointRangeWarning,
    babinet_offset_deg,
    brewster_offset_deg,
)
from pyskylumos.sky_models.QuEEN import QuEEN

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)

DOP, AOP, RADIANCE, SCATTERING = 0, 1, 2, 3
SUN_AZ, SUN_ALT = 4, 5
ABOVE_SUN_AZ, ABOVE_SUN_ALT = 6, 7
BELOW_SUN_AZ, BELOW_SUN_ALT = 8, 9
ANTI_SUN_AZ, ANTI_SUN_ALT = 10, 11
ABOVE_ANTI_AZ, ABOVE_ANTI_ALT = 12, 13
BELOW_ANTI_AZ, BELOW_ANTI_ALT = 14, 15


def make_sun(altitude_deg, azimuth_deg=137.0, times=TIMES):
    """Return an explicit sun position in the simulator's AltAz frame."""
    return SkyCoord(
        az=[azimuth_deg] * len(times) * deg,
        alt=[altitude_deg] * len(times) * deg,
        frame=AltAz(obstime=times, location=LOCATION),
    )


def make_grid(rows=13, columns=19):
    """Return a modest azimuth/altitude sampling grid."""
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns, dtype=np.float32), (rows, 1))
    altitudes = np.tile(np.linspace(2.0, 88.0, rows, dtype=np.float32)[:, None], (1, columns))
    return azimuths, altitudes


def simulate(sun_altitude_deg, sun_azimuth_deg=137.0, azimuths=None, altitudes=None, **kwargs):
    """Run QuEEN for an explicit sun position, suppressing range warnings."""
    if azimuths is None or altitudes is None:
        azimuths, altitudes = make_grid()
    model = QuEEN(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        return model.simulate_sky(
            cie_sky_type=4,
            sun_position=make_sun(sun_altitude_deg, sun_azimuth_deg),
        )


def separation_deg(azimuth_a_rad, altitude_a_rad, azimuth_b_rad, altitude_b_rad):
    """Return the great-circle separation between two AltAz directions, in degrees."""
    return np.rad2deg(
        np.arccos(
            np.clip(
                np.sin(altitude_a_rad) * np.sin(altitude_b_rad)
                + np.cos(altitude_a_rad)
                * np.cos(altitude_b_rad)
                * np.cos(azimuth_a_rad - azimuth_b_rad),
                -1.0,
                1.0,
            )
        )
    )


def wrap_to_half_pi(angle):
    """Wrap an axial angle onto (-pi/2, pi/2]."""
    return (angle + np.pi / 2) % np.pi - np.pi / 2


# --------------------------------------------------------------------------- #
# Q1 -- the four roots are zeros of the field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 20.0, 33.0, 50.0, 60.0])
def test_all_four_roots_have_vanishing_degree_of_polarization(sun_altitude_deg):
    result = simulate(sun_altitude_deg)

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

    at_neutral_points = simulate(
        sun_altitude_deg,
        azimuths=neutral_azimuths[None, :],
        altitudes=neutral_altitudes[None, :],
    )

    np.testing.assert_allclose(at_neutral_points[DOP], 0.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# Q2 -- anti-solar roots are reciprocal-conjugate antipodes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 33.0, 60.0])
def test_anti_solar_roots_are_antipodes_of_the_solar_roots(sun_altitude_deg):
    result = simulate(sun_altitude_deg)

    brewster_to_arago = separation_deg(
        result[BELOW_SUN_AZ], result[BELOW_SUN_ALT], result[ABOVE_ANTI_AZ], result[ABOVE_ANTI_ALT]
    )
    babinet_to_fourth = separation_deg(
        result[ABOVE_SUN_AZ], result[ABOVE_SUN_ALT], result[BELOW_ANTI_AZ], result[BELOW_ANTI_ALT]
    )

    np.testing.assert_allclose(brewster_to_arago, 180.0, atol=1e-9)
    np.testing.assert_allclose(babinet_to_fourth, 180.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Q3-Q5 -- the neutral points sit at the intended angular offsets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [0.0, 15.0, 27.0, 30.0, 45.0, 64.0])
def test_neutral_points_lie_at_pans_fitted_distances(sun_altitude_deg):
    result = simulate(sun_altitude_deg)
    expected_brewster = brewster_offset_deg(sun_altitude_deg)
    expected_babinet = babinet_offset_deg(sun_altitude_deg)

    np.testing.assert_allclose(
        separation_deg(
            result[SUN_AZ], result[SUN_ALT], result[ABOVE_SUN_AZ], result[ABOVE_SUN_ALT]
        ),
        expected_babinet,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        separation_deg(
            result[SUN_AZ], result[SUN_ALT], result[BELOW_SUN_AZ], result[BELOW_SUN_ALT]
        ),
        expected_brewster,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        separation_deg(
            result[ANTI_SUN_AZ], result[ANTI_SUN_ALT], result[ABOVE_ANTI_AZ], result[ABOVE_ANTI_ALT]
        ),
        expected_brewster,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        separation_deg(
            result[ANTI_SUN_AZ], result[ANTI_SUN_ALT], result[BELOW_ANTI_AZ], result[BELOW_ANTI_ALT]
        ),
        expected_babinet,
        atol=1e-9,
    )


def test_babinet_is_above_and_brewster_below_the_sun():
    result = simulate(33.0)

    assert result[ABOVE_SUN_ALT] > result[SUN_ALT]
    assert result[BELOW_SUN_ALT] < result[SUN_ALT]
    assert result[ABOVE_ANTI_ALT] > result[ANTI_SUN_ALT]
    assert result[BELOW_ANTI_ALT] < result[ANTI_SUN_ALT]


# --------------------------------------------------------------------------- #
# Q6-Q7 -- metadata provenance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 33.0, 60.0])
def test_metadata_matches_an_independent_astropy_construction(sun_altitude_deg):
    from astropy.units import rad

    result = simulate(sun_altitude_deg)
    sun = make_sun(sun_altitude_deg)
    anti_sun = sun.directional_offset_by(position_angle=0 * deg, separation=180 * deg)
    brewster_offset = np.deg2rad(brewster_offset_deg(sun_altitude_deg))
    babinet_offset = np.deg2rad(babinet_offset_deg(sun_altitude_deg))

    expected = {
        (ABOVE_SUN_AZ, ABOVE_SUN_ALT): sun.directional_offset_by(0 * deg, babinet_offset * rad),
        (BELOW_SUN_AZ, BELOW_SUN_ALT): sun.directional_offset_by(0 * deg, -brewster_offset * rad),
        (ABOVE_ANTI_AZ, ABOVE_ANTI_ALT): anti_sun.directional_offset_by(
            0 * deg, brewster_offset * rad
        ),
        (BELOW_ANTI_AZ, BELOW_ANTI_ALT): anti_sun.directional_offset_by(
            0 * deg, -babinet_offset * rad
        ),
    }

    for (azimuth_index, altitude_index), point in expected.items():
        np.testing.assert_allclose(
            np.rad2deg(result[azimuth_index]).ravel(), point.az.deg, atol=1e-6
        )
        np.testing.assert_allclose(
            np.rad2deg(result[altitude_index]).ravel(), point.alt.deg, atol=1e-6
        )


def test_metadata_azimuths_use_the_same_wrap_as_astropy():
    result = simulate(33.0, sun_azimuth_deg=350.0)

    for index in (SUN_AZ, ANTI_SUN_AZ, ABOVE_SUN_AZ, BELOW_SUN_AZ, ABOVE_ANTI_AZ, BELOW_ANTI_AZ):
        assert np.all(result[index] >= 0.0)
        assert np.all(result[index] < 2 * np.pi)


# --------------------------------------------------------------------------- #
# Q8-Q9 -- rotation covariance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rotation_deg", [37.0, 123.0, 270.0])
def test_rotating_the_scene_rotates_aop_and_leaves_dop_unchanged(rotation_deg):
    # float64 grids: a float32 azimuth grid rounds at ~5e-7 rad, which would
    # dominate the residual and hide the property under test.
    rows, columns = 9, 15
    relative_azimuths = np.tile(np.linspace(-170.0, 170.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(5.0, 85.0, rows)[:, None], (1, columns))

    base = simulate(33.0, 100.0, azimuths=relative_azimuths + 100.0, altitudes=altitudes)
    rotated = simulate(
        33.0,
        100.0 + rotation_deg,
        azimuths=relative_azimuths + 100.0 + rotation_deg,
        altitudes=altitudes,
    )

    np.testing.assert_allclose(
        wrap_to_half_pi(rotated[AOP] - base[AOP] - np.deg2rad(rotation_deg)), 0.0, atol=1e-9
    )
    np.testing.assert_allclose(rotated[DOP], base[DOP], atol=1e-12)


# --------------------------------------------------------------------------- #
# Q10 -- axial angle conventions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 33.0, 60.0])
def test_aop_lies_on_the_half_pi_interval_and_is_axial(sun_altitude_deg):
    result = simulate(sun_altitude_deg)
    aop = result[AOP]

    assert np.all(aop > -np.pi / 2 - 1e-12)
    assert np.all(aop <= np.pi / 2 + 1e-12)
    np.testing.assert_allclose(np.cos(2 * aop), np.cos(2 * (aop + np.pi)), atol=1e-12)


# --------------------------------------------------------------------------- #
# Q12-Q14 -- the degree of polarization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [-10.0, 0.0, 15.0, 30.0, 45.0, 64.0, 90.0])
def test_degree_of_polarization_stays_within_bounds(sun_altitude_deg):
    azimuths = np.tile(np.linspace(-180.0, 180.0, 121, dtype=np.float32), (61, 1))
    altitudes = np.tile(np.linspace(0.0, 90.0, 61, dtype=np.float32)[:, None], (1, 121))

    dop = simulate(sun_altitude_deg, azimuths=azimuths, altitudes=altitudes)[DOP]

    assert np.nanmin(dop) >= 0.0
    assert np.nanmax(dop) <= 1.0 + 1e-9


@pytest.mark.parametrize("dop_max", [0.25, 0.7, 1.0])
def test_dop_max_scales_the_degree_of_polarization_linearly(dop_max):
    reference = simulate(33.0)[DOP]
    scaled = simulate(33.0, dop_max=dop_max)[DOP]

    np.testing.assert_allclose(scaled, dop_max * reference, atol=1e-12)


@pytest.mark.parametrize("dop_max", [0.0, -0.1, 1.5, np.nan, np.inf])
def test_invalid_dop_max_is_rejected(dop_max):
    azimuths, altitudes = make_grid()
    with pytest.raises(ValueError, match=r"dop_max must lie in \(0, 1\]"):
        QuEEN(
            times=TIMES,
            observation_location=LOCATION,
            azimuths=azimuths,
            altitudes=altitudes,
            dop_max=dop_max,
        )


def test_non_numeric_dop_max_is_rejected():
    azimuths, altitudes = make_grid()
    with pytest.raises(TypeError, match="dop_max must be a float"):
        QuEEN(
            times=TIMES,
            observation_location=LOCATION,
            azimuths=azimuths,
            altitudes=altitudes,
            dop_max="1.0",
        )


# --------------------------------------------------------------------------- #
# Q15 -- the 27 degree Brewster breakpoint, reproduced as published
# --------------------------------------------------------------------------- #


def test_brewster_breakpoint_shows_the_published_discontinuity():
    below = simulate(27.0 - 1e-6)
    above = simulate(27.0 + 1e-6)

    distance_below = separation_deg(
        below[SUN_AZ], below[SUN_ALT], below[BELOW_SUN_AZ], below[BELOW_SUN_ALT]
    )
    distance_above = separation_deg(
        above[SUN_AZ], above[SUN_ALT], above[BELOW_SUN_AZ], above[BELOW_SUN_ALT]
    )

    np.testing.assert_allclose(distance_below - distance_above, 0.48, atol=1e-5)


def test_fields_stay_bounded_across_the_breakpoint():
    """The 0.48 degree jump perturbs the fields, but only mildly and locally.

    The AOP response is largest close to a neutral point, where the polarization
    direction is undefined and turns quickly, so the maximum is bounded loosely
    while the bulk of the sky is pinned tightly.
    """
    below = simulate(26.999)
    above = simulate(27.001)

    degree_change = np.abs(below[DOP] - above[DOP])
    angle_change = np.rad2deg(np.abs(wrap_to_half_pi(below[AOP] - above[AOP])))

    assert np.nanmax(degree_change) <= 0.01
    assert np.median(angle_change) <= 0.25
    assert np.percentile(angle_change, 99) <= 1.0
    assert np.nanmax(angle_change) <= 6.0


# --------------------------------------------------------------------------- #
# Q17-Q20 -- the out-of-range policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [0.0, 32.0, 64.0])
def test_no_warning_inside_pans_measured_range(sun_altitude_deg):
    azimuths, altitudes = make_grid()
    model = QuEEN(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", NeutralPointRangeWarning)
        model.simulate_sky(cie_sky_type=4, sun_position=make_sun(sun_altitude_deg))


@pytest.mark.parametrize("sun_altitude_deg", [-5.0, 70.0])
def test_warns_outside_pans_measured_range(sun_altitude_deg):
    azimuths, altitudes = make_grid()
    model = QuEEN(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )

    with pytest.warns(NeutralPointRangeWarning, match="outside the range measured"):
        model.simulate_sky(cie_sky_type=4, sun_position=make_sun(sun_altitude_deg))


def test_negative_babinet_offset_warns_and_places_the_point_below_the_sun():
    result = simulate(80.0)

    assert babinet_offset_deg(80.0) < 0.0
    assert result[ABOVE_SUN_ALT] < result[SUN_ALT]

    azimuths, altitudes = make_grid()
    model = QuEEN(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )
    with pytest.warns(NeutralPointRangeWarning) as warning_records:
        model.simulate_sky(cie_sky_type=4, sun_position=make_sun(80.0))
    assert any("negative Babinet offset" in str(record.message) for record in warning_records)


def test_raise_policy_rejects_out_of_range_solar_elevation():
    azimuths, altitudes = make_grid()
    model = QuEEN(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        out_of_range="raise",
    )

    with pytest.raises(ValueError, match="outside the range measured"):
        model.simulate_sky(cie_sky_type=4, sun_position=make_sun(70.0))


def test_ignore_policy_is_silent_out_of_range():
    azimuths, altitudes = make_grid()
    model = QuEEN(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        out_of_range="ignore",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", NeutralPointRangeWarning)
        model.simulate_sky(cie_sky_type=4, sun_position=make_sun(80.0))


def test_unknown_out_of_range_policy_is_rejected():
    azimuths, altitudes = make_grid()
    with pytest.raises(ValueError, match="out_of_range must be one of"):
        QuEEN(
            times=TIMES,
            observation_location=LOCATION,
            azimuths=azimuths,
            altitudes=altitudes,
            out_of_range="clamp",
        )


# --------------------------------------------------------------------------- #
# Q21-Q22 -- shapes, units and horizon masking
# --------------------------------------------------------------------------- #


def test_output_shapes_and_parameter_schema():
    times = Time(["2026-06-21T09:00:00", "2026-06-21T12:00:00", "2026-06-21T15:00:00"])
    azimuths, altitudes = make_grid(rows=7, columns=11)
    model = QuEEN(
        times=times, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )

    result = model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0, times=times))

    assert len(result) == len(model.parameters_simulated) == 16
    for index in (DOP, AOP, RADIANCE, SCATTERING):
        assert result[index].shape == (3, 7, 11)
    for index in range(SUN_AZ, 16):
        assert result[index].shape == (3, 1, 1)


def test_altitude_min_clip_masks_only_below_the_horizon():
    rows, columns = 15, 9
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns, dtype=np.float32), (rows, 1))
    altitudes = np.tile(np.linspace(-20.0, 80.0, rows, dtype=np.float32)[:, None], (1, columns))

    model = QuEEN(
        times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes
    )
    result = model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0), altitude_min_clip=0.0)

    mask = model.sky_map.alt.deg <= 0.0
    for index in (DOP, AOP, RADIANCE, SCATTERING):
        assert np.all(np.isnan(result[index][mask]))
        assert np.all(np.isfinite(result[index][~mask]))


# --------------------------------------------------------------------------- #
# Q24 -- documented horizon behaviour (Berry 2004, section 4)
# --------------------------------------------------------------------------- #


def test_polarization_maximum_falls_on_the_horizon():
    """Berry's Eq. (4.2) puts the polarization maxima on the horizon.

    QuEEN inherits this, and the ``|omega| / (2 - |omega|)`` map does not correct
    it because it depends on ``|omega|`` alone and not on position. Pinned here so
    the behaviour is documented rather than surprising; see README section 5.
    """
    rows, columns = 91, 181
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns, dtype=np.float32), (rows, 1))
    altitudes = np.tile(np.linspace(0.0, 90.0, rows, dtype=np.float32)[:, None], (1, columns))

    result = simulate(30.0, 0.0, azimuths=azimuths, altitudes=altitudes)
    dop = result[DOP][0]

    peak_row, _ = np.unravel_index(np.nanargmax(dop), dop.shape)

    assert altitudes[peak_row, 0] <= 1.0
    assert np.nanmax(dop) >= 0.99
