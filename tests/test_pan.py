"""Tests for the published Pan et al. (2023) sky polarization model.

These tests reimplement Pan's equations independently of the shared utilities
wherever practical, so that the model is validated against the paper rather than
against QuEEN. The final section asserts that the two models cannot become
aliases of one another.
"""

import warnings

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg, rad

from pyskylumos.sky_models.NeutralPointOffsets import (
    NeutralPointRangeWarning,
    babinet_offset_deg,
    brewster_offset_deg,
)
from pyskylumos.sky_models.Pan import Pan, PanFidelityWarning
from pyskylumos.sky_models.QuEEN import QuEEN
from pyskylumos.sky_models.Rayleigh import Rayleigh

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
    """Return a modest float64 azimuth/altitude sampling grid."""
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(2.0, 88.0, rows)[:, None], (1, columns))
    return azimuths, altitudes


def simulate(
    model_class, sun_altitude_deg, sun_azimuth_deg=137.0, azimuths=None, altitudes=None, **kwargs
):
    """Run a model for an explicit sun position, suppressing advisory warnings."""
    if azimuths is None or altitudes is None:
        azimuths, altitudes = make_grid()
    model = model_class(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        warnings.simplefilter("ignore", PanFidelityWarning)
        return model.simulate_sky(
            cie_sky_type=4,
            sun_position=make_sun(sun_altitude_deg, sun_azimuth_deg),
        )


def wrap_to_half_pi(angle):
    """Wrap an axial angle onto (-pi/2, pi/2]."""
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def pan_field(sun_altitude_deg, sun_azimuth_deg, point_azimuths_deg, point_altitudes_deg):
    """Independently rebuild Pan Eqs. (11)-(13) with errata E5-E8 applied.

    Written out longhand rather than calling ``StereographicQuartic`` so that the
    model is checked against the paper, not against the shared implementation.
    """
    sun_zenith = np.deg2rad(90.0 - sun_altitude_deg)
    sun_azimuth = np.deg2rad(sun_azimuth_deg)
    point_zenith = np.deg2rad(90.0 - np.asarray(point_altitudes_deg, dtype=float))
    point_azimuth = np.deg2rad(np.asarray(point_azimuths_deg, dtype=float))

    # E5: y_s is the stereographic radius of the sun, tan(phi_s / 2).
    solar_radius = np.tan(sun_zenith / 2)
    brewster_parameter = np.tan(np.deg2rad(brewster_offset_deg(sun_altitude_deg)) / 2)
    babinet_parameter = np.tan(np.deg2rad(babinet_offset_deg(sun_altitude_deg)) / 2)

    observed = np.tan(point_zenith / 2) * np.exp(1j * point_azimuth)
    # E8: Brewster below the sun, Babinet above it.
    brewster = np.exp(1j * sun_azimuth) * (
        (solar_radius + brewster_parameter) / (1 - solar_radius * brewster_parameter)
    )
    # E6: the second root's denominator carries a plus sign.
    babinet = np.exp(1j * sun_azimuth) * (
        (solar_radius - babinet_parameter) / (1 + solar_radius * babinet_parameter)
    )
    # E7: the anti-solar roots are reciprocal conjugates.
    arago = -1 / np.conj(brewster)
    fourth = -1 / np.conj(babinet)

    numerator = (
        -4 * (observed - brewster) * (observed - babinet) * (observed - arago) * (observed - fourth)
    )
    denominator = (
        ((1 + np.abs(observed) ** 2) ** 2) * np.abs(brewster - arago) * np.abs(babinet - fourth)
    )
    return numerator / denominator


# --------------------------------------------------------------------------- #
# P1-P2 -- Pan Eq. (14), DoP = |omega|
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 20.0, 33.0, 50.0, 60.0])
def test_degree_of_polarization_is_the_field_modulus(sun_altitude_deg):
    azimuths, altitudes = make_grid()
    result = simulate(Pan, sun_altitude_deg, azimuths=azimuths, altitudes=altitudes)

    expected = np.abs(pan_field(sun_altitude_deg, 137.0, azimuths, altitudes))

    np.testing.assert_allclose(result[DOP][0], expected, atol=1e-12)


@pytest.mark.parametrize("sun_altitude_deg", [-10.0, 0.0, 15.0, 30.0, 45.0, 64.0, 90.0])
def test_degree_of_polarization_stays_within_bounds(sun_altitude_deg):
    azimuths = np.tile(np.linspace(-180.0, 180.0, 121), (61, 1))
    altitudes = np.tile(np.linspace(0.0, 90.0, 61)[:, None], (1, 121))

    dop = simulate(Pan, sun_altitude_deg, azimuths=azimuths, altitudes=altitudes)[DOP]

    assert np.nanmin(dop) >= 0.0
    assert np.nanmax(dop) <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# P3-P4 -- Pan Eqs. (15), (23) and (24)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 33.0, 60.0])
def test_angle_of_polarization_is_meridian_referenced(sun_altitude_deg):
    azimuths, altitudes = make_grid()
    result = simulate(Pan, sun_altitude_deg, azimuths=azimuths, altitudes=altitudes)

    field = pan_field(sun_altitude_deg, 137.0, azimuths, altitudes)
    expected = wrap_to_half_pi(
        0.5 * np.angle(field * np.exp(-2j * np.deg2rad(137.0))) - np.deg2rad(azimuths)
    )

    np.testing.assert_allclose(result[AOP][0], expected, atol=1e-12)


def test_angle_of_polarization_is_wrapped_per_equation_24():
    result = simulate(Pan, 33.0)

    assert np.all(result[AOP] > -np.pi / 2 - 1e-12)
    assert np.all(result[AOP] <= np.pi / 2 + 1e-12)


# --------------------------------------------------------------------------- #
# P5 -- a meridian-referenced angle is invariant under a rigid scene rotation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rotation_deg", [37.0, 123.0, 270.0])
def test_meridian_referenced_aop_is_rotation_invariant(rotation_deg):
    rows, columns = 9, 15
    relative_azimuths = np.tile(np.linspace(-170.0, 170.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(5.0, 85.0, rows)[:, None], (1, columns))

    base = simulate(Pan, 33.0, 100.0, azimuths=relative_azimuths + 100.0, altitudes=altitudes)
    rotated = simulate(
        Pan,
        33.0,
        100.0 + rotation_deg,
        azimuths=relative_azimuths + 100.0 + rotation_deg,
        altitudes=altitudes,
    )

    np.testing.assert_allclose(wrap_to_half_pi(rotated[AOP] - base[AOP]), 0.0, atol=1e-9)
    np.testing.assert_allclose(rotated[DOP], base[DOP], atol=1e-12)


def test_dropping_the_phase_factor_would_shift_aop_by_the_solar_azimuth():
    """Errata E9 is pinned: the ``e^{-2i alpha_s}`` factor must stay applied."""
    sun_azimuth_deg = 137.0
    azimuths, altitudes = make_grid()
    result = simulate(Pan, 33.0, sun_azimuth_deg, azimuths=azimuths, altitudes=altitudes)

    field = pan_field(33.0, sun_azimuth_deg, azimuths, altitudes)
    as_printed = 0.5 * np.angle(field) - np.deg2rad(azimuths)

    np.testing.assert_allclose(
        wrap_to_half_pi(result[AOP][0] - as_printed + np.deg2rad(sun_azimuth_deg)),
        0.0,
        atol=1e-9,
    )


# --------------------------------------------------------------------------- #
# P6 -- physical validation against Rayleigh in the small-splitting limit
# --------------------------------------------------------------------------- #


def test_matches_the_rayleigh_meridian_angle_in_the_small_splitting_limit(monkeypatch):
    """With the neutral points collapsed onto the sun, Pan must reduce to Rayleigh.

    ``Rayleigh.__get_aop`` returns the frame-referenced angle, so the local
    meridian angle it implies is that value minus the point's own azimuth --
    exactly what Pan Eq. (23) constructs.
    """
    negligible = np.float64(1e-6)
    monkeypatch.setattr(
        Pan,
        "_neutral_point_offsets",
        lambda self, sun_altitudes_deg: (negligible, negligible),
    )

    sun_altitude_deg, sun_azimuth_deg = 30.0, 137.0
    azimuths, altitudes = make_grid()
    result = simulate(
        Pan, sun_altitude_deg, sun_azimuth_deg, azimuths=azimuths, altitudes=altitudes
    )

    rayleigh_frame_aop = Rayleigh._Rayleigh__get_aop(
        observed_point_altitude=np.deg2rad(altitudes),
        solar_altitude=np.deg2rad(sun_altitude_deg),
        azimuthal_difference=np.deg2rad(azimuths - sun_azimuth_deg),
        observed_particle_azimuth=np.deg2rad(azimuths),
    )
    rayleigh_meridian_aop = rayleigh_frame_aop - np.deg2rad(azimuths)

    np.testing.assert_allclose(
        wrap_to_half_pi(result[AOP][0] - rayleigh_meridian_aop), 0.0, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# P7-P8 -- neutral point placement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [0.0, 15.0, 27.0, 45.0, 64.0])
def test_neutral_points_sit_at_the_published_distances(sun_altitude_deg):
    result = simulate(Pan, sun_altitude_deg)
    sun = make_sun(sun_altitude_deg)
    anti_sun = sun.directional_offset_by(position_angle=0 * deg, separation=180 * deg)

    babinet = SkyCoord(
        az=result[ABOVE_SUN_AZ].ravel() * rad,
        alt=result[ABOVE_SUN_ALT].ravel() * rad,
        frame=sun.frame,
    )
    brewster = SkyCoord(
        az=result[BELOW_SUN_AZ].ravel() * rad,
        alt=result[BELOW_SUN_ALT].ravel() * rad,
        frame=sun.frame,
    )
    arago = SkyCoord(
        az=result[ABOVE_ANTI_AZ].ravel() * rad,
        alt=result[ABOVE_ANTI_ALT].ravel() * rad,
        frame=sun.frame,
    )

    assert sun.separation(babinet).deg[0] == pytest.approx(
        babinet_offset_deg(sun_altitude_deg), abs=1e-6
    )
    assert sun.separation(brewster).deg[0] == pytest.approx(
        brewster_offset_deg(sun_altitude_deg), abs=1e-6
    )
    assert anti_sun.separation(arago).deg[0] == pytest.approx(
        brewster_offset_deg(sun_altitude_deg), abs=1e-6
    )
    assert result[ABOVE_SUN_ALT] > result[SUN_ALT]
    assert result[BELOW_SUN_ALT] < result[SUN_ALT]


def test_brewster_point_falls_below_the_horizon_at_moderate_solar_elevation():
    """Pan's fitted Brewster distance is large enough to sink the point."""
    result = simulate(Pan, 33.0)

    assert np.rad2deg(result[BELOW_SUN_ALT]).ravel()[0] == pytest.approx(-15.59, abs=1e-4)


# --------------------------------------------------------------------------- #
# P9-P11 -- range policy and the fidelity warning
# --------------------------------------------------------------------------- #


def test_brewster_breakpoint_is_reproduced_as_published():
    below = simulate(Pan, 27.0 - 1e-6)
    above = simulate(Pan, 27.0 + 1e-6)

    distance_below = 90.0 - np.rad2deg(below[BELOW_SUN_ALT]).ravel()[0] - (27.0 - 1e-6)
    distance_above = 90.0 - np.rad2deg(above[BELOW_SUN_ALT]).ravel()[0] - (27.0 + 1e-6)

    assert distance_below - distance_above == pytest.approx(0.48, abs=1e-4)


@pytest.mark.parametrize("sun_altitude_deg", [-5.0, 70.0])
def test_warns_outside_pans_measured_range(sun_altitude_deg):
    azimuths, altitudes = make_grid()
    model = Pan(times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        with pytest.warns(NeutralPointRangeWarning, match="outside the range measured"):
            model.simulate_sky(cie_sky_type=4, sun_position=make_sun(sun_altitude_deg))


def test_raise_policy_rejects_out_of_range_solar_elevation():
    azimuths, altitudes = make_grid()
    model = Pan(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        out_of_range="raise",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        with pytest.raises(ValueError, match="outside the range measured"):
            model.simulate_sky(cie_sky_type=4, sun_position=make_sun(70.0))


def test_fidelity_warning_is_issued_once_per_simulator():
    azimuths, altitudes = make_grid()
    model = Pan(times=TIMES, observation_location=LOCATION, azimuths=azimuths, altitudes=altitudes)

    with pytest.warns(PanFidelityWarning, match="referenced to the local meridian"):
        model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error", PanFidelityWarning)
        model.simulate_sky(cie_sky_type=4, sun_position=make_sun(33.0))


# --------------------------------------------------------------------------- #
# P12 -- Pan has no dop_max
# --------------------------------------------------------------------------- #


def test_pan_does_not_accept_dop_max():
    azimuths, altitudes = make_grid()
    with pytest.raises(TypeError):
        Pan(
            times=TIMES,
            observation_location=LOCATION,
            azimuths=azimuths,
            altitudes=altitudes,
            dop_max=0.8,
        )


# --------------------------------------------------------------------------- #
# P14-P15 -- Pan and QuEEN are distinct models
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 30.0, 60.0])
def test_pan_is_not_queen(sun_altitude_deg):
    pan = simulate(Pan, sun_altitude_deg)
    queen = simulate(QuEEN, sun_altitude_deg)

    assert np.nanmax(np.abs(pan[DOP] - queen[DOP])) > 0.05
    assert np.nanmax(np.abs(wrap_to_half_pi(pan[AOP] - queen[AOP]))) > 0.1


@pytest.mark.parametrize("sun_altitude_deg", [5.0, 30.0, 60.0])
def test_exact_relation_between_pan_and_queen(sun_altitude_deg):
    azimuths, altitudes = make_grid()
    pan = simulate(Pan, sun_altitude_deg, azimuths=azimuths, altitudes=altitudes)
    queen = simulate(QuEEN, sun_altitude_deg, azimuths=azimuths, altitudes=altitudes)

    np.testing.assert_allclose(
        wrap_to_half_pi(queen[AOP][0] - (pan[AOP][0] + np.deg2rad(azimuths))), 0.0, atol=1e-9
    )
    np.testing.assert_allclose(queen[DOP], pan[DOP] / (2 - pan[DOP]), atol=1e-9)


def test_pan_and_queen_agree_on_neutral_point_positions():
    """The models differ in DOP and AOP, not in where the singularities are."""
    pan = simulate(Pan, 33.0)
    queen = simulate(QuEEN, 33.0)

    for index in (
        ABOVE_SUN_AZ,
        ABOVE_SUN_ALT,
        BELOW_SUN_AZ,
        BELOW_SUN_ALT,
        ABOVE_ANTI_AZ,
        ABOVE_ANTI_ALT,
        BELOW_ANTI_AZ,
        BELOW_ANTI_ALT,
    ):
        np.testing.assert_allclose(pan[index], queen[index], atol=1e-9)
