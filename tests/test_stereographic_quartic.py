"""Tests for the shared stereographic geometry and Berry's quartic field."""

import numpy as np
import pytest

from pyskylumos.sky_models.Rayleigh import Rayleigh
from pyskylumos.sky_models.StereographicQuartic import (
    antipode,
    offset_along_solar_vertical,
    omega,
    project,
    unproject,
)

SUN_ZENITH = np.deg2rad(60.0)
SUN_AZIMUTH = np.deg2rad(137.0)
BELOW_SUN_OFFSET = np.deg2rad(49.34)
ABOVE_SUN_OFFSET = np.deg2rad(25.73)


def angular_separation(zenith_a, azimuth_a, zenith_b, azimuth_b):
    """Return the great-circle separation between two directions, in radians."""
    return np.arccos(
        np.clip(
            np.cos(zenith_a) * np.cos(zenith_b)
            + np.sin(zenith_a) * np.sin(zenith_b) * np.cos(azimuth_a - azimuth_b),
            -1.0,
            1.0,
        )
    )


def wrap_to_half_pi(angle):
    """Wrap an axial angle onto (-pi/2, pi/2]."""
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def solar_side_roots(below_offset=BELOW_SUN_OFFSET, above_offset=ABOVE_SUN_OFFSET):
    """Return the (below-sun, above-sun) roots for the module-level sun position."""
    below = offset_along_solar_vertical(
        sun_zenith_angle=SUN_ZENITH,
        sun_azimuth=SUN_AZIMUTH,
        angular_offset=below_offset,
        above_sun=False,
    )
    above = offset_along_solar_vertical(
        sun_zenith_angle=SUN_ZENITH,
        sun_azimuth=SUN_AZIMUTH,
        angular_offset=above_offset,
        above_sun=True,
    )
    return below, above


# --------------------------------------------------------------------------- #
# projection round trip
# --------------------------------------------------------------------------- #


def test_unproject_inverts_project():
    zenith = np.deg2rad(np.linspace(0.5, 179.5, 61))[:, None]
    azimuth = np.linspace(-np.pi + 1e-6, np.pi, 73)[None, :]

    recovered_zenith, recovered_azimuth = unproject(project(zenith, azimuth))

    np.testing.assert_allclose(
        recovered_zenith, np.broadcast_to(zenith, recovered_zenith.shape), atol=1e-12
    )
    np.testing.assert_allclose(
        recovered_azimuth, np.broadcast_to(azimuth, recovered_azimuth.shape), atol=1e-12
    )


def test_project_maps_zenith_to_origin_and_horizon_to_unit_circle():
    assert abs(project(np.float64(0.0), np.float64(1.234))) == pytest.approx(0.0, abs=1e-15)
    assert abs(project(np.float64(np.pi / 2), np.float64(1.234))) == pytest.approx(1.0, abs=1e-15)


def test_antipode_is_an_involution():
    zenith = np.deg2rad(np.linspace(1.0, 179.0, 37))[:, None]
    azimuth = np.linspace(-np.pi + 1e-6, np.pi, 41)[None, :]
    projection = project(zenith, azimuth)

    np.testing.assert_allclose(antipode(antipode(projection)), projection, atol=1e-12)


def test_antipode_is_diametrically_opposite_on_the_sphere():
    zenith = np.deg2rad(np.array([10.0, 45.0, 90.0, 130.0]))
    azimuth = np.deg2rad(np.array([0.0, 95.0, 210.0, 330.0]))

    anti_zenith, anti_azimuth = unproject(antipode(project(zenith, azimuth)))

    np.testing.assert_allclose(
        angular_separation(zenith, azimuth, anti_zenith, anti_azimuth),
        np.pi,
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# half-angle placement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("offset_deg", [0.5, 8.93, 15.0, 25.73, 49.34, 60.0])
@pytest.mark.parametrize("above_sun", [True, False])
def test_offset_along_solar_vertical_gives_exact_angular_distance(offset_deg, above_sun):
    offset = np.deg2rad(offset_deg)

    zenith, azimuth = unproject(
        offset_along_solar_vertical(
            sun_zenith_angle=SUN_ZENITH,
            sun_azimuth=SUN_AZIMUTH,
            angular_offset=offset,
            above_sun=above_sun,
        )
    )

    assert angular_separation(SUN_ZENITH, SUN_AZIMUTH, zenith, azimuth) == pytest.approx(
        offset, abs=1e-9
    )
    expected_zenith = SUN_ZENITH - offset if above_sun else SUN_ZENITH + offset
    assert zenith == pytest.approx(expected_zenith, abs=1e-12)


def test_offset_along_solar_vertical_stays_on_the_solar_meridian():
    below, above = solar_side_roots()

    for root in (below, above):
        assert np.angle(root) == pytest.approx(np.angle(np.exp(1j * SUN_AZIMUTH)), abs=1e-12)


# --------------------------------------------------------------------------- #
# the quartic field
# --------------------------------------------------------------------------- #


def test_omega_vanishes_at_all_four_roots():
    below, above = solar_side_roots()
    roots = (below, above, antipode(below), antipode(above))

    for root in roots:
        assert abs(omega(root, below, above)) == pytest.approx(0.0, abs=1e-10)


def test_omega_modulus_is_antipodally_invariant():
    below, above = solar_side_roots()
    zenith = np.deg2rad(np.array([12.0, 47.0, 88.0, 121.0, 165.0]))
    azimuth = np.deg2rad(np.array([5.0, 99.0, 187.0, 264.0, 341.0]))
    projection = project(zenith, azimuth)

    np.testing.assert_allclose(
        np.abs(omega(antipode(projection), below, above)),
        np.abs(omega(projection, below, above)),
        atol=1e-12,
    )


def test_omega_modulus_is_normalized_to_unity_over_the_sphere():
    below, above = solar_side_roots()
    zenith = np.deg2rad(np.linspace(0.001, 179.999, 1200))[:, None]
    azimuth = np.deg2rad(np.linspace(0.0, 360.0, 721))[None, :]

    modulus = np.abs(omega(project(zenith, azimuth), below, above))

    assert modulus.max() <= 1.0 + 1e-12
    assert modulus.max() >= 0.999
    assert modulus.min() >= 0.0


@pytest.mark.parametrize("sun_elevation_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_omega_normalization_holds_for_symmetric_berry_offsets(sun_elevation_deg):
    sun_zenith = np.deg2rad(90.0 - sun_elevation_deg)
    offset = np.deg2rad(15.0)
    below = offset_along_solar_vertical(sun_zenith, 0.0, offset, above_sun=False)
    above = offset_along_solar_vertical(sun_zenith, 0.0, offset, above_sun=True)

    zenith = np.deg2rad(np.linspace(0.001, 179.999, 900))[:, None]
    azimuth = np.deg2rad(np.linspace(0.0, 360.0, 541))[None, :]

    modulus = np.abs(omega(project(zenith, azimuth), below, above))

    assert modulus.max() <= 1.0 + 1e-12
    assert modulus.max() >= 0.999


# --------------------------------------------------------------------------- #
# Global AOP convention: e^{-2i alpha_s} (README mathematical reference, section 4)
# --------------------------------------------------------------------------- #


def test_compensated_phase_reproduces_rayleigh_in_the_small_splitting_limit():
    """Collapsing both offsets recovers the Rayleigh limit exactly.

    This is the strongest single check on the construction: it simultaneously
    validates the root placement, the leading ``-4``, the azimuth handedness and
    the ``e^{-2i*alpha_s}`` phase factor against an independently written model.

    Both sides use ``float64``. The residual tolerance reflects the deliberately
    small but nonzero root separation used to approximate the collapsed Rayleigh
    limit, rather than a dtype mismatch.
    """
    negligible_offset = 1e-6
    below, above = solar_side_roots(negligible_offset, negligible_offset)

    point_zenith = np.deg2rad(np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 35.0, 45.0]))
    point_azimuth = np.deg2rad(np.array([10.0, 200.0, 300.0, 45.0, 260.0, 330.0, 15.0, 95.0]))

    field = omega(project(point_zenith, point_azimuth), below, above)
    compensated_aop = 0.5 * np.angle(field * np.exp(-2j * SUN_AZIMUTH))

    rayleigh_aop = Rayleigh._Rayleigh__get_aop(
        observed_point_altitude=np.pi / 2 - point_zenith,
        solar_altitude=np.pi / 2 - SUN_ZENITH,
        azimuthal_difference=point_azimuth - SUN_AZIMUTH,
        observed_particle_azimuth=point_azimuth,
    )

    np.testing.assert_allclose(
        wrap_to_half_pi(compensated_aop - rayleigh_aop),
        0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("rotation_deg", [37.0, 123.0, 270.0])
def test_compensated_phase_is_rotation_covariant(rotation_deg):
    """Rigidly rotating the scene by beta must rotate a frame-referenced AOP by beta."""
    rotation = np.deg2rad(rotation_deg)
    point_zenith = np.deg2rad(np.array([20.0, 50.0, 80.0, 110.0]))
    relative_azimuth = np.deg2rad(np.array([13.0, 145.0, 260.0, 300.0]))

    def evaluate(sun_azimuth):
        below = offset_along_solar_vertical(
            SUN_ZENITH, sun_azimuth, BELOW_SUN_OFFSET, above_sun=False
        )
        above = offset_along_solar_vertical(
            SUN_ZENITH, sun_azimuth, ABOVE_SUN_OFFSET, above_sun=True
        )
        field = omega(project(point_zenith, sun_azimuth + relative_azimuth), below, above)
        return 0.5 * np.angle(field * np.exp(-2j * sun_azimuth)), np.abs(field)

    base_aop, base_modulus = evaluate(SUN_AZIMUTH)
    rotated_aop, rotated_modulus = evaluate(SUN_AZIMUTH + rotation)

    np.testing.assert_allclose(wrap_to_half_pi(rotated_aop - base_aop - rotation), 0.0, atol=1e-9)
    np.testing.assert_allclose(rotated_modulus, base_modulus, atol=1e-12)


@pytest.mark.parametrize("rotation_deg", [37.0, 123.0])
def test_uncompensated_phase_rotates_twice_too_fast(rotation_deg):
    """The bare quartic gains exp(4 i beta): without compensation AOP moves by 2 beta."""
    rotation = np.deg2rad(rotation_deg)
    point_zenith = np.deg2rad(np.array([20.0, 50.0, 80.0, 110.0]))
    relative_azimuth = np.deg2rad(np.array([13.0, 145.0, 260.0, 300.0]))

    def evaluate(sun_azimuth):
        below = offset_along_solar_vertical(
            SUN_ZENITH, sun_azimuth, BELOW_SUN_OFFSET, above_sun=False
        )
        above = offset_along_solar_vertical(
            SUN_ZENITH, sun_azimuth, ABOVE_SUN_OFFSET, above_sun=True
        )
        return 0.5 * np.angle(
            omega(project(point_zenith, sun_azimuth + relative_azimuth), below, above)
        )

    base_aop = evaluate(SUN_AZIMUTH)
    rotated_aop = evaluate(SUN_AZIMUTH + rotation)

    np.testing.assert_allclose(
        wrap_to_half_pi(rotated_aop - base_aop - 2 * rotation), 0.0, atol=1e-9
    )


def test_leading_minus_four_sets_the_absolute_aop_orientation():
    """Dropping the leading -4 would rotate every AOP by exactly 90 degrees."""
    below, above = solar_side_roots()
    projection = project(
        np.deg2rad(np.array([25.0, 55.0, 85.0])), np.deg2rad(np.array([30.0, 150.0, 290.0]))
    )

    signed = omega(projection, below, above)
    unsigned = -signed

    np.testing.assert_allclose(
        wrap_to_half_pi(0.5 * np.angle(unsigned) - 0.5 * np.angle(signed) - np.pi / 2),
        0.0,
        atol=1e-12,
    )
