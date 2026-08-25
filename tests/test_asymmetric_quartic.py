"""Mathematical and public-contract tests for the asymmetric quartic model."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.exceptions import ConfigurationError, InputValidationError
from pyskylumos.sky_models import AsymmetricQuartic, Berry, QuEEN
from pyskylumos.sky_models.NeutralPointOffsets import NeutralPointRangeWarning
from pyskylumos.sky_models.StereographicQuartic import (
    antipode,
    half_angle_modulus,
    omega_from_roots,
    project,
    unproject,
)

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
SUN = SkyCoord(
    az=[137.0] * deg,
    alt=[33.0] * deg,
    frame=AltAz(obstime=TIMES, location=LOCATION),
)
AZIMUTHS = np.tile(np.linspace(0.0, 359.0, 65), (65, 1))
ALTITUDES = np.tile(np.linspace(0.5, 89.5, 65)[:, None], (1, 65))


class FixedOffsetAsymmetric(AsymmetricQuartic):
    """Asymmetric model fixture with Berry's fixed 15-degree solar offsets."""

    def _neutral_point_offsets(self, sun_altitudes_deg):
        offset = np.full_like(sun_altitudes_deg, np.deg2rad(15.0), dtype=np.float64)
        return offset, offset


def make_model(model_class=AsymmetricQuartic, **kwargs):
    """Construct a model on the shared 65-by-65 reference grid."""
    kwargs.setdefault("out_of_range", "ignore")
    return model_class(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=AZIMUTHS,
        altitudes=ALTITUDES,
        **kwargs,
    )


def components(model, sun=SUN):
    """Return geometry, signed offsets, roots and raw normalized field."""
    sun_zenith = np.asarray(np.pi / 2 - sun.alt.radian, dtype=np.float64)
    sun_azimuth = np.asarray(sun.az.radian, dtype=np.float64)
    observed_zenith = np.asarray(np.pi / 2 - model.sky_map.alt.radian, dtype=np.float64)
    observed_azimuth = np.asarray(model.sky_map.az.radian, dtype=np.float64)
    below, above = model._neutral_point_offsets(np.asarray(sun.alt.deg, dtype=np.float64))
    signed = model._signed_offsets(below, above)
    roots = model._roots(sun_zenith, sun_azimuth, signed)
    field, _ = model._build_field(
        project(observed_zenith, observed_azimuth),
        observed_zenith,
        observed_azimuth,
        sun_zenith,
        sun_azimuth,
        below,
        above,
    )
    return sun_zenith, sun_azimuth, observed_zenith, observed_azimuth, signed, roots, field


def axial_residual(first, second):
    """Return an axial-angle difference on [-pi/2, pi/2)."""
    return 0.5 * np.arctan2(np.sin(2 * (first - second)), np.cos(2 * (first - second)))


def full_angle_shape(observed_zenith, observed_azimuth, signed, sun_zenith, sun_azimuth):
    """Evaluate the rejected full-angle product as an independent test oracle."""
    azimuth_difference = np.cos(observed_azimuth - sun_azimuth)
    product = np.ones_like(observed_zenith, dtype=np.float64)
    for offset in signed:
        meridian_angle = sun_zenith + offset
        cosine_separation = (
            np.cos(observed_zenith) * np.cos(meridian_angle)
            + np.sin(observed_zenith) * np.sin(meridian_angle) * azimuth_difference
        )
        separation = np.arccos(np.clip(cosine_separation, -1.0, 1.0))
        product = product * np.abs(np.sin(separation))
    return np.sqrt(product)


def test_four_requested_roots_are_zeros_of_both_field_constructions():
    model = make_model(arago_offset=31.0, fourth_offset=43.0, normalisation="berry")
    sun_zenith, sun_azimuth, _, _, signed, roots, _ = components(model)

    analytic_residuals = []
    half_angle_residuals = []
    for root in roots:
        root_zenith, root_azimuth = unproject(root)
        analytic_residuals.append(float(np.max(np.abs(omega_from_roots(root, roots)))))
        half_angle_residuals.append(
            float(
                np.max(
                    4
                    * half_angle_modulus(
                        root_zenith,
                        root_azimuth,
                        signed,
                        sun_zenith,
                        sun_azimuth,
                    )
                )
            )
        )

    assert max(analytic_residuals) < 2e-14
    assert max(half_angle_residuals) < 1e-7


def test_berry_limit_matches_raw_modulus_and_fixed_world_aop():
    asymmetric = make_model(
        FixedOffsetAsymmetric,
        arago_offset="brewster",
        fourth_offset="babinet",
        normalisation="berry",
    )
    berry = Berry(TIMES, LOCATION, azimuths=AZIMUTHS, altitudes=ALTITUDES)

    *_, field = components(asymmetric)
    expected = berry.simulate_sky(cie_sky_type=4, sun_position=SUN)
    actual_aop = asymmetric._get_aop(
        field,
        np.asarray(SUN.az.radian, dtype=np.float64),
        np.asarray(asymmetric.sky_map.az.radian, dtype=np.float64),
    )

    np.testing.assert_allclose(np.abs(field), expected[0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(axial_residual(actual_aop, expected[1]), 0.0, atol=1e-12, rtol=0.0)


def test_queen_limit_matches_public_dop_and_aop():
    asymmetric = make_model(
        arago_offset="brewster",
        fourth_offset="babinet",
        normalisation="berry",
    )
    queen = QuEEN(
        TIMES,
        LOCATION,
        azimuths=AZIMUTHS,
        altitudes=ALTITUDES,
        out_of_range="ignore",
    )

    actual = asymmetric.simulate_sky(cie_sky_type=4, sun_position=SUN)
    expected = queen.simulate_sky(cie_sky_type=4, sun_position=SUN)

    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(axial_residual(actual[1], expected[1]), 0.0, atol=1e-12, rtol=0.0)


def test_full_angle_form_agrees_only_in_the_symmetric_limit_and_adds_a_false_zero():
    symmetric = make_model(
        FixedOffsetAsymmetric,
        arago_offset="brewster",
        fourth_offset="babinet",
        normalisation="berry",
    )
    sun_zenith, sun_azimuth, zenith, azimuth, signed, _, _ = components(symmetric)
    correct = 4 * half_angle_modulus(zenith, azimuth, signed, sun_zenith, sun_azimuth)
    rejected = full_angle_shape(zenith, azimuth, signed, sun_zenith, sun_azimuth)
    np.testing.assert_allclose(rejected, correct, atol=2e-15, rtol=0.0)

    asymmetric = make_model(normalisation="berry")
    sun_zenith, sun_azimuth, _, _, signed, roots, _ = components(asymmetric)
    false_zero_zenith, false_zero_azimuth = unproject(antipode(roots[3]))
    rejected_at_antipode = full_angle_shape(
        false_zero_zenith,
        false_zero_azimuth,
        signed,
        sun_zenith,
        sun_azimuth,
    )
    correct_at_antipode = 4 * half_angle_modulus(
        false_zero_zenith,
        false_zero_azimuth,
        signed,
        sun_zenith,
        sun_azimuth,
    )

    assert float(np.max(rejected_at_antipode)) < 1e-7
    assert float(np.max(correct_at_antipode)) == pytest.approx(0.4217, abs=5e-4)


def test_each_broken_symmetry_root_has_index_one_half():
    model = make_model(arago_offset=31.0, fourth_offset=43.0, normalisation="berry")
    *_, roots, _ = components(model)
    circle_angle = np.linspace(0.0, 2 * np.pi, 1441)
    indices = []

    for root in roots:
        radius = 1e-6 * np.maximum(1.0, np.abs(root))
        circle = root + radius * np.exp(1j * circle_angle)
        phase = np.unwrap(np.angle(omega_from_roots(circle, roots)))
        indices.append(float((phase[-1] - phase[0]) / (4 * np.pi)))

    np.testing.assert_allclose(indices, 0.5, atol=1e-8, rtol=0.0)
    assert sum(indices) == pytest.approx(2.0, abs=1e-8)


def test_antipodal_invariance_holds_only_for_paired_roots():
    symmetric = make_model(
        FixedOffsetAsymmetric,
        arago_offset="brewster",
        fourth_offset="babinet",
        normalisation="berry",
    )
    sun_zenith, sun_azimuth, zenith, azimuth, signed, _, _ = components(symmetric)
    symmetric_modulus = half_angle_modulus(zenith, azimuth, signed, sun_zenith, sun_azimuth)
    symmetric_antipode = half_angle_modulus(
        np.pi - zenith,
        azimuth + np.pi,
        signed,
        sun_zenith,
        sun_azimuth,
    )
    np.testing.assert_allclose(symmetric_antipode, symmetric_modulus, atol=5e-15, rtol=0.0)

    asymmetric = make_model(normalisation="berry")
    sun_zenith, sun_azimuth, zenith, azimuth, signed, _, _ = components(asymmetric)
    asymmetric_modulus = half_angle_modulus(zenith, azimuth, signed, sun_zenith, sun_azimuth)
    asymmetric_antipode = half_angle_modulus(
        np.pi - zenith,
        azimuth + np.pi,
        signed,
        sun_zenith,
        sun_azimuth,
    )

    assert float(np.max(np.abs(asymmetric_modulus - asymmetric_antipode))) == pytest.approx(
        0.1050, abs=5e-4
    )


def test_default_offsets_match_pan_fits_and_horvath_anchor():
    model = make_model()
    below, above = model._neutral_point_offsets(np.asarray(SUN.alt.deg, dtype=np.float64))
    signed = model._signed_offsets(below, above)

    assert np.rad2deg(above).item() == pytest.approx(24.05, abs=1e-12)
    assert np.rad2deg(below).item() == pytest.approx(48.59, abs=1e-12)
    assert np.rad2deg(signed[2] + np.pi).item() == pytest.approx(48.59, abs=1e-12)
    assert np.rad2deg(-np.pi - signed[3]).item() == pytest.approx(48.59, abs=1e-12)


@pytest.mark.parametrize("sun_elevation", [5.0, 15.0, 33.0, 45.0, 60.0])
@pytest.mark.parametrize(
    "offsets",
    [
        {"arago_offset": "brewster", "fourth_offset": "brewster"},
        {"arago_offset": "brewster", "fourth_offset": "babinet"},
        {"arago_offset": 31.0, "fourth_offset": 43.0},
    ],
)
def test_peak_normalisation_is_bounded_over_reference_sweep(sun_elevation, offsets):
    sun = SkyCoord(
        az=[137.0] * deg,
        alt=[sun_elevation] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )
    model = make_model(normalisation="peak", **offsets)
    result = model.simulate_sky(cie_sky_type=4, sun_position=sun)
    modulus = 2 * result[0] / (1 + result[0])

    assert float(np.nanmax(modulus)) <= 1.0 + 1e-12
    assert float(np.nanmax(modulus)) >= 0.95


def test_shared_invariants_dtypes_and_schema():
    result = make_model().simulate_sky(cie_sky_type=4, sun_position=SUN)
    dop, aop = result[:2]

    assert len(result) == 16
    assert len(make_model().parameters_simulated) == 16
    assert all(value.dtype == np.float64 for value in result)
    assert np.nanmin(dop) == pytest.approx(0.011034184943970482, abs=1e-12)
    assert np.nanmax(dop) == pytest.approx(0.9988496246816693, abs=1e-12)
    assert np.all(dop[np.isfinite(dop)] >= 0.0)
    assert np.all(dop[np.isfinite(dop)] <= 1.0)
    assert np.all(aop[np.isfinite(aop)] > -np.pi / 2)
    assert np.all(aop[np.isfinite(aop)] <= np.pi / 2)


@pytest.mark.parametrize(
    "name,value",
    [
        ("arago_offset", "arago"),
        ("arago_offset", True),
        ("arago_offset", np.nan),
        ("arago_offset", np.inf),
        ("arago_offset", 1 + 2j),
        ("arago_offset", np.array([15.0])),
        ("fourth_offset", "fourth"),
        ("normalisation", "closed_form"),
        ("dop_max", 0.0),
        ("dop_max", 1.01),
        ("dop_max", True),
        ("out_of_range", "clamp"),
    ],
)
def test_invalid_model_options_are_rejected(name, value):
    with pytest.raises(ConfigurationError):
        make_model(**{name: value})


@pytest.mark.parametrize("name,value", [("arago_offset", -720.0), ("fourth_offset", 810.0)])
def test_finite_constant_offsets_are_not_artificially_range_limited(name, value):
    assert make_model(**{name: value}) is not None


def test_out_of_range_policy_is_forwarded_to_pan_offsets():
    model = make_model(out_of_range="raise")
    high_sun = SkyCoord(
        az=[137.0] * deg,
        alt=[70.0] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )
    with pytest.raises(InputValidationError, match="outside the range measured"):
        model.simulate_sky(cie_sky_type=4, sun_position=high_sun)

    with warnings.catch_warnings():
        warnings.simplefilter("error", NeutralPointRangeWarning)
        make_model(out_of_range="ignore").simulate_sky(cie_sky_type=4, sun_position=high_sun)
