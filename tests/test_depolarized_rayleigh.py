"""Tests for molecular-depolarized Rayleigh skylight polarization."""

from itertools import pairwise

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.engine import Engine
from pyskylumos.exceptions import ConfigurationError, InputTypeError
from pyskylumos.sky_models import DepolarizedRayleigh, Rayleigh

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)

DOP, AOP, RADIANCE, SCATTERING = 0, 1, 2, 3

DEFAULT_DELTA = 0.0279
EXPECTED_KING_FACTOR = 0.9587257754327136
EXPECTED_ANISOTROPY = 0.05740150190309639
EXPECTED_DOP_MAXIMUM = 0.945714563673509

SCATTERING_ANGLES_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0, 180.0])
EXPECTED_DEFAULT_DOP = np.array(
    [
        0.0,
        0.03365495381817989,
        0.13832012407689348,
        0.3210475907394563,
        0.5736569821193225,
        0.8297954425401549,
        0.945714563673509,
        0.5736569821193227,
        0.13832012407689348,
        0.0,
    ]
)


def make_grid(rows: int = 13, columns: int = 19) -> tuple[np.ndarray, np.ndarray]:
    """Return a representative world-AltAz sampling grid."""
    azimuths = np.tile(np.linspace(-179.0, 179.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(2.0, 88.0, rows)[:, None], (1, columns))
    return azimuths, altitudes


def make_sun(altitude_deg: float = 33.0, azimuth_deg: float = 137.0) -> SkyCoord:
    """Return an explicit sun position in the simulator's AltAz frame."""
    return SkyCoord(
        az=[azimuth_deg] * deg,
        alt=[altitude_deg] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )


def make_model(
    depolarization_ratio: float = DEFAULT_DELTA,
    *,
    azimuths: np.ndarray | None = None,
    altitudes: np.ndarray | None = None,
) -> DepolarizedRayleigh:
    """Construct the model on a deterministic grid."""
    if azimuths is None or altitudes is None:
        azimuths, altitudes = make_grid()
    return DepolarizedRayleigh(
        times=TIMES,
        observation_location=LOCATION,
        altitudes=altitudes,
        azimuths=azimuths,
        depolarization_ratio=depolarization_ratio,
    )


def axial_residual(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a signed axial-angle difference."""
    difference = first - second
    return 0.5 * np.arctan2(np.sin(2 * difference), np.cos(2 * difference))


# --------------------------------------------------------------------------- #
# published equation and fixtures
# --------------------------------------------------------------------------- #


def test_default_fixture_constants_are_independently_reproduced() -> None:
    king_factor = (1.0 - DEFAULT_DELTA) / (1.0 + 0.5 * DEFAULT_DELTA)
    anisotropy = 4.0 * (1.0 - king_factor) / (3.0 * king_factor)

    assert king_factor == pytest.approx(EXPECTED_KING_FACTOR, abs=1e-15)
    assert anisotropy == pytest.approx(EXPECTED_ANISOTROPY, abs=1e-15)
    assert 1.0 / (1.0 + anisotropy) == pytest.approx(EXPECTED_DOP_MAXIMUM, abs=1e-15)


def test_default_dop_matches_the_verified_scattering_angle_table() -> None:
    model = make_model()
    actual = model._dop_from_scattering_angle(np.deg2rad(SCATTERING_ANGLES_DEG))

    np.testing.assert_allclose(actual, EXPECTED_DEFAULT_DOP, rtol=0.0, atol=1e-12)


def test_zero_ratio_matches_ideal_rayleigh_element_for_element() -> None:
    azimuths, altitudes = make_grid()
    sun = make_sun()
    ideal = Rayleigh(TIMES, LOCATION, altitudes=altitudes, azimuths=azimuths).simulate_sky(
        cie_sky_type=4, sun_position=sun
    )
    depolarized = make_model(0.0, azimuths=azimuths, altitudes=altitudes).simulate_sky(
        cie_sky_type=4, sun_position=sun
    )

    np.testing.assert_allclose(depolarized[DOP], ideal[DOP], rtol=2e-15, atol=2e-15)


def test_dop_limits_symmetry_and_monotonic_depolarization() -> None:
    scattering = np.linspace(0.0, np.pi, 1001)
    ratios = (0.0, 0.01, DEFAULT_DELTA, 0.1, 0.25, 0.5)
    fields = [make_model(ratio)._dop_from_scattering_angle(scattering) for ratio in ratios]

    for dop in fields:
        assert np.all(np.isfinite(dop))
        assert np.all(dop >= 0.0)
        assert np.all(dop <= 1.0)
        np.testing.assert_allclose(dop, dop[::-1], rtol=0.0, atol=5e-15)

    for less_depolarized, more_depolarized in pairwise(fields):
        assert np.all(more_depolarized <= less_depolarized + 1e-15)

    assert fields[-1][500] == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert fields[2][500] == pytest.approx(EXPECTED_DOP_MAXIMUM, abs=1e-15)
    np.testing.assert_allclose(fields[2][[0, -1]], 0.0, rtol=0.0, atol=1e-30)


# --------------------------------------------------------------------------- #
# inherited Rayleigh behavior
# --------------------------------------------------------------------------- #


def test_only_dop_changes_from_rayleigh() -> None:
    azimuths, altitudes = make_grid()
    sun = make_sun()
    ideal = Rayleigh(TIMES, LOCATION, altitudes=altitudes, azimuths=azimuths).simulate_sky(
        cie_sky_type=4, sun_position=sun
    )
    actual = make_model(azimuths=azimuths, altitudes=altitudes).simulate_sky(
        cie_sky_type=4, sun_position=sun
    )

    assert np.nanmax(actual[DOP]) < np.nanmax(ideal[DOP])
    np.testing.assert_allclose(actual[AOP], ideal[AOP], rtol=0.0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(actual[RADIANCE], ideal[RADIANCE])
    np.testing.assert_array_equal(actual[SCATTERING], ideal[SCATTERING])


def test_parameters_and_return_values_keep_rayleighs_eight_entry_schema() -> None:
    model = make_model()
    result = model.simulate_sky(cie_sky_type=4, sun_position=make_sun())

    assert len(model.parameters_simulated) == 8
    assert len(result) == 8
    assert model.parameters_simulated == (
        "degree of polarization",
        "angle of polarization",
        "radiance",
        "scattering angle",
        "sun azimuth",
        "sun elevation",
        "anti-sun azimuth",
        "anti-sun elevation",
    )
    assert all(value.dtype == np.float64 for value in result)


def test_aop_is_nan_at_the_sun_and_anti_sun() -> None:
    azimuths = np.array([[137.0, -43.0, 90.0]])
    altitudes = np.array([[33.0, -33.0, 0.0]])
    result = make_model(azimuths=azimuths, altitudes=altitudes).simulate_sky(
        cie_sky_type=4, sun_position=make_sun()
    )

    singular = result[DOP] <= 1e-15
    assert singular.sum() == 2
    assert np.isnan(result[AOP][singular]).all()
    assert np.isfinite(result[AOP][~singular]).all()


@pytest.mark.parametrize("rotation_deg", [13.7, 137.0])
def test_rotation_covariance(rotation_deg: float) -> None:
    azimuths, altitudes = make_grid(rows=9, columns=15)

    def evaluate(sun_azimuth_deg: float) -> list[np.ndarray]:
        model = make_model(
            azimuths=azimuths + sun_azimuth_deg,
            altitudes=altitudes,
        )
        return model.simulate_sky(
            cie_sky_type=4,
            sun_position=make_sun(33.0, sun_azimuth_deg),
        )

    base = evaluate(100.0)
    rotated = evaluate(100.0 + rotation_deg)

    np.testing.assert_allclose(rotated[DOP], base[DOP], rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(rotated[SCATTERING], base[SCATTERING], rtol=0.0, atol=2e-14)
    valid = np.isfinite(base[AOP]) & np.isfinite(rotated[AOP])
    expected = np.deg2rad(rotation_deg)
    np.testing.assert_allclose(
        axial_residual(rotated[AOP][valid], base[AOP][valid] + expected),
        0.0,
        rtol=0.0,
        atol=2e-12,
    )


# --------------------------------------------------------------------------- #
# validation and Engine dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ratio", [0.0, 0.5])
def test_validation_accepts_published_range_boundaries(ratio: float) -> None:
    assert make_model(ratio).depolarization_ratio == ratio


@pytest.mark.parametrize(
    "ratio",
    [-1e-9, 0.500000001, np.nan, np.inf, -np.inf, 0.1 + 0j, np.array([0.1]), "0.1", True],
)
def test_validation_rejects_invalid_ratios(ratio: object) -> None:
    with pytest.raises(ConfigurationError, match="depolarization_ratio"):
        make_model(ratio)  # type: ignore[arg-type]


@pytest.mark.parametrize("option", ["turbidity", "wavelength_nm", "albedo"])
def test_engine_rejects_unsupported_options(option: str) -> None:
    azimuths, altitudes = make_grid(rows=3, columns=5)
    with pytest.raises(InputTypeError, match="does not accept model_options"):
        Engine._Engine__get_sky_simulator(
            times=TIMES,
            sky_model="DEPOLARIZED_RAYLEIGH",
            azimuths=azimuths,
            altitudes=altitudes,
            observation_location=LOCATION,
            model_options={option: 1.0},
        )


def test_engine_forwards_the_ratio_and_matches_direct_construction() -> None:
    azimuths, altitudes = make_grid(rows=7, columns=11)
    direct = make_model(0.1, azimuths=azimuths, altitudes=altitudes)
    through_engine = Engine._Engine__get_sky_simulator(
        times=TIMES,
        sky_model="DEPOLARIZED_RAYLEIGH",
        azimuths=azimuths,
        altitudes=altitudes,
        observation_location=LOCATION,
        model_options={"depolarization_ratio": 0.1},
    )

    assert isinstance(through_engine, DepolarizedRayleigh)
    assert through_engine.depolarization_ratio == 0.1
    direct_result = direct.simulate_sky(cie_sky_type=4, sun_position=make_sun())
    engine_result = through_engine.simulate_sky(cie_sky_type=4, sun_position=make_sun())
    for direct_value, engine_value in zip(direct_result, engine_result, strict=True):
        np.testing.assert_array_equal(direct_value, engine_value)
