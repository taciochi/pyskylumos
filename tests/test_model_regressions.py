"""Cross-model regression tests.

The golden arrays in ``tests/data/golden_sky_models.npz`` were captured from the
pre-split code base. The ``queen_*`` entries come from the model that shipped as
``Pan`` up to version 0.0.6, which is the guarantee behind the migration note in
README migration note: ``sky_model="QUEEN"`` reproduces the old ``sky_model="PAN"``.
Berry's historical DOP fixture contains the former project mapping; it is
analytically inverted below so the fixture remains immutable compatibility evidence.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

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

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_sky_models.npz"
SUN_ELEVATIONS = [5.0, 15.0, 27.0, 30.0, 45.0, 60.0]
SUN_AZIMUTH = 137.0
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
TIMES = Time(["2026-06-21T12:00:00"])

DOP, AOP, RADIANCE = 0, 1, 2

MODELS = {"rayleigh": Rayleigh, "berry": Berry, "queen": QuEEN}


@pytest.fixture(scope="module")
def golden():
    """Load the pre-split golden arrays."""
    return np.load(GOLDEN_PATH)


@pytest.fixture(scope="module")
def grid(golden):
    """Return the azimuth/altitude grid the goldens were captured on."""
    return golden["azimuths"], golden["altitudes"]


def simulate(model_class, sun_elevation_deg, grid, **kwargs):
    """Run a model on the golden grid for a fixed sun position."""
    azimuths, altitudes = grid
    sun = SkyCoord(
        az=[SUN_AZIMUTH] * deg,
        alt=[sun_elevation_deg] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )
    if model_class is AsymmetricQuartic:
        kwargs.setdefault("out_of_range", "ignore")
    model = model_class(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        return model.simulate_sky(cie_sky_type=4, sun_position=sun, altitude_min_clip=0.0)


def wrap_to_half_pi(angle):
    """Wrap an axial angle onto (-pi/2, pi/2]."""
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def finite_pair(first, second):
    """Return the two arrays restricted to positions finite in both."""
    mask = np.isfinite(first) & np.isfinite(second)
    return first[mask], second[mask]


# --------------------------------------------------------------------------- #
# R1 -- goldens
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", sorted(MODELS))
@pytest.mark.parametrize("sun_elevation_deg", SUN_ELEVATIONS)
def test_matches_the_pre_split_golden(golden, grid, model_name, sun_elevation_deg):
    result = simulate(MODELS[model_name], sun_elevation_deg, grid)

    for key, index in (("dop", DOP), ("aop", AOP), ("radiance", RADIANCE)):
        expected = golden[f"{model_name}_{sun_elevation_deg:g}_{key}"]
        if model_name == "berry" and key == "dop":
            # Legacy q = |omega| / (2 - |omega|), hence |omega| = 2q / (1 + q).
            expected = 2 * expected / (1 + expected)
        np.testing.assert_allclose(
            np.asarray(result[index], dtype=np.float64),
            expected,
            atol=1e-6,
            equal_nan=True,
        )


def test_queen_reproduces_the_model_that_shipped_as_pan(golden, grid):
    """The migration guarantee: QUEEN in 0.1.0 equals PAN in 0.0.6."""
    for sun_elevation_deg in SUN_ELEVATIONS:
        result = simulate(QuEEN, sun_elevation_deg, grid)
        for key, index in (("dop", DOP), ("aop", AOP)):
            np.testing.assert_allclose(
                np.asarray(result[index], dtype=np.float64),
                golden[f"queen_{sun_elevation_deg:g}_{key}"],
                atol=1e-12,
                equal_nan=True,
            )


# --------------------------------------------------------------------------- #
# R2-R4 -- the models are mutually distinct
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_elevation_deg", SUN_ELEVATIONS)
def test_rayleigh_and_queen_disagree_once_the_neutral_points_split(grid, sun_elevation_deg):
    rayleigh = simulate(Rayleigh, sun_elevation_deg, grid)
    queen = simulate(QuEEN, sun_elevation_deg, grid)

    rayleigh_aop, queen_aop = finite_pair(rayleigh[AOP], queen[AOP])

    assert np.max(np.abs(wrap_to_half_pi(rayleigh_aop - queen_aop))) > 0.1


@pytest.mark.parametrize("sun_elevation_deg", SUN_ELEVATIONS)
def test_berry_and_queen_place_their_neutral_points_differently(grid, sun_elevation_deg):
    berry = simulate(Berry, sun_elevation_deg, grid)
    queen = simulate(QuEEN, sun_elevation_deg, grid)

    # Index 7 is the above-sun (Babinet) elevation, index 9 the below-sun
    # (Brewster) elevation, and index 5 the sun's own elevation.
    # Pan's Babinet fit crosses Berry's constant 15 degrees near a solar
    # elevation of 49 degrees, so the Babinet offsets can nearly coincide; the
    # Brewster offsets never do.
    berry_offsets = np.rad2deg([berry[7] - berry[5], berry[9] - berry[5]]).ravel()
    queen_offsets = np.rad2deg([queen[7] - queen[5], queen[9] - queen[5]]).ravel()

    assert np.max(np.abs(berry_offsets - queen_offsets)) > 5.0

    berry_dop, queen_dop = finite_pair(berry[DOP], queen[DOP])
    assert np.max(np.abs(berry_dop - queen_dop)) > 0.05


@pytest.mark.parametrize("sun_elevation_deg", SUN_ELEVATIONS)
def test_pan_and_queen_are_not_aliases(grid, sun_elevation_deg):
    pan = simulate(Pan, sun_elevation_deg, grid)
    queen = simulate(QuEEN, sun_elevation_deg, grid)

    pan_dop, queen_dop = finite_pair(pan[DOP], queen[DOP])
    pan_aop, queen_aop = finite_pair(pan[AOP], queen[AOP])

    assert np.max(np.abs(pan_dop - queen_dop)) > 0.05
    assert np.max(np.abs(wrap_to_half_pi(pan_aop - queen_aop))) > 0.1


# --------------------------------------------------------------------------- #
# R5 -- no two models coincide
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sun_elevation_deg", SUN_ELEVATIONS)
def test_no_two_models_produce_identical_fields(grid, sun_elevation_deg):
    results = {
        name: simulate(model_class, sun_elevation_deg, grid)
        for name, model_class in (
            ("rayleigh", Rayleigh),
            ("depolarized_rayleigh", DepolarizedRayleigh),
            ("asymmetric", AsymmetricQuartic),
            ("berry", Berry),
            ("pan", Pan),
            ("queen", QuEEN),
        )
    }
    names = sorted(results)

    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            first, second = results[first_name], results[second_name]

            first_dop, second_dop = finite_pair(first[DOP], second[DOP])
            first_aop, second_aop = finite_pair(first[AOP], second[AOP])

            differs = (
                np.max(np.abs(first_dop - second_dop)) > 1e-6
                or np.max(np.abs(wrap_to_half_pi(first_aop - second_aop))) > 1e-6
            )
            assert differs, f"{first_name} and {second_name} produce identical fields"


# --------------------------------------------------------------------------- #
# R6 -- shared invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model_name", ["rayleigh", "depolarized_rayleigh", "asymmetric", "berry", "pan", "queen"]
)
@pytest.mark.parametrize("sun_elevation_deg", SUN_ELEVATIONS)
def test_every_model_respects_the_shared_invariants(grid, model_name, sun_elevation_deg):
    model_class = {
        "rayleigh": Rayleigh,
        "depolarized_rayleigh": DepolarizedRayleigh,
        "asymmetric": AsymmetricQuartic,
        "berry": Berry,
        "pan": Pan,
        "queen": QuEEN,
    }[model_name]
    _, altitudes = grid
    result = simulate(model_class, sun_elevation_deg, grid)

    dop, aop, radiance = result[DOP], result[AOP], result[RADIANCE]
    finite = np.isfinite(dop)

    assert np.all(dop[finite] >= 0.0)
    assert np.all(dop[finite] <= 1.0 + 1e-9)
    assert np.all(aop[np.isfinite(aop)] > -np.pi / 2 - 1e-9)
    assert np.all(aop[np.isfinite(aop)] <= np.pi / 2 + 1e-9)
    assert np.all(radiance[np.isfinite(radiance)] > 0.0)

    expected_mask = np.broadcast_to(altitudes <= 0.0, dop.shape)
    assert np.all(np.isnan(dop[expected_mask]))
    assert np.all(np.isfinite(dop[~expected_mask]))


# --------------------------------------------------------------------------- #
# R6 -- below-horizon radiance
# --------------------------------------------------------------------------- #


def simulate_unclipped(model_class, azimuths, altitudes):
    """Run a model with no altitude mask, so radiance is returned unmasked."""
    sun = SkyCoord(
        az=[SUN_AZIMUTH] * deg,
        alt=[33.0] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )
    model = model_class(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        values = model.simulate_sky(cie_sky_type=4, sun_position=sun, altitude_min_clip=None)
    return values


# Rayleigh and QuEEN cover both families that share SkySimulator._get_radiance.
RADIANCE_MODELS = [Rayleigh, QuEEN]


@pytest.mark.parametrize("model_class", RADIANCE_MODELS)
def test_below_horizon_radiance_is_nan_and_never_infinite(model_class):
    azimuths = np.zeros((1, 5))
    altitudes = np.array([[-45.0, -20.0, -1.0, -1e-9, 30.0]])

    radiance = simulate_unclipped(model_class, azimuths, altitudes)[RADIANCE][0, 0]

    assert not np.isinf(radiance).any()
    assert np.isnan(radiance[:4]).all()
    assert np.isfinite(radiance[4])


@pytest.mark.parametrize("model_class", RADIANCE_MODELS)
def test_horizon_boundary_is_not_moved_by_the_below_horizon_mask(model_class):
    # Exactly 0 degrees must stay finite: cos(zenith) is +6.1e-17 there, so the
    # mask tests cos <= 0 rather than the altitude, and no epsilon may creep in.
    azimuths = np.zeros((1, 3))
    altitudes = np.array([[0.0, 1e-9, -1e-9]])

    radiance = simulate_unclipped(model_class, azimuths, altitudes)[RADIANCE][0, 0]

    assert np.isfinite(radiance[0])
    assert np.isfinite(radiance[1])
    assert np.isnan(radiance[2])


@pytest.mark.parametrize("model_class", RADIANCE_MODELS)
def test_above_horizon_radiance_is_unaffected(model_class):
    azimuths = np.array([[0.0, 90.0, 180.0, 270.0]])
    altitudes = np.array([[5.0, 30.0, 60.0, 89.0]])

    radiance = simulate_unclipped(model_class, azimuths, altitudes)[RADIANCE][0, 0]

    assert np.isfinite(radiance).all()
    assert (radiance > 0.0).all()


@pytest.mark.parametrize("model_class", RADIANCE_MODELS)
def test_below_horizon_directions_raise_no_warning(model_class):
    azimuths = np.zeros((1, 4))
    altitudes = np.array([[-30.0, -5.0, 10.0, 70.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulate_unclipped(model_class, azimuths, altitudes)

    overflow = [item for item in caught if "overflow" in str(item.message)]
    assert overflow == []
