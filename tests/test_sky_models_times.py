"""Tests for observation-time handling across every sky model.

A scalar ``Time`` used to raise ``AttributeError`` inside every model
constructor, because the simulators broadcast with ``times[:, None, None]``.
Scalars are now promoted to shape ``(1,)``.
"""

import warnings

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

LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
TIMESTAMP = "2026-06-21T12:00:00"
MODELS = [Rayleigh, DepolarizedRayleigh, AsymmetricQuartic, Berry, Pan, QuEEN]


def make_grid(rows=5, columns=7):
    """Return a small sampling grid."""
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns), (rows, 1))
    altitudes = np.tile(np.linspace(10.0, 80.0, rows)[:, None], (1, columns))
    return azimuths, altitudes


def simulate(model_class, times):
    """Run a model for a fixed sun position, suppressing advisory warnings."""
    azimuths, altitudes = make_grid()
    model = model_class(
        times=times,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
    )
    sun = SkyCoord(
        az=[137.0] * deg,
        alt=[33.0] * deg,
        frame=AltAz(obstime=Time([TIMESTAMP]), location=LOCATION),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        return model.simulate_sky(cie_sky_type=4, sun_position=sun)


@pytest.mark.parametrize("model_class", MODELS, ids=lambda c: c.__name__)
def test_scalar_time_is_accepted(model_class):
    result = simulate(model_class, Time(TIMESTAMP))

    assert result[0].shape == (1, 5, 7)


@pytest.mark.parametrize("model_class", MODELS, ids=lambda c: c.__name__)
def test_scalar_time_matches_a_one_element_time(model_class):
    scalar = simulate(model_class, Time(TIMESTAMP))
    sequence = simulate(model_class, Time([TIMESTAMP]))

    for from_scalar, from_sequence in zip(scalar, sequence, strict=False):
        np.testing.assert_allclose(from_scalar, from_sequence, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("model_class", MODELS, ids=lambda c: c.__name__)
def test_multi_element_time_keeps_its_leading_axis(model_class):
    times = Time([TIMESTAMP, "2026-06-21T13:00:00", "2026-06-21T14:00:00"])
    azimuths, altitudes = make_grid()
    model = model_class(
        times=times,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
    )
    sun = SkyCoord(
        az=[137.0, 150.0, 163.0] * deg,
        alt=[33.0, 30.0, 25.0] * deg,
        frame=AltAz(obstime=times, location=LOCATION),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        result = model.simulate_sky(cie_sky_type=4, sun_position=sun)

    assert result[0].shape == (3, 5, 7)
    assert result[4].shape == (3, 1, 1)


@pytest.mark.parametrize("model_class", MODELS, ids=lambda c: c.__name__)
def test_non_time_arguments_are_rejected(model_class):
    azimuths, altitudes = make_grid()

    with pytest.raises(TypeError, match=r"times must be an astropy\.time\.Time"):
        model_class(
            times=TIMESTAMP,
            observation_location=LOCATION,
            azimuths=azimuths,
            altitudes=altitudes,
        )


def test_a_scalar_time_keeps_a_leading_time_axis():
    """Guards against the tempting but wrong `times[..., None, None]` fix.

    That form yields shape (1, 1) for a scalar, which broadcasts against the
    sampling grid and silently drops the time axis.
    """
    result = simulate(QuEEN, Time(TIMESTAMP))

    assert result[0].ndim == 3
    assert result[0].shape[0] == 1
    assert result[4].shape == (1, 1, 1)
