"""Tests for the Pan (2023) neutral-point offset regressions and range policy."""

import warnings

import numpy as np
import pytest

from pyskylumos.sky_models.NeutralPointOffsets import (
    BABINET_SIGN_CHANGE_DEG,
    BREWSTER_BREAKPOINT_DEG,
    BREWSTER_BREAKPOINT_DISCONTINUITY_DEG,
    MEASURED_ELEVATION_RANGE_DEG,
    NeutralPointRangeWarning,
    babinet_offset_deg,
    brewster_offset_deg,
    check_elevation_range,
)

# --------------------------------------------------------------------------- #
# Pan Eq. (26) -- Babinet
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "elevation_deg, expected_deg",
    [(0.0, 42.53), (27.0, 42.53 - 0.56 * 27.0), (60.0, 8.93), (64.0, 42.53 - 0.56 * 64.0)],
)
def test_babinet_offset_matches_published_fit(elevation_deg, expected_deg):
    assert babinet_offset_deg(elevation_deg) == pytest.approx(expected_deg, abs=1e-12)


def test_babinet_offset_crosses_zero_at_the_documented_elevation():
    assert pytest.approx(75.946428571428571, abs=1e-9) == BABINET_SIGN_CHANGE_DEG
    assert babinet_offset_deg(BABINET_SIGN_CHANGE_DEG) == pytest.approx(0.0, abs=1e-12)
    assert babinet_offset_deg(BABINET_SIGN_CHANGE_DEG + 1.0) < 0.0


# --------------------------------------------------------------------------- #
# Pan Eq. (25) -- Brewster, including the 27 degree breakpoint
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "elevation_deg, expected_deg",
    [
        (0.0, 37.34),
        (20.0, 37.34 + 0.49 * 20.0),
        (27.0, 50.57),
        (30.0, 56.84 - 0.25 * 30.0),
        (64.0, 56.84 - 0.25 * 64.0),
    ],
)
def test_brewster_offset_matches_published_fit(elevation_deg, expected_deg):
    assert brewster_offset_deg(elevation_deg) == pytest.approx(expected_deg, abs=1e-12)


def test_brewster_breakpoint_takes_the_lower_branch():
    assert brewster_offset_deg(BREWSTER_BREAKPOINT_DEG) == pytest.approx(50.57, abs=1e-12)


def test_brewster_breakpoint_discontinuity_is_reproduced_as_published():
    just_below = brewster_offset_deg(BREWSTER_BREAKPOINT_DEG - 1e-9)
    just_above = brewster_offset_deg(BREWSTER_BREAKPOINT_DEG + 1e-9)

    assert just_below == pytest.approx(50.57, abs=1e-8)
    assert just_above == pytest.approx(56.84 - 0.25 * 27.0, abs=1e-8)
    assert just_below - just_above == pytest.approx(BREWSTER_BREAKPOINT_DISCONTINUITY_DEG, abs=1e-8)
    # 50.57 - 50.09 exactly; the branches disagree by 0.48 degrees at the breakpoint.
    assert pytest.approx(0.48, abs=1e-9) == BREWSTER_BREAKPOINT_DISCONTINUITY_DEG


# --------------------------------------------------------------------------- #
# vectorisation
# --------------------------------------------------------------------------- #


def test_offsets_are_vectorised_and_shape_preserving():
    elevations = np.linspace(-10.0, 90.0, 24).reshape(2, 3, 4)

    assert babinet_offset_deg(elevations).shape == elevations.shape
    assert brewster_offset_deg(elevations).shape == elevations.shape


def test_offsets_accept_scalars():
    assert np.ndim(babinet_offset_deg(30.0)) == 0
    assert np.ndim(brewster_offset_deg(30.0)) == 0


# --------------------------------------------------------------------------- #
# range policy -- never clamps
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("elevation_deg", [0.0, 27.0, 32.0, 64.0])
def test_no_warning_inside_the_measured_range(elevation_deg):
    with warnings.catch_warnings():
        warnings.simplefilter("error", NeutralPointRangeWarning)
        check_elevation_range(np.array([elevation_deg]))


@pytest.mark.parametrize("elevation_deg", [-5.0, 70.0])
def test_warns_outside_the_measured_range(elevation_deg):
    with pytest.warns(NeutralPointRangeWarning, match="outside the range measured"):
        check_elevation_range(np.array([elevation_deg]))


@pytest.mark.parametrize("elevation_deg", [-5.0, 70.0, 80.0])
def test_out_of_range_offsets_are_extrapolated_never_clamped(elevation_deg):
    minimum_deg, maximum_deg = MEASURED_ELEVATION_RANGE_DEG

    assert babinet_offset_deg(elevation_deg) == pytest.approx(
        42.53 - 0.56 * elevation_deg, abs=1e-12
    )
    if elevation_deg <= BREWSTER_BREAKPOINT_DEG:
        assert brewster_offset_deg(elevation_deg) == pytest.approx(
            37.34 + 0.49 * elevation_deg, abs=1e-12
        )
    else:
        assert brewster_offset_deg(elevation_deg) == pytest.approx(
            56.84 - 0.25 * elevation_deg, abs=1e-12
        )

    clamped = np.clip(elevation_deg, minimum_deg, maximum_deg)
    assert babinet_offset_deg(elevation_deg) != pytest.approx(babinet_offset_deg(clamped), abs=1e-9)


def test_negative_babinet_elevation_emits_a_second_distinct_warning():
    with pytest.warns(NeutralPointRangeWarning) as records:
        check_elevation_range(np.array([80.0]))

    messages = [str(record.message) for record in records]
    assert len(messages) == 2
    assert any("outside the range measured" in message for message in messages)
    assert any("negative Babinet offset" in message for message in messages)


def test_elevation_just_out_of_range_does_not_trigger_the_babinet_warning():
    with pytest.warns(NeutralPointRangeWarning) as records:
        check_elevation_range(np.array([70.0]))

    messages = [str(record.message) for record in records]
    assert len(messages) == 1
    assert "negative Babinet offset" not in messages[0]


def test_raise_policy_raises():
    with pytest.raises(ValueError, match="outside the range measured"):
        check_elevation_range(np.array([70.0]), policy="raise")


def test_ignore_policy_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", NeutralPointRangeWarning)
        check_elevation_range(np.array([-30.0, 80.0]), policy="ignore")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="out_of_range must be one of"):
        check_elevation_range(np.array([30.0]), policy="clamp")


def test_nan_elevations_are_ignored():
    with warnings.catch_warnings():
        warnings.simplefilter("error", NeutralPointRangeWarning)
        check_elevation_range(np.array([np.nan, 30.0]))


def test_one_call_reports_once_regardless_of_array_size():
    with pytest.warns(NeutralPointRangeWarning) as records:
        check_elevation_range(np.full(500, 70.0))

    assert len(records) == 1
