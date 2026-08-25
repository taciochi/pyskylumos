"""Solar-elevation-dependent neutral-point offsets fitted by Pan et al. (2023).

The angular distances from the sun to the Babinet point (above the sun) and to
the Brewster point (below the sun) are linear functions of solar elevation,
regressed from measurements taken on 5 August 2022 at Hefei University of
Technology over a solar elevation range of 0 to 64 degrees.

References:
    Pan P, Wang X, Yang T, Pu X, Wang W, Bao C and Gao J 2023 *High-similarity
    analytical model of skylight polarization pattern based on position
    variations of neutral points* Opt. Express **31** 15189 -- Eqs. (25) and (26).

:class:`~pyskylumos.sky_models.AsymmetricQuartic.AsymmetricQuartic`,
:class:`~pyskylumos.sky_models.Pan.Pan` and
:class:`~pyskylumos.sky_models.QuEEN.QuEEN` draw their solar-side offsets from
here, so the fit, its measured range and its documented discontinuity are
defined once.

Two properties of the published fit deserve attention and are deliberately
reproduced rather than smoothed over:

* The Brewster regression is piecewise with a breakpoint at 27 degrees, and the
  two branches disagree there by about 0.48 degrees. See
  :data:`BREWSTER_BREAKPOINT_DISCONTINUITY_DEG`.
* The Babinet regression crosses zero at about 75.95 degrees of solar elevation,
  above which the fitted "Babinet" offset is negative and the point crosses to
  the far side of the sun. See :data:`BABINET_SIGN_CHANGE_DEG`.

Neither condition is clamped or extrapolated silently. Values outside the
measured range are still evaluated -- the linear fits are simply extended -- but
:func:`check_elevation_range` reports them according to the caller's policy.
"""

from typing import Final
from warnings import warn

from numpy import any as np_any
from numpy import asarray, deg2rad, float64, isnan, nanmax, nanmin, where

from pyskylumos._types import FloatArray
from pyskylumos.exceptions import InputValidationError

#: Solar elevation range, in degrees, over which Pan et al. fitted their data.
MEASURED_ELEVATION_RANGE_DEG: Final[tuple[float, float]] = (0.0, 64.0)

#: Solar elevation, in degrees, at which the Brewster regression switches branch.
BREWSTER_BREAKPOINT_DEG: Final[float] = 27.0

#: Babinet regression coefficients, Pan Eq. (26): ``42.53 - 0.56 * elevation``.
BABINET_INTERCEPT_DEG: Final[float] = 42.53
BABINET_SLOPE_DEG_PER_DEG: Final[float] = -0.56

#: Brewster regression coefficients below the breakpoint, Pan Eq. (25).
BREWSTER_LOW_INTERCEPT_DEG: Final[float] = 37.34
BREWSTER_LOW_SLOPE_DEG_PER_DEG: Final[float] = 0.49

#: Brewster regression coefficients above the breakpoint, Pan Eq. (25).
BREWSTER_HIGH_INTERCEPT_DEG: Final[float] = 56.84
BREWSTER_HIGH_SLOPE_DEG_PER_DEG: Final[float] = -0.25

#: Solar elevation, in degrees, at which the fitted Babinet offset reaches zero.
BABINET_SIGN_CHANGE_DEG: Final[float] = -BABINET_INTERCEPT_DEG / BABINET_SLOPE_DEG_PER_DEG

#: Size of the jump in the Brewster offset across the 27 degree breakpoint.
BREWSTER_BREAKPOINT_DISCONTINUITY_DEG: Final[float] = (
    BREWSTER_LOW_INTERCEPT_DEG + BREWSTER_LOW_SLOPE_DEG_PER_DEG * BREWSTER_BREAKPOINT_DEG
) - (BREWSTER_HIGH_INTERCEPT_DEG + BREWSTER_HIGH_SLOPE_DEG_PER_DEG * BREWSTER_BREAKPOINT_DEG)

#: Accepted values for the ``out_of_range`` policy.
RANGE_POLICIES: Final[tuple[str, ...]] = ("warn", "raise", "ignore")


class NeutralPointRangeWarning(UserWarning):
    """Raised when a solar elevation falls outside Pan's measured fitting range.

    Filterable and escalatable in the usual way, for example with
    ``-W error::pyskylumos.sky_models.NeutralPointRangeWarning``.
    """


def babinet_offset_deg(sun_altitudes_deg: FloatArray) -> FloatArray:
    """Return the sun-to-Babinet angular distance for a solar elevation.

    Implements Pan Eq. (26), ``delta_Babinet = 42.53 - 0.56 * elevation``. The
    Babinet point lies above the sun on the solar vertical.

    Args:
        sun_altitudes_deg: Solar elevation in degrees.

    Returns:
        Angular distance from the sun to the Babinet point, in degrees. Negative
        above :data:`BABINET_SIGN_CHANGE_DEG`, where the fit is extrapolated far
        beyond its measured range.
    """
    altitudes = asarray(sun_altitudes_deg, dtype=float64)
    return BABINET_INTERCEPT_DEG + BABINET_SLOPE_DEG_PER_DEG * altitudes


def brewster_offset_deg(sun_altitudes_deg: FloatArray) -> FloatArray:
    """Return the sun-to-Brewster angular distance for a solar elevation.

    Implements Pan Eq. (25)::

        delta_Brewster = 37.34 + 0.49 * elevation    (elevation <= 27)
        delta_Brewster = 56.84 - 0.25 * elevation    (elevation >  27)

    The Brewster point lies below the sun on the solar vertical. The two branches
    disagree by :data:`BREWSTER_BREAKPOINT_DISCONTINUITY_DEG` at the breakpoint;
    this is a property of the published fit and is reproduced as printed, with
    the breakpoint itself taking the lower branch.

    Args:
        sun_altitudes_deg: Solar elevation in degrees.

    Returns:
        Angular distance from the sun to the Brewster point, in degrees.
    """
    altitudes = asarray(sun_altitudes_deg, dtype=float64)
    return where(
        altitudes <= BREWSTER_BREAKPOINT_DEG,
        BREWSTER_LOW_INTERCEPT_DEG + BREWSTER_LOW_SLOPE_DEG_PER_DEG * altitudes,
        BREWSTER_HIGH_INTERCEPT_DEG + BREWSTER_HIGH_SLOPE_DEG_PER_DEG * altitudes,
    )


def pan_offsets(
    sun_altitudes_deg: FloatArray, policy: str = "warn"
) -> tuple[FloatArray, FloatArray]:
    """Return Pan's fitted sun-to-neutral-point distances, Eqs. (25) and (26).

    Args:
        sun_altitudes_deg: Solar elevation in degrees.
        policy: Out-of-range policy, one of :data:`RANGE_POLICIES`.

    Returns:
        Tuple of the below-sun (Brewster) and above-sun (Babinet) offsets, in radians.
    """
    check_elevation_range(sun_altitudes_deg, policy=policy)

    return (
        deg2rad(brewster_offset_deg(sun_altitudes_deg)),
        deg2rad(babinet_offset_deg(sun_altitudes_deg)),
    )


def check_elevation_range(sun_altitudes_deg: FloatArray, policy: str = "warn") -> None:
    """Report solar elevations that fall outside Pan's measured fitting range.

    This function never clamps and never modifies the offsets. Outside the
    measured range the linear fits are simply extended, which is extrapolation;
    the caller is told so.

    Two independent conditions are reported. The first is any elevation outside
    :data:`MEASURED_ELEVATION_RANGE_DEG`. The second is any elevation above
    :data:`BABINET_SIGN_CHANGE_DEG`, where the fitted Babinet offset turns
    negative and the point's identity as the *Babinet* point stops being
    meaningful.

    Args:
        sun_altitudes_deg: Solar elevations in degrees. NaN entries are ignored.
        policy: ``'warn'`` to emit :class:`NeutralPointRangeWarning`, ``'raise'``
            to raise :class:`ValueError`, or ``'ignore'`` to stay silent.

    Raises:
        ValueError: If ``policy`` is not recognised, or if ``policy='raise'`` and
            an elevation falls outside the measured range.
    """
    if policy not in RANGE_POLICIES:
        raise InputValidationError(f"out_of_range must be one of {RANGE_POLICIES}, got {policy!r}.")

    if policy == "ignore":
        return

    altitudes = asarray(sun_altitudes_deg, dtype=float64)
    finite = ~isnan(altitudes)
    if not np_any(finite):
        return

    minimum_deg, maximum_deg = MEASURED_ELEVATION_RANGE_DEG
    outside = finite & ((altitudes < minimum_deg) | (altitudes > maximum_deg))

    if np_any(outside):
        _report(
            f"Solar elevation outside the range measured by Pan et al. (2023), "
            f"[{minimum_deg:g}, {maximum_deg:g}] degrees: observed "
            f"[{nanmin(altitudes[outside]):.4g}, {nanmax(altitudes[outside]):.4g}] degrees. "
            f"The fitted neutral-point offsets are extrapolated, not clamped.",
            policy,
        )

    negative_babinet = finite & (altitudes > BABINET_SIGN_CHANGE_DEG)

    if np_any(negative_babinet):
        _report(
            f"Solar elevation above {BABINET_SIGN_CHANGE_DEG:.4f} degrees, where Pan Eq. (26) "
            f"gives a negative Babinet offset: observed up to "
            f"{nanmax(altitudes[negative_babinet]):.4g} degrees. The Babinet point crosses to "
            f"the far side of the sun and its label is no longer meaningful.",
            policy,
        )


def _report(message: str, policy: str) -> None:
    """Emit or raise an out-of-range report according to the policy.

    Args:
        message: Description of the offending condition.
        policy: Either ``'warn'`` or ``'raise'``.

    Raises:
        ValueError: If ``policy`` is ``'raise'``.
    """
    if policy == "raise":
        raise InputValidationError(message)
    warn(message, NeutralPointRangeWarning, stacklevel=3)
