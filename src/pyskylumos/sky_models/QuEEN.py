"""QuEEN sky polarization model implementation.

QuEEN -- **Qu**artic **E**levation **E**xplained **N**eutralities -- is a
Pan-informed extension of Berry's quartic polarization-singularity model. It is
**not** a reproduction of Pan et al. (2023); for that, see
:class:`~pyskylumos.sky_models.Pan.Pan`.

The construction is:

* Berry's zenith-centred stereographic coordinate ``xi = tan(phi / 2) e^{i alpha}``.
* The solar-side Brewster and Babinet roots placed at Pan's empirically fitted,
  solar-elevation-dependent angular offsets, through the exact stereographic
  half-angle mapping.
* The anti-solar roots as the reciprocal-conjugate antipodes ``-1 / conj(xi)``.
* Berry's normalized quartic field, Eq. (4.2), including its leading ``-4``.

Two OpenSky conventions complete the hybrid. See the project README mathematical
reference, sections 4 and 5.

**Global AOP convention.** OpenSky evaluates
``arg(omega * exp(-2i alpha_s)) / 2``. The quartic is a product of four root
factors, so rigidly rotating the scene by ``beta`` otherwise moves
``arg(omega) / 2`` by ``2 beta`` -- twice the physical rotation. The phase factor
restores covariance and makes QuEEN's AOP agree with Rayleigh in the
small-splitting limit.

**OpenSky intensity-to-DoLP conversion.** Berry normalizes
``max|omega| = 1`` and treats ``|omega|`` as polarization intensity; Pan Eq. (14)
does the same. OpenSky maps this to
``DOP = dop_max |omega| / (2 - |omega|)``, which is ``P / (P + U)`` for
``P = |omega|`` and ``U = 2 (1 - |omega|)``. The factor 2 follows from the two
orthogonal components of Rayleigh's unpolarized contribution. Applying that
conversion to the full quartic is OpenSky's modelling extension, adopted here.

Known limitation, inherited from Berry: Eq. (4.2) places the polarization maxima
on the horizon, whereas observations show none there (Berry 2004, section 4, which
proposes an ad-hoc horizon function ``h(r)`` as a remedy). The ``|omega|/(2-|omega|)``
map does not correct this, because it depends on ``|omega|`` alone and not on
position. Use ``altitude_min_clip`` to exclude the affected band.

Direct ``QuEEN`` output is evaluated in the fixed world stereographic chart.
Use ``Engine.simulate_sky_polarization`` to transport it into a tilted analyzer
frame.

References:
    Berry M V, Dennis M R and Lee R L Jr 2004 New J. Phys. **6** 162.

    Pan P et al. 2023 Opt. Express **31** 15189 -- Eqs. (25) and (26) only.

    Moutenet A et al. 2024 IEEE Trans. Instrum. Meas. **73** -- global AOP
    convention and intensity-to-DoLP conversion.
"""

from math import isfinite
from numbers import Real
from typing import Any

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy import absolute, angle, exp

from pyskylumos._types import ComplexArray, FloatArray, RealArray
from pyskylumos.exceptions import ConfigurationError, InputTypeError
from pyskylumos.sky_models.NeutralPointOffsets import (
    RANGE_POLICIES,
    pan_offsets,
)
from pyskylumos.sky_models.QuarticSkyModel import QuarticSkyModel


class QuEEN(QuarticSkyModel):
    """Simulate QuEEN polarization directly in the fixed world chart."""

    __dop_max: float
    __out_of_range: str

    def __init__(
        self,
        times: Time,
        observation_location: EarthLocation,
        azimuths: RealArray,
        altitudes: RealArray,
        dop_max: Any = 1.0,
        out_of_range: str = "warn",
    ) -> None:
        """Initialize the QuEEN simulator with observation geometry.

        Args:
            times: Observation times for each simulation step.
            observation_location: Location of the observer on Earth.
            azimuths: Grid of world azimuths in degrees for sky sampling.
            altitudes: Grid of world altitudes in degrees for sky sampling.
            dop_max: Peak degree of polarization, on (0, 1].
            out_of_range: Policy for solar elevations outside the range measured
                by Pan et al.; one of ``'warn'``, ``'raise'`` or ``'ignore'``.

        Raises:
            ValueError: If ``dop_max`` is outside (0, 1], or ``out_of_range`` is
                not a recognised policy.
        """
        if isinstance(dop_max, bool) or not isinstance(dop_max, Real):
            raise InputTypeError("dop_max must be a float-valued finite real number.")
        dop_max = float(dop_max)
        if not isfinite(dop_max) or not 0 < dop_max <= 1:
            raise ConfigurationError(f"dop_max must lie in (0, 1] and be finite, got {dop_max!r}.")
        if out_of_range not in RANGE_POLICIES:
            raise ConfigurationError(
                f"out_of_range must be one of {RANGE_POLICIES}, got {out_of_range!r}."
            )

        super().__init__(
            times=times,
            observation_location=observation_location,
            azimuths=azimuths,
            altitudes=altitudes,
        )

        self.__dop_max = dop_max
        self.__out_of_range = out_of_range

    @property
    def dop_max(self) -> float:
        """Return the configured peak degree of polarization.

        Returns:
            Peak degree of polarization.
        """
        return self.__dop_max

    def _neutral_point_offsets(
        self, sun_altitudes_deg: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return Pan's fitted sun-to-neutral-point distances.

        Args:
            sun_altitudes_deg: Solar elevation in degrees.

        Returns:
            Tuple of the below-sun (Brewster) and above-sun (Babinet) offsets,
            both in radians.
        """
        return pan_offsets(sun_altitudes_deg, policy=self.__out_of_range)

    def _get_dop(self, field: ComplexArray) -> FloatArray:
        """Compute degree of polarization under QuEEN's depolarization policy.

        Applies ``DOP = dop_max |omega| / (2 - |omega|)``. Berry's normalization
        guarantees ``|omega| <= 1``, so the result stays on ``[0, dop_max]``.

        Args:
            field: Complex polarization field.

        Returns:
            Degree of polarization for each sampled point.
        """
        modulus: FloatArray = absolute(field)

        return self.__dop_max * modulus / (2 - modulus)

    def _get_aop(
        self,
        field: ComplexArray,
        sun_azimuth: FloatArray,
        observed_point_azimuth: FloatArray,
    ) -> FloatArray:
        """Compute angle of polarization in the fixed world chart.

        Applies the half-phase relationship to the compensated field,
        ``AOP = arg(omega e^{-2i alpha_s}) / 2``. The result is measured from the
        world chart's x-axis, which is azimuth zero, and lies on
        ``(-pi/2, pi/2]``. Engine may subsequently transport this orientation
        into a tilted analyzer chart.

        Args:
            field: Complex polarization field.
            sun_azimuth: Sun azimuth in radians.
            observed_point_azimuth: Observed point azimuth in radians. Unused;
                QuEEN's AOP is fixed-world-frame referenced, not
                meridian-referenced.

        Returns:
            Fixed-world-chart AOP for each sampled point, in radians.
        """
        return 0.5 * angle(field * exp(-2j * sun_azimuth))

    def _singularity_metadata(
        self,
        sun_position: SkyCoord,
        anti_sun_position: SkyCoord,
        below_sun_projection: ComplexArray,
        above_sun_projection: ComplexArray,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> list[FloatArray]:
        """Return neutral-point positions taken from the quartic's own roots.

        Args:
            sun_position: Sun position in the simulator's AltAz frame. Unused.
            anti_sun_position: Anti-sun position. Unused.
            below_sun_projection: Stereographic coordinate of the Brewster point.
            above_sun_projection: Stereographic coordinate of the Babinet point.
            below_sun_offset: Sun-to-Brewster angular distance in radians. Unused.
            above_sun_offset: Sun-to-Babinet angular distance in radians. Unused.

        Returns:
            Eight arrays of azimuths and elevations in radians, in the order
            documented by :meth:`QuarticSkyModel._singularity_metadata`.
        """
        return self._metadata_from_roots(
            below_sun_projection=below_sun_projection, above_sun_projection=above_sun_projection
        )
