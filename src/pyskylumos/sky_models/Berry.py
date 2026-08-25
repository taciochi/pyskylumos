"""Berry sky polarization model implementation.

Implements the quartic polarization-singularity model of:

    Berry M V, Dennis M R and Lee R L Jr 2004 *Polarization singularities in the
    clear sky* New J. Phys. **6** 162.

The four neutral points are split symmetrically about the sun and anti-sun by a
fixed angular separation. Berry parameterises the split as ``A = tan(delta / 4)``
where ``delta = 4 arctan(A)`` is the *pairwise* separation, so each point sits at
``delta / 2`` from the sun. PySkyLumos uses ``delta = 30`` degrees, the midpoint
of the ranges reported in the literature, placing each neutral point 15 degrees
from the sun.

Unlike :class:`~pyskylumos.sky_models.AsymmetricQuartic.AsymmetricQuartic`,
:class:`~pyskylumos.sky_models.Pan.Pan` and
:class:`~pyskylumos.sky_models.QuEEN.QuEEN`, the separation here does not vary
with solar elevation.

The returned ``degree of polarization`` field is Berry's published ``|omega|``:
the normalized-to-unit-maximum intensity of polarization, which the paper also
calls the unnormalized degree. It is not divided by total daylight intensity.

See the project README mathematical reference, sections 3-5, for the field, the
OpenSky-derived ``e^{-2i alpha_s}`` global AOP convention, and the distinction
between Berry's published intensity and physical DoLP.

Direct ``Berry`` output is evaluated in the fixed world stereographic chart.
Use ``Engine.simulate_sky_polarization`` to transport it into a tilted analyzer
frame.
"""

from astropy.coordinates import SkyCoord
from numpy import absolute, angle, deg2rad, exp, full_like

from pyskylumos._types import ComplexArray, FloatArray
from pyskylumos.sky_models.QuarticSkyModel import QuarticSkyModel


class Berry(QuarticSkyModel):
    """Simulate Berry polarization directly in the fixed world chart."""

    #: Pairwise angular separation between the two neutral points of a pair, in degrees.
    NEUTRAL_POINT_SEPARATION_DEG: float = 30.0

    def _neutral_point_offsets(
        self, sun_altitudes_deg: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return the fixed, symmetric sun-to-neutral-point distances.

        Args:
            sun_altitudes_deg: Solar elevation in degrees. Unused; Berry's
                separation does not depend on solar elevation.

        Returns:
            Tuple of the below-sun (Brewster) and above-sun (Babinet) offsets,
            both in radians and both equal to half the pairwise separation.
        """
        offset: FloatArray = full_like(
            sun_altitudes_deg, deg2rad(self.NEUTRAL_POINT_SEPARATION_DEG / 2), dtype="float64"
        )

        return offset, offset

    def _get_dop(self, field: ComplexArray) -> FloatArray:
        """Return Berry's published normalized polarization intensity.

        Berry 2004, section 4, models ``|omega|`` as the intensity of
        polarization (the unnormalized degree) and normalizes its maximum to
        unity. No project-specific depolarization mapping is applied here.

        Args:
            field: Complex polarization field.

        Returns:
            Published field modulus for each sampled point.
        """
        return absolute(field)

    def _get_aop(
        self,
        field: ComplexArray,
        sun_azimuth: FloatArray,
        observed_point_azimuth: FloatArray,
    ) -> FloatArray:
        """Compute angle of polarization in the fixed world chart.

        The chart's x-axis is world azimuth zero. Camera-pose and analyzer-basis
        transport are intentionally outside this direct-model method.

        Args:
            field: Complex polarization field.
            sun_azimuth: Sun azimuth in radians.
            observed_point_azimuth: Observed point azimuth in radians. Unused;
                Berry's AOP is fixed-world-frame referenced, not
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
        """Return neutral-point positions offset along the solar vertical.

        Args:
            sun_position: Sun position in the simulator's AltAz frame.
            anti_sun_position: Anti-sun position in the simulator's AltAz frame.
            below_sun_projection: Stereographic coordinate of the Brewster point.
                Unused.
            above_sun_projection: Stereographic coordinate of the Babinet point.
                Unused.
            below_sun_offset: Sun-to-Brewster angular distance in radians.
            above_sun_offset: Sun-to-Babinet angular distance in radians.

        Returns:
            Eight arrays of azimuths and elevations in radians, in the order
            documented by :meth:`QuarticSkyModel._singularity_metadata`.
        """
        return self._metadata_from_offsets(
            sun_position=sun_position,
            anti_sun_position=anti_sun_position,
            below_sun_offset=below_sun_offset,
            above_sun_offset=above_sun_offset,
        )
