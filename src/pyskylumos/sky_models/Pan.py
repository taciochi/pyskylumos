"""Pan et al. (2023) sky polarization model with documented interpretations.

This class implements the analytical model of:

    Pan P, Wang X, Yang T, Pu X, Wang W, Bao C and Gao J 2023 *High-similarity
    analytical model of skylight polarization pattern based on position
    variations of neutral points* Opt. Express **31**(9) 15189.

It is **not** the model PySkyLumos shipped under the name ``Pan`` up to version
0.0.6. That model is now :class:`~pyskylumos.sky_models.QuEEN.QuEEN`.

Relative to QuEEN, this class differs in two published definitions:

* **Degree of polarization**, Pan Eq. (14): ``DoP = |omega|``. Berry's
  normalization already bounds this to ``[0, 1]``. QuEEN instead applies a
  depolarization map.
* **Angle of polarization**, Pan Eq. (15) with Eq. (23) and the wrapping of
  Eq. (24): the angle is referenced to the **local meridian at each pixel**,
  not to a world-fixed or sensor analyzer frame.

Because the AOP is meridian-referenced, it is *not* the quantity
:class:`~pyskylumos.sensor.MicroPolarizer.MicroPolarizer` expects, which assumes
an angle measured in the active analyzer frame. The high-level
:meth:`~pyskylumos.engine.Engine.Engine.simulate_sky_polarization` API performs
the local-meridian-to-world conversion automatically and also transports AOP
into the sensor chart when integrated tilt is supplied. When using ``Pan``
directly with a world-aligned, untilted sensor, convert before measurement::

    aop_world = aop_pan + observed_point_azimuth   (mod pi)

That addition alone is not sufficient for a tilted sensor because the
world-to-analyzer correction varies across the image. Use Engine's integrated
tilt path in that case.

Several equations as printed in the paper are ambiguous or inconsistent. Each is
implemented according to a documented decision rather than silently, and the full
errata table is in the project README mathematical reference, section 8. The
decisions that change numbers are:

E5  Eq. (9) text gives ``y_s = r cos(delta_s)``, mixing a stereographic radius
    with an orthographic factor. Implemented as ``y_s = tan(phi_s / 2)`` (Berry
    Eq. 2.6), the only reading under which the Moebius forms place the neutral
    points at their intended angular distances.
E6  Eqs. (9), (11) and (12) print the second root's denominator as
    ``(1 - A_- y_s)``. Implemented as ``(1 + A_- y_s)``, per Berry Eq. (2.6) and
    the tangent subtraction identity.
E7  Eqs. (10) and (13) print the fourth factor as ``(mu + mu*_-)``, which is not
    the antipode of ``mu_-``. Implemented as ``(mu + 1 / mu*_-)``.
E8  Eq. (11) labels ``mu_+`` the Babinet point, but its Moebius form increases the
    zenith angle and so places it *below* the sun. Implemented with the Babinet
    point above the sun at ``delta_+`` and the Brewster point below at
    ``delta_-``; the printed pairing would put Brewster past the zenith.
E9  Eq. (15)'s complex logarithm is evaluated equivalently as ``arg(omega) / 2``.
    The OpenSky-derived phase factor ``e^{-2i alpha_s}`` is then applied because
    the quartic is a product of four root factors: rotating the scene by ``beta``
    otherwise moves the angle by ``2 beta`` rather than ``beta``. See the project
    README mathematical reference, section 4.
E13 Eq. (23) subtracts ``arctan[(y - n/2) / (n/2 - x)]``, which is the complement
    of the azimuth defined by Eq. (20) and so differs from it by 90 degrees.
    Implemented by subtracting the observed point's own azimuth, which is what
    Eq. (23) describes in words.

Pan's camera model (Eqs. 17-20) is a separate stage and is not applied here. It
is the equidistant fisheye mapping, available as
``OpticalConjugator(lens_conjugation_type='equi_angle')``.
"""

from typing import ClassVar
from warnings import warn

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy import absolute, angle, exp, pi

from pyskylumos._types import ComplexArray, FloatArray, RealArray
from pyskylumos.exceptions import ConfigurationError
from pyskylumos.sky_models.NeutralPointOffsets import (
    RANGE_POLICIES,
    pan_offsets,
)
from pyskylumos.sky_models.QuarticSkyModel import QuarticSkyModel


class PanFidelityWarning(UserWarning):
    """Raised once per simulator to flag Pan's meridian-referenced AOP.

    Filterable and escalatable in the usual way, for example with
    ``-W ignore::pyskylumos.sky_models.PanFidelityWarning``.
    """


class Pan(QuarticSkyModel):
    """Simulate documented Pan fields directly in world/local-meridian geometry."""

    #: Pan Eq. (24) references the angle to the local meridian at each sampled
    #: direction, so Engine adds the observed world azimuth before transport.
    AOP_REFERENCE: ClassVar[str] = "local_meridian"

    def __init__(
        self,
        times: Time,
        observation_location: EarthLocation,
        azimuths: RealArray,
        altitudes: RealArray,
        out_of_range: str = "warn",
    ) -> None:
        """Initialize the Pan simulator with observation geometry.

        Args:
            times: Observation times for each simulation step.
            observation_location: Location of the observer on Earth.
            azimuths: Grid of world azimuths in degrees for sky sampling.
            altitudes: Grid of world altitudes in degrees for sky sampling.
            out_of_range: Policy for solar elevations outside the range measured
                by Pan et al.; one of ``'warn'``, ``'raise'`` or ``'ignore'``.

        Raises:
            ValueError: If ``out_of_range`` is not a recognised policy.
        """
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

        self.__out_of_range = out_of_range
        self.__fidelity_warning_issued = False

    def _neutral_point_offsets(
        self, sun_altitudes_deg: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return Pan's fitted sun-to-neutral-point distances, Eqs. (25) and (26).

        Args:
            sun_altitudes_deg: Solar elevation in degrees.

        Returns:
            Tuple of the below-sun (Brewster) and above-sun (Babinet) offsets,
            both in radians.
        """
        return pan_offsets(sun_altitudes_deg, policy=self.__out_of_range)

    def _get_dop(self, field: ComplexArray) -> FloatArray:
        """Compute degree of polarization as Pan Eq. (14), ``DoP = |omega|``.

        Args:
            field: Complex polarization field.

        Returns:
            Degree of polarization for each sampled point.
        """
        return absolute(field)

    def _get_aop(
        self,
        field: ComplexArray,
        sun_azimuth: FloatArray,
        observed_point_azimuth: FloatArray,
    ) -> FloatArray:
        """Compute angle of polarization relative to the local meridian.

        Implements Pan Eq. (15) with the phase correction of errata E9, then the
        meridian referencing of Eq. (23) and the wrapping of Eq. (24)::

            AoP = wrap[ arg(omega e^{-2i alpha_s}) / 2 - alpha_p ]

        The result is the angle between the electric vector and the local
        meridian of each world sampling direction, wrapped onto
        ``[-pi/2, pi/2)``. It is not ready for a tilted analyzer without the
        additional Engine basis transport.

        Args:
            field: Complex polarization field.
            sun_azimuth: Sun azimuth in radians.
            observed_point_azimuth: Observed point azimuth in radians.

        Returns:
            Local-meridian AOP for each sampled point, in radians.
        """
        frame_referenced: FloatArray = 0.5 * angle(field * exp(-2j * sun_azimuth))

        return (frame_referenced - observed_point_azimuth + pi / 2) % pi - pi / 2

    def _singularity_metadata(
        self,
        sun_position: SkyCoord,
        anti_sun_position: SkyCoord,
        below_sun_projection: ComplexArray,
        above_sun_projection: ComplexArray,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> list[FloatArray]:
        """Return neutral-point positions from Pan's own angular definitions.

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

    def simulate_sky(
        self,
        cie_sky_type: int,
        altitude_min_clip: float | None = None,
        accuracy: bool = False,
        sun_position: SkyCoord | None = None,
    ) -> list[FloatArray]:
        """Simulate direct Pan fields at world AltAz sampling directions.

        Args:
            cie_sky_type: CIE sky type index for radiance model.
            altitude_min_clip: Minimum altitude (degrees) to keep; lower values masked.
            accuracy: Whether to use high-accuracy ephemeris for sun position.
            sun_position: Optional explicit sun position to use.

        Returns:
            List of arrays for polarization metrics, radiance, scattering angle,
            and singularity points relative to sun and anti-sun. AOP remains
            local-meridian referenced and has not been converted or transported
            into a camera analyzer frame.
        """
        if not self.__fidelity_warning_issued:
            self.__fidelity_warning_issued = True
            warn(
                "Pan angle of polarization is referenced to the local meridian at each pixel "
                "(Pan Eq. 23), not to the active analyzer frame. "
                "Engine.simulate_sky_polarization converts Pan output automatically and "
                "also handles tilted sensors. For a world-aligned untilted sensor only, "
                "direct users may call Engine.convert_local_meridian_aop_to_sensor or use "
                '"aop_world = aop_pan + observed_point_azimuth (mod pi)" before measurement.',
                PanFidelityWarning,
                stacklevel=2,
            )

        return super().simulate_sky(
            cie_sky_type=cie_sky_type,
            altitude_min_clip=altitude_min_clip,
            accuracy=accuracy,
            sun_position=sun_position,
        )
