"""Abstract base class for world-coordinate sky polarization simulators."""

from abc import ABC, abstractmethod
from typing import ClassVar
from urllib.error import URLError

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from numpy import cos, exp, pi

from pyskylumos._types import FloatArray, ParameterNames, RealArray
from pyskylumos._validation import require_real_array, require_same_shape
from pyskylumos.exceptions import InputTypeError, InputValidationError


class SkySimulator(ABC):
    """Provide shared utilities for direct world-coordinate sky models.

    Concrete simulators interpret their azimuth and altitude grids as world
    AltAz directions. Camera-pose handling and transport into an analyzer frame
    belong to :class:`pyskylumos.engine.Engine.Engine`, not to direct models.
    """

    #: Reference axis of the angle of polarization returned by
    #: :meth:`simulate_sky`, declared so that
    #: :class:`~pyskylumos.engine.Engine.Engine` can transport it without
    #: testing for concrete model types.
    #:
    #: ``"fixed_world"`` means the angle is measured from the world azimuth-zero
    #: chart axis and is ready for analyzer transport as returned.
    #: ``"local_meridian"`` means it is measured from the local meridian at each
    #: sampled direction, so Engine must add the observed world azimuth first.
    AOP_REFERENCE: ClassVar[str] = "fixed_world"

    __cie_sky_types: ClassVar[dict[int, dict[str, float]]] = {
        1: {"A": 4.0, "B": -0.7, "C": 0.0, "D": -1.0, "E": 0.0},
        2: {"A": 4.0, "B": -0.7, "C": 2.0, "D": -1.5, "E": 0.15},
        3: {"A": 1.1, "B": -0.8, "C": 0.0, "D": -1.0, "E": 0.0},
        4: {"A": 1.1, "B": -0.8, "C": 2.0, "D": -1.5, "E": 0.15},
        5: {"A": 0.0, "B": -1.0, "C": 0.0, "D": -1.0, "E": 0.0},
        6: {"A": 0.0, "B": -1.0, "C": 2.0, "D": -1.5, "E": 0.15},
        7: {"A": 0.0, "B": -1.0, "C": 5.0, "D": -2.5, "E": 0.3},
        8: {"A": 0.0, "B": -1.0, "C": 10.0, "D": -3.0, "E": 0.45},
        9: {"A": -1.0, "B": -0.55, "C": 2.0, "D": -1.5, "E": 0.15},
        10: {"A": -1.0, "B": -0.55, "C": 5.0, "D": -2.5, "E": 0.3},
        11: {"A": -1.0, "B": -0.55, "C": 10.0, "D": -3.0, "E": 0.45},
        12: {"A": -1.0, "B": -0.32, "C": 10.0, "D": -3.0, "E": 0.45},
        13: {"A": -1.0, "B": -0.32, "C": 16.0, "D": -3.0, "E": 0.3},
        14: {"A": -1.0, "B": -0.15, "C": 16.0, "D": -3.0, "E": 0.3},
        15: {"A": -1.0, "B": -0.15, "C": 24.0, "D": -2.8, "E": 0.15},
    }

    @abstractmethod
    def __init__(
        self,
        times: Time,
        observation_location: EarthLocation,
        azimuths: RealArray,
        altitudes: RealArray,
    ) -> None:
        """Initialize a simulator with world AltAz observation geometry.

        Args:
            times: Scalar or one-dimensional observation times.
            observation_location: Earth location defining the AltAz frame.
            azimuths: Two-dimensional world azimuth grid in degrees.
            altitudes: Two-dimensional world altitude grid in degrees.
        """
        ...

    @property
    @abstractmethod
    def parameters_simulated(self) -> ParameterNames:
        """Return the list of parameter names produced by the simulator.

        Returns:
            Names of sky parameters simulated by this model.
        """
        ...

    @property
    @abstractmethod
    def sky_map(self) -> SkyCoord:
        """Return the current sky map coordinates.

        Returns:
            The current sky map as an astropy SkyCoord.
        """
        ...

    @sky_map.setter
    @abstractmethod
    def sky_map(self, new_sky_map: SkyCoord) -> None:
        """Update the sky map coordinates.

        Args:
            new_sky_map: New sky coordinate grid.
        """
        ...

    @abstractmethod
    def simulate_sky(
        self,
        cie_sky_type: int,
        altitude_min_clip: float | None = None,
        accuracy: bool = False,
        sun_position: SkyCoord | None = None,
    ) -> list[FloatArray]:
        """Simulate polarization and radiance in the model's world frame.

        Args:
            cie_sky_type: CIE sky type index for radiance model.
            altitude_min_clip: Minimum altitude (degrees) to keep; lower values masked.
            accuracy: Whether to use high-accuracy ephemeris for sun position.
            sun_position: Optional explicit sun position to use.

        Returns:
            Arrays named by :attr:`parameters_simulated`. Direct-model AOP uses
            the model's documented world or local-meridian reference and has
            not been transported into a tilted sensor frame.
        """
        ...

    @staticmethod
    def _normalise_times(times: Time) -> Time:
        """Return the observation times as an array-like Time.

        The simulators broadcast times against the sampling grid with
        ``times[:, None, None]``, which a scalar Time cannot support. A scalar is
        promoted to shape ``(1,)`` so that a single observation still yields a
        leading time axis.

        Note that ``times[..., None, None]`` is not an alternative: on a scalar it
        produces shape ``(1, 1)``, which broadcasts against the grid and silently
        drops the time axis.

        Args:
            times: Observation times, scalar or array-like.

        Returns:
            Array-like observation times.

        Raises:
            TypeError: If ``times`` is not an astropy Time.
        """
        if not isinstance(times, Time):
            raise InputTypeError(f"times must be an astropy.time.Time, got {type(times).__name__}.")
        if not times.isscalar and times.ndim != 1:
            raise InputValidationError(
                f"times must be scalar or one-dimensional; got shape {times.shape}."
            )

        return times.reshape(1) if times.isscalar else times

    @classmethod
    def _validate_geometry(
        cls,
        times: Time,
        observation_location: EarthLocation,
        azimuths: RealArray,
        altitudes: RealArray,
    ) -> Time:
        """Validate world AltAz geometry and return normalized observation times.

        Args:
            times: Scalar or one-dimensional observation times.
            observation_location: Earth location defining the AltAz frame.
            azimuths: Two-dimensional world azimuth grid in degrees.
            altitudes: Two-dimensional world altitude grid in degrees.

        Returns:
            Observation times normalized to a one-dimensional array.
        """
        normalized_times = cls._normalise_times(times)
        if not isinstance(observation_location, EarthLocation):
            raise InputTypeError(
                "observation_location must be an astropy.coordinates.EarthLocation."
            )
        azimuths = require_real_array("azimuths", azimuths, rank=2)
        altitudes = require_real_array("altitudes", altitudes, rank=2)
        require_same_shape(azimuths=azimuths, altitudes=altitudes)
        if np.isinf(azimuths).any() or np.isinf(altitudes).any():
            raise InputValidationError("azimuths and altitudes must not contain infinity.")
        finite_altitudes = altitudes[np.isfinite(altitudes)]
        if ((finite_altitudes < -90.0) | (finite_altitudes > 90.0)).any():
            raise InputValidationError("finite altitudes must lie in [-90, 90] degrees.")
        return normalized_times

    def _get_sun(self, accuracy: bool = False, sun_position: SkyCoord | None = None) -> SkyCoord:
        """Return the sun position for the simulation frame.

        Args:
            accuracy: Whether to use the optional JPL DE430 ephemeris for higher
                accuracy. Requires installation of ``pyskylumos[jpl]`` and a
                downloaded or pre-cached DE430 kernel.
            sun_position: Optional explicit sun position. This takes precedence
                over ``accuracy`` and bypasses ephemeris lookup.

        Returns:
            Sun position in the simulator's AltAz frame.
        """
        if not isinstance(accuracy, bool):
            raise InputTypeError("accuracy must be a bool.")
        obstime = self.sky_map.obstime[..., 0, 0]
        frame = AltAz(obstime=obstime, location=self.sky_map.location)

        if sun_position is not None:
            if not isinstance(sun_position, SkyCoord):
                raise InputTypeError(
                    "sun_position must be an astropy.coordinates.SkyCoord or None."
                )
            sun_position = sun_position.transform_to(frame)
            if sun_position.shape == obstime.shape:
                return sun_position[..., None, None]
            return sun_position

        if accuracy:
            try:
                sun = get_body("sun", obstime, ephemeris="de430")
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    "accuracy=True requires the optional JPL ephemeris dependency. "
                    'Install it with `python -m pip install "pyskylumos[jpl]"`.'
                ) from error
            except (URLError, TimeoutError) as error:
                raise RuntimeError(
                    "The JPL DE430 ephemeris kernel is unavailable. Astropy downloads and "
                    "caches it on first use; enable network access or pre-populate the Astropy "
                    "download cache before using accuracy=True."
                ) from error

            return sun.transform_to(frame)[..., None, None]

        return get_sun(obstime).transform_to(frame)[..., None, None]

    @classmethod
    def _get_radiance(
        cls,
        cie_sky_type: int,
        observed_point_zenith_angle: FloatArray,
        sun_zenith_angle: FloatArray,
        scattering_angle: FloatArray,
    ) -> FloatArray:
        """Compute CIE sky radiance for the provided geometry.

        Args:
            cie_sky_type: CIE sky type index for radiance model.
            observed_point_zenith_angle: Zenith angle of the observed point (radians).
            sun_zenith_angle: Zenith angle of the sun (radians).
            scattering_angle: Scattering angle between sun and observation point (radians).

        Returns:
            Radiance value for each sampled point. Directions below the horizon
            carry no sky radiance and return NaN: the CIE luminance gradation
            term is defined for an upward hemisphere, and continuing it past the
            horizon makes ``exp(B / cos(zenith))`` diverge rather than describe
            anything physical.
        """
        if isinstance(cie_sky_type, bool) or not isinstance(cie_sky_type, (int, np.integer)):
            raise InputTypeError("cie_sky_type must be an integer from 1 to 15.")
        if int(cie_sky_type) not in cls.__cie_sky_types:
            raise InputValidationError(
                f"cie_sky_type must be an integer from 1 to 15; got {cie_sky_type!r}."
            )
        radiance_parameters: dict[str, float] = cls.__cie_sky_types[int(cie_sky_type)]
        half_pi: float = pi / 2

        # Every tabulated B is negative, so a below-horizon direction flips the
        # exponent positive and the gradation term diverges. Those samples are
        # masked rather than clamped: a direction with no sky above it has no
        # radiance, and substituting the horizon's value would report a
        # measurement where there is none.
        cosine_zenith = cos(observed_point_zenith_angle)
        below_horizon = cosine_zenith <= 0.0

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            radiance = (
                (1 + radiance_parameters["A"] * exp(radiance_parameters["B"] / cosine_zenith))
                / (1 + radiance_parameters["A"] * exp(radiance_parameters["B"]))
            ) * (
                (
                    1
                    + radiance_parameters["C"]
                    * (
                        exp(radiance_parameters["D"] * scattering_angle)
                        - exp(radiance_parameters["D"] * half_pi)
                    )
                    + radiance_parameters["E"] * cos(scattering_angle) ** 2
                )
                / (
                    1
                    + radiance_parameters["C"]
                    * (
                        exp(radiance_parameters["D"] * sun_zenith_angle)
                        - exp(radiance_parameters["D"] * half_pi)
                    )
                    + radiance_parameters["E"] * cos(sun_zenith_angle) ** 2
                )
            )
        return np.asarray(np.where(below_horizon, np.nan, radiance), dtype=np.float64)
