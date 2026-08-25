"""Direct world-coordinate Rayleigh sky polarization model."""

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg
from numpy import arctan2, cos, sin

from pyskylumos._types import BoolArray, FloatArray, ParameterNames, RealArray
from pyskylumos.sky_models.SkySimulator import SkySimulator


class Rayleigh(SkySimulator):
    """Simulate Rayleigh polarization directly in the fixed world chart."""

    __PARAMETERS_SIMULATED: ParameterNames = (
        "degree of polarization",
        "angle of polarization",
        "radiance",
        "scattering angle",
        "sun azimuth",
        "sun elevation",
        "anti-sun azimuth",
        "anti-sun elevation",
    )

    def __init__(
        self,
        times: Time,
        observation_location: EarthLocation,
        altitudes: RealArray,
        azimuths: RealArray,
    ) -> None:
        """Initialize the Rayleigh simulator with observation geometry.

        Args:
            times: Observation times for each simulation step.
            observation_location: Location of the observer on Earth.
            altitudes: Grid of world altitudes in degrees for sky sampling.
            azimuths: Grid of world azimuths in degrees for sky sampling.
        """
        normalized_times = self._validate_geometry(times, observation_location, azimuths, altitudes)
        self._sky_map = SkyCoord(
            alt=altitudes * deg,
            az=azimuths * deg,
            frame=AltAz(obstime=normalized_times[:, None, None], location=observation_location),
        )

    @property
    def parameters_simulated(self) -> ParameterNames:
        """Return the list of parameters produced by the simulation.

        Returns:
            Names of sky parameters simulated by this model.
        """
        return self.__PARAMETERS_SIMULATED

    @property
    def sky_map(self) -> SkyCoord:
        """Return the current sky map coordinates.

        Returns:
            The current sky map as an astropy SkyCoord.
        """
        return self._sky_map

    @sky_map.setter
    def sky_map(self, new_sky_map: SkyCoord) -> None:
        """Update the sky map coordinates.

        Args:
            new_sky_map: New sky coordinate grid.
        """
        self._sky_map = new_sky_map

    @staticmethod
    def __get_dop(scattering_angle: FloatArray) -> FloatArray:
        """Compute degree of polarization from the scattering angle.

        Args:
            scattering_angle: Scattering angle between sun and observation point.

        Returns:
            Degree of polarization for each sampled point.
        """
        dop = sin(scattering_angle) ** 2 / (1 + cos(scattering_angle) ** 2)
        return np.asarray(dop, dtype=np.float64)

    def _dop_from_scattering_angle(self, scattering_angle: FloatArray) -> FloatArray:
        """Return the degree of polarization for a scattering angle.

        Ideal single-dipole scattering, ``sin^2(gamma) / (1 + cos^2(gamma))``.
        Subclasses override this to change only the polarization law while
        reusing the geometry, radiance and masking of this class.

        Args:
            scattering_angle: Scattering angle between sun and observation point,
                in radians.

        Returns:
            Degree of polarization for each sampled point.
        """
        return self.__get_dop(scattering_angle=scattering_angle)

    @staticmethod
    def __get_aop(
        observed_point_altitude: FloatArray,
        solar_altitude: FloatArray,
        azimuthal_difference: FloatArray,
        observed_particle_azimuth: FloatArray,
    ) -> FloatArray:
        """Compute fixed-world-chart AOP from scattering geometry.

        The result is referenced to the world azimuth-zero chart axis. Direct
        Rayleigh has no camera-pose information; Engine performs any requested
        transport into a tilted analyzer frame.

        Args:
            observed_point_altitude: Altitude of the observed point in radians.
            solar_altitude: Solar altitude in radians.
            azimuthal_difference: Difference between observed and solar azimuths in radians.
            observed_particle_azimuth: Observed particle azimuth in radians.

        Returns:
            Fixed-world-chart angle of polarization for each sampled point, in
            radians.
        """
        numerator = sin(observed_point_altitude) * cos(solar_altitude) * cos(
            azimuthal_difference
        ) - cos(observed_point_altitude) * sin(solar_altitude)
        denominator = cos(solar_altitude) * sin(azimuthal_difference)
        frame_angle = arctan2(numerator, denominator) + observed_particle_azimuth
        return np.asarray((frame_angle + np.pi / 2) % np.pi - np.pi / 2, dtype=np.float64)

    def simulate_sky(
        self,
        cie_sky_type: int,
        altitude_min_clip: float | None = None,
        accuracy: bool = False,
        sun_position: SkyCoord | None = None,
    ) -> list[FloatArray]:
        """Simulate Rayleigh fields at world AltAz sampling directions.

        Args:
            cie_sky_type: CIE sky type index for radiance model.
            altitude_min_clip: Minimum altitude (degrees) to keep; lower values masked.
            accuracy: Whether to use high-accuracy ephemeris for sun position.
            sun_position: Optional explicit sun position to use.

        Returns:
            List of arrays for degree/angle of polarization, radiance, scattering angle,
            and sun/anti-sun azimuth/elevation values. AOP remains in the fixed
            world chart; no camera-pose transport is applied here.
        """
        sun_position = self._get_sun(accuracy=accuracy, sun_position=sun_position)
        anti_sun_position: SkyCoord = sun_position.directional_offset_by(
            position_angle=0 * deg, separation=180 * deg
        )

        scattering_angle: FloatArray = np.asarray(
            self.sky_map.separation(sun_position).radian, dtype=np.float64
        )

        radiance: FloatArray = np.asarray(
            self._get_radiance(
                cie_sky_type=cie_sky_type,
                observed_point_zenith_angle=(90 * deg - self.sky_map.alt).radian,
                sun_zenith_angle=(90 * deg - sun_position.alt).radian,
                scattering_angle=scattering_angle,
            ),
            dtype=np.float64,
        )

        dop: FloatArray = self._dop_from_scattering_angle(scattering_angle)

        azimuthal_difference: FloatArray = np.asarray(
            (self.sky_map.az - sun_position.az).wrap_at(360 * deg).radian,
            dtype=np.float64,
        )

        aop: FloatArray = self.__get_aop(
            solar_altitude=np.asarray(sun_position.alt.radian, dtype=np.float64),
            observed_point_altitude=np.asarray(self.sky_map.alt.radian, dtype=np.float64),
            azimuthal_difference=azimuthal_difference,
            observed_particle_azimuth=np.asarray(self.sky_map.az.radian, dtype=np.float64),
        )
        aop = np.where(dop <= 1e-15, np.nan, aop)

        if altitude_min_clip is not None:
            mask: BoolArray = np.asarray(self.sky_map.alt.deg <= altitude_min_clip)
            radiance[mask] = np.nan
            dop[mask] = np.nan
            aop[mask] = np.nan
            scattering_angle[mask] = np.nan

        return [
            dop,
            aop,
            radiance,
            scattering_angle,
            np.asarray(sun_position.az.radian, dtype=np.float64),
            np.asarray(sun_position.alt.radian, dtype=np.float64),
            np.asarray(anti_sun_position.az.radian, dtype=np.float64),
            np.asarray(anti_sun_position.alt.radian, dtype=np.float64),
        ]
