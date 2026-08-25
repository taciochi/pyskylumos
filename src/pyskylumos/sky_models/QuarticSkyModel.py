"""Shared scaffolding for sky models built on Berry's quartic singularity field.

This class holds only the parts that
:class:`~pyskylumos.sky_models.AsymmetricQuartic.AsymmetricQuartic`,
:class:`~pyskylumos.sky_models.Berry.Berry`,
:class:`~pyskylumos.sky_models.Pan.Pan` and
:class:`~pyskylumos.sky_models.QuEEN.QuEEN` have in common: observation geometry,
the parameter schema, horizon masking, and the orchestration of a simulation run.

Every equation that distinguishes one model from another -- the neutral-point
offsets, the angle of polarization and the degree of polarization -- stays in the
concrete model's own file, so the physics remains readable model by model.
See the project README mathematical reference, section 7, for the formula sheet.

This is a direct world-coordinate model layer. It does not know camera pose or
the analyzer basis; Engine performs that transport when integrated tilt is used.
"""

from abc import abstractmethod

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg, rad
from numpy import pi

from pyskylumos._types import BoolArray, ComplexArray, FloatArray, ParameterNames, RealArray
from pyskylumos.sky_models.SkySimulator import SkySimulator
from pyskylumos.sky_models.StereographicQuartic import (
    antipode,
    offset_along_solar_vertical,
    omega,
    project,
    unproject,
)

_TWO_PI: float = 2 * pi


class QuarticSkyModel(SkySimulator):
    """Base class for world-coordinate Berry-quartic sky models."""

    __PARAMETERS_SIMULATED: ParameterNames = (
        "degree of polarization",
        "angle of polarization",
        "radiance",
        "scattering angle",
        "sun azimuth",
        "sun elevation",
        "above sun singularity point azimuth",
        "above sun singularity point elevation",
        "below sun singularity point azimuth",
        "below sun singularity point elevation",
        "anti-sun azimuth",
        "anti-sun elevation",
        "above anti-sun singularity point azimuth",
        "above anti-sun singularity point elevation",
        "below anti-sun singularity point azimuth",
        "below anti-sun singularity point elevation",
    )

    def __init__(
        self,
        times: Time,
        observation_location: EarthLocation,
        azimuths: RealArray,
        altitudes: RealArray,
    ) -> None:
        """Initialize the simulator with observation geometry.

        Args:
            times: Observation times for each simulation step.
            observation_location: Location of the observer on Earth.
            azimuths: Grid of world azimuths in degrees for sky sampling.
            altitudes: Grid of world altitudes in degrees for sky sampling.
        """
        normalized_times = self._validate_geometry(times, observation_location, azimuths, altitudes)
        self._sky_map = SkyCoord(
            alt=altitudes * deg,
            az=azimuths * deg,
            frame=AltAz(location=observation_location, obstime=normalized_times[:, None, None]),
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

    # ------------------------------------------------------------------ #
    # model-specific hooks
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _neutral_point_offsets(
        self, sun_altitudes_deg: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return the angular distances from the sun to its two neutral points.

        Args:
            sun_altitudes_deg: Solar elevation in degrees.

        Returns:
            Tuple of the below-sun (Brewster) and above-sun (Babinet) offsets,
            both in radians.
        """
        ...

    @abstractmethod
    def _get_dop(self, field: ComplexArray) -> FloatArray:
        """Compute the degree of polarization from the quartic field.

        Args:
            field: Complex polarization field.

        Returns:
            Degree of polarization for each sampled point.
        """
        ...

    @abstractmethod
    def _get_aop(
        self,
        field: ComplexArray,
        sun_azimuth: FloatArray,
        observed_point_azimuth: FloatArray,
    ) -> FloatArray:
        """Compute direct-model AOP from the quartic field.

        Args:
            field: Complex polarization field.
            sun_azimuth: Sun azimuth in radians.
            observed_point_azimuth: Observed point azimuth in radians.

        Returns:
            World-chart or local-meridian AOP, as defined by the concrete model,
            for each sampled point in radians.
        """
        ...

    @abstractmethod
    def _singularity_metadata(
        self,
        sun_position: SkyCoord,
        anti_sun_position: SkyCoord,
        below_sun_projection: ComplexArray,
        above_sun_projection: ComplexArray,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> list[FloatArray]:
        """Return the positions of the four neutral points.

        Implementations normally delegate to :meth:`_metadata_from_roots` or
        :meth:`_metadata_from_offsets`.

        Args:
            sun_position: Sun position in the simulator's AltAz frame.
            anti_sun_position: Anti-sun position in the simulator's AltAz frame.
            below_sun_projection: Stereographic coordinate of the Brewster point.
            above_sun_projection: Stereographic coordinate of the Babinet point.
            below_sun_offset: Sun-to-Brewster angular distance in radians.
            above_sun_offset: Sun-to-Babinet angular distance in radians.

        Returns:
            Eight arrays, ordered as above-sun azimuth and elevation, below-sun
            azimuth and elevation, above-anti-sun azimuth and elevation, and
            below-anti-sun azimuth and elevation. All values are in radians.
        """
        ...

    def _build_field(
        self,
        observed_point_projection: ComplexArray,
        observed_point_zenith_angle: FloatArray,
        observed_point_azimuth: FloatArray,
        sun_zenith_angle: FloatArray,
        sun_azimuth: FloatArray,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> tuple[ComplexArray, tuple[ComplexArray, ...]]:
        """Build the complex polarization field and return it with its roots.

        The default places two roots on the solar vertical and derives the anti-solar
        pair as their antipodes, which is Berry's minimal theory. Subclasses that
        place the four neutral points independently override this.

        Returns:
            The complex field, and the roots used to build it, ordered below-sun then
            above-sun.
        """
        below_sun_projection: ComplexArray = offset_along_solar_vertical(
            sun_zenith_angle=sun_zenith_angle,
            sun_azimuth=sun_azimuth,
            angular_offset=below_sun_offset,
            above_sun=False,
        )
        above_sun_projection: ComplexArray = offset_along_solar_vertical(
            sun_zenith_angle=sun_zenith_angle,
            sun_azimuth=sun_azimuth,
            angular_offset=above_sun_offset,
            above_sun=True,
        )

        field: ComplexArray = omega(
            observed_point_projection=observed_point_projection,
            below_sun_projection=below_sun_projection,
            above_sun_projection=above_sun_projection,
        )

        return field, (below_sun_projection, above_sun_projection)

    # ------------------------------------------------------------------ #
    # metadata helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _metadata_from_roots(
        below_sun_projection: ComplexArray, above_sun_projection: ComplexArray
    ) -> list[FloatArray]:
        """Return neutral-point positions by inverting the quartic's own roots.

        The four positions reported are exactly the four zeros used to build the
        polarization field, so the metadata cannot drift away from the model.

        Args:
            below_sun_projection: Stereographic coordinate of the Brewster point.
            above_sun_projection: Stereographic coordinate of the Babinet point.

        Returns:
            Eight arrays in the order described by :meth:`_singularity_metadata`.
        """
        return QuarticSkyModel._metadata_from_root_tuple(
            (
                above_sun_projection,
                below_sun_projection,
                antipode(below_sun_projection),
                antipode(above_sun_projection),
            )
        )

    @staticmethod
    def _metadata_from_root_tuple(
        roots: tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray],
    ) -> list[FloatArray]:
        """Return neutral-point positions by inverting four explicit roots.

        Args:
            roots: The four zeros, ordered above-sun, below-sun, above-anti-sun and
                below-anti-sun.

        Returns:
            Eight arrays in the order described by :meth:`_singularity_metadata`.
        """
        metadata: list[FloatArray] = []

        for root in roots:
            zenith_angle, azimuth = unproject(root)
            metadata.append(np.asarray(azimuth % _TWO_PI, dtype=np.float64))
            metadata.append(np.asarray(pi / 2 - zenith_angle, dtype=np.float64))

        return metadata

    @staticmethod
    def _metadata_from_offsets(
        sun_position: SkyCoord,
        anti_sun_position: SkyCoord,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> list[FloatArray]:
        """Return neutral-point positions by offsetting along the solar vertical.

        A position angle of zero in the AltAz frame points at the zenith, so a
        positive separation moves a point upwards along the solar vertical and a
        negative separation moves it downwards.

        Args:
            sun_position: Sun position in the simulator's AltAz frame.
            anti_sun_position: Anti-sun position in the simulator's AltAz frame.
            below_sun_offset: Sun-to-Brewster angular distance in radians.
            above_sun_offset: Sun-to-Babinet angular distance in radians.

        Returns:
            Eight arrays in the order described by :meth:`_singularity_metadata`.
        """
        points: list[SkyCoord] = [
            sun_position.directional_offset_by(
                position_angle=0 * deg, separation=above_sun_offset * rad
            ),
            sun_position.directional_offset_by(
                position_angle=0 * deg, separation=-below_sun_offset * rad
            ),
            anti_sun_position.directional_offset_by(
                position_angle=0 * deg, separation=below_sun_offset * rad
            ),
            anti_sun_position.directional_offset_by(
                position_angle=0 * deg, separation=-above_sun_offset * rad
            ),
        ]

        metadata: list[FloatArray] = []
        for point in points:
            metadata.append(np.asarray(point.az.radian, dtype=np.float64))
            metadata.append(np.asarray(point.alt.radian, dtype=np.float64))

        return metadata

    # ------------------------------------------------------------------ #
    # simulation
    # ------------------------------------------------------------------ #

    def simulate_sky(
        self,
        cie_sky_type: int,
        altitude_min_clip: float | None = None,
        accuracy: bool = False,
        sun_position: SkyCoord | None = None,
    ) -> list[FloatArray]:
        """Simulate direct-model fields at world AltAz sampling directions.

        Args:
            cie_sky_type: CIE sky type index for radiance model.
            altitude_min_clip: Minimum altitude (degrees) to keep; lower values masked.
            accuracy: Whether to use high-accuracy ephemeris for sun position.
            sun_position: Optional explicit sun position to use.

        Returns:
            List of arrays for polarization metrics, radiance, scattering angle,
            and singularity points relative to sun and anti-sun. AOP retains the
            concrete model's direct reference frame; no camera-pose transport is
            applied here.
        """
        sun_position = self._get_sun(accuracy=accuracy, sun_position=sun_position)
        anti_sun_position: SkyCoord = sun_position.directional_offset_by(
            position_angle=0 * deg, separation=180 * deg
        )

        sun_zenith_angle: FloatArray = np.asarray(
            (90 * deg - sun_position.alt).radian, dtype=np.float64
        )
        sun_azimuth: FloatArray = np.asarray(sun_position.az.radian, dtype=np.float64)
        observed_point_zenith_angle: FloatArray = np.asarray(
            (90 * deg - self.sky_map.alt).radian, dtype=np.float64
        )
        observed_point_azimuth: FloatArray = np.asarray(self.sky_map.az.radian, dtype=np.float64)

        below_sun_offset: FloatArray
        above_sun_offset: FloatArray
        below_sun_offset, above_sun_offset = self._neutral_point_offsets(sun_position.alt.deg)

        observed_point_projection: ComplexArray = project(
            zenith_angle=observed_point_zenith_angle, azimuth=observed_point_azimuth
        )
        field, roots = self._build_field(
            observed_point_projection=observed_point_projection,
            observed_point_zenith_angle=observed_point_zenith_angle,
            observed_point_azimuth=observed_point_azimuth,
            sun_zenith_angle=sun_zenith_angle,
            sun_azimuth=sun_azimuth,
            below_sun_offset=below_sun_offset,
            above_sun_offset=above_sun_offset,
        )
        below_sun_projection, above_sun_projection = roots[0], roots[1]

        scattering_angle: FloatArray = np.asarray(
            self.sky_map.separation(sun_position).radian, dtype=np.float64
        )

        radiance: FloatArray = self._get_radiance(
            cie_sky_type=cie_sky_type,
            observed_point_zenith_angle=observed_point_zenith_angle,
            sun_zenith_angle=sun_zenith_angle,
            scattering_angle=scattering_angle,
        )

        dop: FloatArray = self._get_dop(field)
        aop: FloatArray = self._get_aop(
            field=field, sun_azimuth=sun_azimuth, observed_point_azimuth=observed_point_azimuth
        )

        singularity_metadata: list[FloatArray] = self._singularity_metadata(
            sun_position=sun_position,
            anti_sun_position=anti_sun_position,
            below_sun_projection=below_sun_projection,
            above_sun_projection=above_sun_projection,
            below_sun_offset=below_sun_offset,
            above_sun_offset=above_sun_offset,
        )

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
            sun_azimuth,
            np.asarray(sun_position.alt.radian, dtype=np.float64),
            *singularity_metadata[:4],
            np.asarray(anti_sun_position.az.radian, dtype=np.float64),
            np.asarray(anti_sun_position.alt.radian, dtype=np.float64),
            *singularity_metadata[4:],
        ]
