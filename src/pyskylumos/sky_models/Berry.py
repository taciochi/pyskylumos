from typing import List

from astropy.units import deg
from astropy.time import Time
from numpy.typing import NDArray
from numpy import float32, tan, exp, angle, absolute, deg2rad
from astropy.coordinates import AltAz, SkyCoord, EarthLocation

from pyskylumos.sky_models.SkySimulator import SkySimulator


class Berry(SkySimulator):
    sky_map: SkyCoord
    __PARAMETERS_SIMULATED: List[str] = [
        'degree of polarization',
        'angle of polarization',
        'radiance',
        'scattering angle',

        'sun azimuth',
        'sun elevation',

        'above sun singularity point azimuth',
        'above sun singularity point elevation',

        'below sun singularity point azimuth',
        'below sun singularity point elevation',

        'anti-sun azimuth',
        'anti-sun elevation',

        'above anti-sun singularity point azimuth',
        'above anti-sun singularity point elevation',

        'below anti-sun singularity point azimuth',
        'below anti-sun singularity point elevation',
    ]

    def __init__(
            self,
            times: Time,
            observation_location: EarthLocation,
            azimuths: NDArray[float32],
            altitudes: NDArray[float32]
    ) -> None:
        self._sky_map = SkyCoord(
            alt=altitudes * deg, az=azimuths * deg,
            frame=AltAz(
                location=observation_location,
                obstime=times[:, None, None]
            )
        )

    @property
    def parameters_simulated(self) -> List[str]:
        return self.__PARAMETERS_SIMULATED

    @property
    def sky_map(self) -> SkyCoord:
        return self._sky_map

    @sky_map.setter
    def sky_map(self, new_sky_map: SkyCoord) -> None:
        self._sky_map = new_sky_map

    @staticmethod
    def __get_omega(
            angle_between_neutral_points: NDArray[float32],
            sun_zenith_angle: NDArray[float32],
            sun_azimuth: NDArray[float32],
            observed_point_zenith: NDArray[float32],
            observed_point_azimuth: NDArray[float32]
    ) -> NDArray[complex]:
        observed_point_projection: NDArray[complex] = (
                tan(observed_point_zenith / 2) *
                exp(observed_point_azimuth * 1j)
        )

        brewster_projection: NDArray[complex] = (
                exp(sun_azimuth * 1j) *
                (
                        (tan(sun_zenith_angle / 2) + tan(angle_between_neutral_points / 4)) /
                        (1 - tan(sun_zenith_angle / 2) * tan(angle_between_neutral_points / 4))
                )
        )

        babinet_projection: NDArray[complex] = (
                exp(sun_azimuth * 1j) *
                (
                        (tan(sun_zenith_angle / 2) - tan(angle_between_neutral_points / 4)) /
                        (1 + tan(sun_zenith_angle / 2) * tan(angle_between_neutral_points / 4))
                )
        )

        arago_projection: NDArray[complex] = (-1 / brewster_projection.conjugate())

        fourth_projection: NDArray[complex] = (-1 / babinet_projection.conjugate())

        return (
                (
                        -4 * (observed_point_projection - brewster_projection) *
                        (observed_point_projection - babinet_projection) *
                        (observed_point_projection - arago_projection) *
                        (observed_point_projection - fourth_projection)
                ) / (
                        ((1 + absolute(observed_point_projection) ** 2) ** 2) *
                        absolute(brewster_projection + -1 * arago_projection) *
                        absolute(babinet_projection + -1 * fourth_projection)
                )
        )

    @staticmethod
    def __get_dop(omega: NDArray[complex]) -> NDArray[float32]:
        return absolute(omega) / (2 - absolute(omega))

    @staticmethod
    def __get_aop(omega: NDArray[complex], sun_azimuth: NDArray[float32]) -> NDArray[float32]:
        return 0.5 * angle(
            z=(omega * exp(-2j * sun_azimuth))
        )

    def simulate_sky(
            self,
            cie_sky_type: int,
            altitude_min_clip: float | None = None,
            accuracy: bool = False,
            sun_position: SkyCoord | None = None,
    ) -> List[NDArray[float32]]:
        sun_position = self._get_sun(accuracy=accuracy, sun_position=sun_position)
        anti_sun_position: SkyCoord = sun_position.directional_offset_by(
            position_angle=0 * deg,
            separation=180 * deg
        )

        sun_zenith_angle: NDArray[float32] = (90 * deg - sun_position.alt).radian
        observed_point_zenith_angle: NDArray[float32] = (90 * deg - self.sky_map.alt).radian
        angle_between_neutral_points: NDArray[float32] = deg2rad(30)

        omega: NDArray[complex] = self.__get_omega(
            angle_between_neutral_points=angle_between_neutral_points,
            sun_zenith_angle=sun_zenith_angle,
            sun_azimuth=sun_position.az.radian,
            observed_point_zenith=observed_point_zenith_angle,
            observed_point_azimuth=self.sky_map.az.radian
        )

        scattering_angle: NDArray[float32] = self.sky_map.separation(sun_position).radian

        radiance: NDArray[float32] = self._get_radiance(
            cie_sky_type=cie_sky_type,
            observed_point_zenith_angle=(90 * deg - self.sky_map.alt).radian,
            sun_zenith_angle=(90 * deg - sun_position.alt).radian,
            scattering_angle=scattering_angle
        )

        aop = self.__get_aop(omega, sun_azimuth=sun_position.az.radian)
        dop = self.__get_dop(omega)

        mask: NDArray[bool] = self.sky_map.alt.deg <= altitude_min_clip
        radiance[mask] = None
        dop[mask] = None
        aop[mask] = None
        scattering_angle[mask] = None

        above_sun_singularity_point: SkyCoord = sun_position.directional_offset_by(
            position_angle=0 * deg,
            separation=15 * deg
        )

        below_sun_singularity_point: SkyCoord = sun_position.directional_offset_by(
            position_angle=0 * deg,
            separation=-15 * deg
        )

        above_anti_sun_singularity_point: SkyCoord = anti_sun_position.directional_offset_by(
            position_angle=0 * deg,
            separation=15 * deg
        )

        below_anti_sun_singularity_point: SkyCoord = anti_sun_position.directional_offset_by(
            position_angle=0 * deg,
            separation=-15 * deg
        )

        return [
            dop,
            aop,
            radiance,
            scattering_angle,

            sun_position.alt.radian,
            sun_position.az.radian,

            above_sun_singularity_point.az.radian,
            above_sun_singularity_point.alt.radian,

            below_sun_singularity_point.az.radian,
            below_sun_singularity_point.alt.radian,

            anti_sun_position.alt.radian,
            anti_sun_position.az.radian,

            above_anti_sun_singularity_point.az.radian,
            above_anti_sun_singularity_point.alt.radian,

            below_anti_sun_singularity_point.az.radian,
            below_anti_sun_singularity_point.alt.radian,
        ]
