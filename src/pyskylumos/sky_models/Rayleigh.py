from typing import List

from astropy.units import deg
from astropy.time import Time
from numpy.typing import NDArray
from numpy import sin, cos, float32, arctan, tan
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

from pyskylumos.sky_models.SkySimulator import SkySimulator


class Rayleigh(SkySimulator):
    sky_map: SkyCoord
    __PARAMETERS_SIMULATED: List[str] = [
        'degree of polarization',
        'angle of polarization',
        'radiance',
        'scattering angle',
        'sun azimuth',
        'sun elevation',
        'anti-sun azimuth',
        'anti-sun elevation'
    ]

    def __init__(
            self,
            times: Time,
            observation_location: EarthLocation,
            altitudes: NDArray[float32],
            azimuths: NDArray[float32]
    ) -> None:
        self._sky_map = SkyCoord(
            alt=altitudes * deg,
            az=azimuths * deg,
            frame=AltAz(
                obstime=times[:, None, None],
                location=observation_location
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
    def __get_dop(scattering_angle: NDArray[NDArray[float32]]) -> NDArray[float32]:
        return (
                sin(scattering_angle) ** 2 / (1 + cos(scattering_angle) ** 2)
        ).astype(float32)

    @staticmethod
    def __get_aop(
            observed_point_altitude: NDArray[float32],
            solar_altitude: NDArray[float32],
            azimuthal_difference: NDArray[float32],
            observed_particle_azimuth: NDArray[float32]
    ) -> NDArray[float32]:
        return arctan(
            tan(
                arctan(
                    (
                            sin(observed_point_altitude) * cos(solar_altitude) * cos(azimuthal_difference) -
                            cos(observed_point_altitude) * sin(solar_altitude)
                    ) / (
                            cos(solar_altitude) * sin(azimuthal_difference)
                    )
                ) + observed_particle_azimuth
            )
        ).astype(float32)

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

        scattering_angle: NDArray[float32] = self.sky_map.separation(sun_position).radian

        radiance: NDArray[float32] = self._get_radiance(
            cie_sky_type=cie_sky_type,
            observed_point_zenith_angle=(90 * deg - self.sky_map.alt).radian,
            sun_zenith_angle=(90 * deg - sun_position.alt).radian,
            scattering_angle=scattering_angle
        )

        dop: NDArray[float32] = self.__get_dop(
            scattering_angle=scattering_angle
        )

        azimuthal_difference: NDArray[float32] = (self.sky_map.az - sun_position.az).wrap_at(360 * deg).radian

        aop: NDArray[float32] = self.__get_aop(
            solar_altitude=sun_position.alt.radian,
            observed_point_altitude=self.sky_map.alt.radian,
            azimuthal_difference=azimuthal_difference,
            observed_particle_azimuth=self.sky_map.az.radian
        )

        mask: NDArray[bool] = self.sky_map.alt.deg <= altitude_min_clip
        radiance[mask] = None
        dop[mask] = None
        aop[mask] = None
        scattering_angle[mask] = None

        return [
            dop,
            aop,
            radiance,
            scattering_angle,
            sun_position.az.radian,
            sun_position.alt.radian,
            anti_sun_position.az.radian,
            anti_sun_position.alt.radian
        ]
