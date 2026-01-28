from typing import Dict, List
from abc import ABC, abstractmethod

from numpy.typing import NDArray
from numpy import float32, exp, cos, pi
from astropy.coordinates import SkyCoord, AltAz, get_sun, get_body


class SkySimulator(ABC):
    __cie_sky_types: Dict[int, Dict[str, float]] = {
        1: {
            'A': 4.0,
            'B': -0.7,
            'C': 0.0,
            'D': -1.0,
            'E': 0.0
        },
        2: {
            'A': 4.0,
            'B': -0.7,
            'C': 2.0,
            'D': -1.5,
            'E': 0.15
        },
        3: {
            'A': 1.1,
            'B': -0.8,
            'C': 0.0,
            'D': -1.0,
            'E': 0.0
        },
        4: {
            'A': 1.1,
            'B': -0.8,
            'C': 2.0,
            'D': -1.5,
            'E': 0.15
        },
        5: {
            'A': 0.0,
            'B': -1.0,
            'C': 0.0,
            'D': -1.0,
            'E': 0.0
        },
        6: {
            'A': 0.0,
            'B': -1.0,
            'C': 2.0,
            'D': -1.5,
            'E': 0.15
        },
        7: {
            'A': 0.0,
            'B': -1.0,
            'C': 5.0,
            'D': -2.5,
            'E': 0.3
        },
        8: {
            'A': 0.0,
            'B': -1.0,
            'C': 10.0,
            'D': -3.0,
            'E': 0.45
        },
        9: {
            'A': -1.0,
            'B': -0.55,
            'C': 2.0,
            'D': -1.5,
            'E': 0.15
        },
        10: {
            'A': -1.0,
            'B': -0.55,
            'C': 5.0,
            'D': -2.5,
            'E': 0.3
        },
        11: {
            'A': -1.0,
            'B': -0.55,
            'C': 10.0,
            'D': -3.0,
            'E': 0.45
        },
        12: {
            'A': -1.0,
            'B': -0.32,
            'C': 10.0,
            'D': -3.0,
            'E': 0.45
        },
        13: {
            'A': -1.0,
            'B': -0.32,
            'C': 16.0,
            'D': -3.0,
            'E': 0.3
        },
        14: {
            'A': -1.0,
            'B': -0.15,
            'C': 16.0,
            'D': -3.0,
            'E': 0.3
        },
        15: {
            'A': -1.0,
            'B': -0.15,
            'C': 24.0,
            'D': -2.8,
            'E': 0.15
        }
    }

    @property
    @abstractmethod
    def parameters_simulated(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def sky_map(self) -> SkyCoord:
        ...

    @sky_map.setter
    @abstractmethod
    def sky_map(self, new_sky_map: SkyCoord) -> None:
        ...

    @abstractmethod
    def simulate_sky(
            self,
            cie_sky_type: int,
            altitude_min_clip: float | None = None,
            accuracy: bool = False,
            sun_position: SkyCoord | None = None,
    ) -> List[NDArray[float32]]:
        ...

    def _get_sun(
            self,
            accuracy: bool = False,
            sun_position: SkyCoord | None = None
    ) -> SkyCoord:
        obstime = self.sky_map.obstime[..., 0, 0]
        frame = AltAz(obstime=obstime, location=self.sky_map.location)

        if sun_position is not None:
            if not isinstance(sun_position, SkyCoord):
                raise TypeError('sun_position must be an astropy.coordinates.SkyCoord or None.')
            sun_position = sun_position.transform_to(frame)
            if sun_position.shape == obstime.shape:
                return sun_position[..., None, None]
            return sun_position

        if accuracy:
            return get_body("sun", obstime, ephemeris="jpl").transform_to(frame)[..., None, None]

        return get_sun(obstime).transform_to(frame)[..., None, None]

    @classmethod
    def _get_radiance(
            cls,
            cie_sky_type: int,
            observed_point_zenith_angle: NDArray[float32],
            sun_zenith_angle: NDArray[float32],
            scattering_angle: NDArray[float32]
    ) -> NDArray[float32]:
        radiance_parameters: Dict[str, float] = cls.__cie_sky_types[cie_sky_type]
        half_pi: float = pi / 2

        return (
                (
                        (
                                1 + radiance_parameters['A'] *
                                exp(radiance_parameters['B'] / cos(observed_point_zenith_angle))
                        )
                        /
                        (
                                1 + radiance_parameters['A'] * exp(radiance_parameters['B'])
                        )
                )
                *
                (
                        (
                                1 + radiance_parameters['C'] *
                                (
                                        exp(radiance_parameters['D'] * scattering_angle) -
                                        exp(radiance_parameters['D'] * half_pi)
                                )
                                + radiance_parameters['E'] * cos(scattering_angle) ** 2
                        )
                        /
                        (
                                1 + radiance_parameters['C'] *
                                (
                                        exp(radiance_parameters['D'] * sun_zenith_angle) -
                                        exp(radiance_parameters['D'] * half_pi)
                                )
                                + radiance_parameters['E'] * cos(sun_zenith_angle) ** 2
                        )
                )
        )
