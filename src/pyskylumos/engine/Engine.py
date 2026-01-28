from math import pi
from typing import Dict, List, Tuple, Optional, Sequence

from astropy.time import Time
from numpy.typing import NDArray
from astropy.coordinates import EarthLocation, SkyCoord
from numpy import float32, arctan2, sqrt, cos, sin, array, where, deg2rad, rad2deg

from pyskylumos.sky_models.Pan import Pan
from pyskylumos.sky_models.Berry import Berry
from pyskylumos.sensor.SensorChip import SensorChip
from pyskylumos.sky_models.Rayleigh import Rayleigh
from pyskylumos.sensor.SlicingPattern import SlicingPattern
from pyskylumos.sky_models.SkySimulator import SkySimulator
from pyskylumos.sensor.MicroPolarizer import MicroPolarizer
from pyskylumos.sensor.StokesCalculator import StokesCalculator
from pyskylumos.sensor.OpticalConjugator import OpticalConjugator


class Engine:
    __micro_polarizer: MicroPolarizer
    __sensor_chip: SensorChip
    __stokes_calculator: StokesCalculator
    __optical_conjugator: OpticalConjugator

    def __init__(
            self,

            sensor_pixel_size_square_micrometers: float,
            lens_conjugation_type: str,
            number_pixels_vertical: int,
            number_pixels_horizontal: int,
            lens_focal_length_micrometers: float,

            tolerance: float,
            extinction_ratio: float,

            pixel_saturation_ratio: float,
            adc_resolution: float,
            signal_to_noise_ratio: float,

            wire_grid_orientations_slicing: Dict[int, SlicingPattern]

    ) -> None:
        self.__optical_conjugator = OpticalConjugator(
            lens_conjugation_type=lens_conjugation_type,
            number_pixels_vertical=number_pixels_vertical,
            number_pixels_horizontal=number_pixels_horizontal,
            lens_focal_length_micrometers=lens_focal_length_micrometers,
            sensor_pixel_size_square_micrometers=sensor_pixel_size_square_micrometers
        )

        self.__micro_polarizer = MicroPolarizer(
            tolerance=tolerance,
            extinction_ratio=extinction_ratio,
            wire_grid_orientations_slicing=wire_grid_orientations_slicing
        )

        self.__sensor_chip = SensorChip(
            pixel_saturation_ratio=pixel_saturation_ratio,
            adc_resolution=adc_resolution,
            signal_to_noise_ratio=signal_to_noise_ratio
        )

        self.__stokes_calculator = StokesCalculator(
            wire_grid_orientations_slicing=wire_grid_orientations_slicing
        )

    @staticmethod
    def __cartesian_to_spherical(
            x: NDArray[float32],
            y: NDArray[float32],
            z: NDArray[float32]
    ) -> Tuple[NDArray[float32], NDArray[float32]]:
        return arctan2(y, x), arctan2(z, sqrt(x ** 2 + y ** 2))

    @staticmethod
    def __spherical_to_cartesian(
            azimuths: NDArray[float32],
            altitudes: NDArray[float32],
            r: int = 1
    ) -> Tuple[NDArray[float32], NDArray[float32], NDArray[float32]]:
        return (
            r * cos(altitudes) * cos(azimuths),
            r * cos(altitudes) * sin(azimuths),
            r * sin(altitudes)
        )

    @staticmethod
    def __wrap_aop(
            aop: NDArray[float32],
            x: float
    ) -> NDArray[float32]:
        aop += deg2rad(x)
        aop = where(aop > pi / 2, aop - pi, aop)
        aop = where(aop < -pi / 2, aop + pi, aop)
        return aop

    def get_initial_azimuth_altitude(
            self,
            altitude_min_clip: Optional[float] = None,
            custom_lens_conjugation_type: Optional[callable] = None,
    ) -> Tuple[NDArray[float32], NDArray[float32]]:
        return self.__optical_conjugator.get_azimuth_altitude(
            altitude_min_clip=altitude_min_clip,
            custom_lens_conjugation=custom_lens_conjugation_type
        )

    def tilt_sensor(
            self,
            azimuths: NDArray[float32],
            altitudes: NDArray[float32],
            azimuthal_tilt: float,
            tilt_angle: float
    ) -> Tuple[NDArray[float32], NDArray[float32]]:
        rotation_canonical: NDArray[float32] = array([
            [1.0, 0.0, 0.0],
            [0.0, cos(tilt_angle), -sin(tilt_angle)],
            [0.0, sin(tilt_angle), cos(tilt_angle)]
        ])

        transition_cartesian_canonical: NDArray[float32] = array([
            [cos(azimuthal_tilt), -sin(azimuthal_tilt), 0],
            [sin(azimuthal_tilt), cos(azimuthal_tilt), 0],
            [0.0, 0.0, 1.0]
        ])

        rotation_cartesian: NDArray[float32] = (
                transition_cartesian_canonical.T @
                rotation_canonical @
                transition_cartesian_canonical
        )

        x: NDArray[float32]
        y: NDArray[float32]
        z: NDArray[float32]
        x, y, z = self.__spherical_to_cartesian(
            azimuths=azimuths,
            altitudes=altitudes
        )

        rotated_x: NDArray[float32] = rotation_cartesian[0, 0] * x + rotation_cartesian[0, 1] * y + rotation_cartesian[0, 2] * z
        rotated_y: NDArray[float32] = rotation_cartesian[1, 0] * x + rotation_cartesian[1, 1] * y + rotation_cartesian[1, 2] * z
        rotated_z: NDArray[float32] = rotation_cartesian[2, 0] * x + rotation_cartesian[2, 1] * y + rotation_cartesian[2, 2] * z

        alt: NDArray[float32]
        az: NDArray[float32]
        az, alt = self.__cartesian_to_spherical(
            x=rotated_x,
            y=rotated_y,
            z=rotated_z
        )

        return rad2deg(az), rad2deg(alt)

    @staticmethod
    def rotate_sensor(
            azimuths: NDArray[float32],
            rotation_angle: Optional[float] = None
    ) -> NDArray[float32]:
        if rotation_angle is None or rotation_angle == 0:
            return azimuths

        azimuths += rotation_angle
        azimuths = (azimuths + 180) % 360 - 180

        return azimuths

    @staticmethod
    def __get_sky_simulator(
            times: Time,
            sky_model: str,
            azimuths: NDArray[float32],
            altitudes: NDArray[float32],
            observation_location: EarthLocation
    ) -> SkySimulator:
        match sky_model.upper():
            case 'RAYLEIGH':
                return Rayleigh(
                    times=times,
                    altitudes=altitudes,
                    azimuths=azimuths,
                    observation_location=observation_location
                )
            case 'BERRY':
                return Berry(
                    times=times,
                    altitudes=altitudes,
                    azimuths=azimuths,
                    observation_location=observation_location
                )
            case 'PAN':
                return Pan(
                    times=times,
                    altitudes=altitudes,
                    azimuths=azimuths,
                    observation_location=observation_location
                )
            case _:
                raise ValueError(f'Sky model {sky_model.upper()} not found')

    def simulate_sky_polarization(
            self,
            sky_model: str,
            observation_location: EarthLocation,
            times: Time,
            cie_sky_type: int,
            altitudes: NDArray[float32],
            azimuths: NDArray[float32],
            altitude_min_clip: Optional[float] = None,
            azimuth_rotation_angle: Optional[float] = None,
            accuracy: Optional[bool] = False,
            sun_position: Optional[SkyCoord] = None,
    ) -> Tuple[Sequence[NDArray[float32]], List[str]]:
        """Simulate sky polarization parameters for a given sky model."""
        if altitude_min_clip is not None and not isinstance(altitude_min_clip, (int, float)):
            raise TypeError('altitude_min_clip must be a float or None.')
        if azimuth_rotation_angle is not None and not isinstance(azimuth_rotation_angle, (int, float)):
            raise TypeError('azimuth_rotation_angle must be a float or None.')
        if sun_position is not None and not isinstance(sun_position, SkyCoord):
            raise TypeError('sun_position must be an astropy.coordinates.SkyCoord or None.')

        sky_simulator: SkySimulator = self.__get_sky_simulator(
            times=times,
            azimuths=azimuths,
            sky_model=sky_model,
            altitudes=altitudes,
            observation_location=observation_location
        )

        sky_polarization_parameters: List[NDArray[float32]] = sky_simulator.simulate_sky(
            cie_sky_type=cie_sky_type,
            altitude_min_clip=altitude_min_clip,
            accuracy=accuracy,
            # Optional sun override for deterministic or external ephemerides.
            sun_position=sun_position,
        )

        if azimuth_rotation_angle is not None:
            sky_polarization_parameters[1] = self.__wrap_aop(
                aop=sky_polarization_parameters[1],
                x=-1 * azimuth_rotation_angle,
            )

        return (
            sky_polarization_parameters,
            sky_simulator.parameters_simulated
        )

    def simulate_measurement(
            self,
            degree_of_polarization: NDArray[float32],
            angle_of_polarization: NDArray[float32],
            radiance: NDArray[float32]
    ) -> Dict[str, NDArray[float32]]:

        iop: NDArray[NDArray[float32]] = self.__micro_polarizer.get_intensity_on_pixel(
            degree_of_polarization=degree_of_polarization,
            angle_of_polarization=angle_of_polarization,
            radiance=radiance
        )

        bits_intensity: NDArray[NDArray[float32]] = self.__sensor_chip.get_bits_intensity(intensity_on_pixel=iop)

        dop, aop = self.__stokes_calculator.simulate_measurements(bits_intensity=bits_intensity)

        return {
            'dop': dop,
            'aop': aop
        }
