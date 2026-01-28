"""Optical conjugation model for mapping sensor pixels to sky coordinates."""

from typing import Tuple, Optional

from numpy.typing import NDArray
from numpy import linspace, ones, angle, arctan, pi, absolute, arcsin, rad2deg, float32, flip


class OpticalConjugator:
    """Convert sensor pixel positions into azimuth/altitude coordinates."""

    __lens_conjugation_type: str
    __number_pixels_vertical: int
    __number_pixels_horizontal: int
    __lens_focal_length_micrometers: float
    __sensor_pixel_size_square_micrometers: float
    __complex_sensor_plane_cache: Optional[NDArray[complex]]
    __azimuth_cache: Optional[NDArray[float32]]
    __altitude_cache: Optional[NDArray[float32]]

    def __init__(
            self,
            lens_conjugation_type: str,
            number_pixels_vertical: int,
            number_pixels_horizontal: int,
            lens_focal_length_micrometers: float,
            sensor_pixel_size_square_micrometers: float
    ) -> None:
        """Initialize optical conjugation parameters and caches.

        Args:
            lens_conjugation_type: Projection model name.
            number_pixels_vertical: Vertical pixel count of the sensor.
            number_pixels_horizontal: Horizontal pixel count of the sensor.
            lens_focal_length_micrometers: Lens focal length in micrometers.
            sensor_pixel_size_square_micrometers: Pixel size in square micrometers.
        """
        self.__lens_conjugation_type = lens_conjugation_type
        self.__number_pixels_vertical = number_pixels_vertical
        self.__number_pixels_horizontal = number_pixels_horizontal
        self.__lens_focal_length_micrometers = lens_focal_length_micrometers
        self.__sensor_pixel_size_square_micrometers = sensor_pixel_size_square_micrometers
        self.__complex_sensor_plane_cache = None
        self.__azimuth_cache = None
        self.__altitude_cache = None

    @property
    def lens_conjugation_type(self) -> str:
        """Return the configured lens conjugation type.

        Returns:
            Lens conjugation model name.
        """
        return self.__lens_conjugation_type

    @property
    def sensor_pixel_size_square_micrometers(self) -> float:
        """Return the sensor pixel size in square micrometers.

        Returns:
            Pixel size in square micrometers.
        """
        return self.__sensor_pixel_size_square_micrometers

    # noinspection PyTypeChecker
    def __get_complex_sensor_plane(
            self,
    ) -> NDArray[complex]:
        """Return the complex sensor plane for the pixel grid.

        Returns:
            Complex-valued grid representing sensor coordinates.
        """
        if self.__complex_sensor_plane_cache is not None:
            return self.__complex_sensor_plane_cache

        start_x: float = (self.__number_pixels_horizontal - 1) / 2
        stop_x: float = -start_x
        x_pixels: NDArray[float32] = linspace(start=start_x, stop=stop_x,
                                              num=self.__number_pixels_horizontal).astype(float32)
        del start_x, stop_x

        start_y: float = (self.__number_pixels_vertical - 1) / 2
        stop_y: float = -start_y
        y_pixels: NDArray[float32] = linspace(start=start_y, stop=stop_y,
                                              num=self.__number_pixels_vertical).astype(float32)
        del start_y, stop_y

        x_micrometers: NDArray[float32] = self.__sensor_pixel_size_square_micrometers * x_pixels
        y_micrometers: NDArray[float32] = self.__sensor_pixel_size_square_micrometers * y_pixels

        real: NDArray[float32] = ones(shape=(self.__number_pixels_vertical, 1)) * x_micrometers
        imaginary: NDArray[float32] = ones(shape=(1, self.__number_pixels_horizontal)) * y_micrometers.T[:, None]
        complex_plane: NDArray[complex] = real + 1j * imaginary

        self.__complex_sensor_plane_cache = complex_plane
        return complex_plane

    def __apply_conjugation(
            self,
            complex_sensor_plane: NDArray[complex],
            custom_lens_conjugation: Optional[callable]
    ) -> NDArray[float32]:
        """Apply the configured lens conjugation to sensor plane coordinates.

        Args:
            complex_sensor_plane: Complex sensor plane coordinates.
            custom_lens_conjugation: Optional custom conjugation function.

        Returns:
            Altitude angles (radians) for each sensor coordinate.
        """
        half_pi: float = pi / 2
        match self.__lens_conjugation_type:
            case 'thin':
                return half_pi - arctan(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers
                ).astype(float32)
            case 'stereographic':
                return half_pi - 2 * arctan(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers / 2
                ).astype(float32)
            case 'equi_angle':
                return half_pi - (absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers).astype(float32)
            case 'equi_solid_angle':
                return half_pi - 2 * arcsin(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers / 2
                ).astype(float32)
            case 'orthogonal':
                return half_pi - arcsin(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers
                )
            case 'custom':
                if custom_lens_conjugation is None:
                    raise ValueError('Custom lens conjugation type requires a custom_lens_conjugation function.')
                return custom_lens_conjugation(
                    complex_sensor_plane=complex_sensor_plane,
                    lens_focal_length_micrometers=self.__lens_focal_length_micrometers
                )
            case _:
                raise ValueError('Invalid lens projection type')

    def get_azimuth_altitude(
            self,
            altitude_min_clip: Optional[float],
            custom_lens_conjugation: Optional[callable] = None,
    ) -> Tuple[NDArray[float32], NDArray[float32]]:
        """Return azimuth and altitude angles for the sensor pixel grid.

        Args:
            altitude_min_clip: Minimum altitude to keep (degrees).
            custom_lens_conjugation: Optional custom conjugation function.

        Returns:
            Tuple of azimuth and altitude grids (degrees).
        """
        if self.__lens_conjugation_type != 'custom' and self.__azimuth_cache is not None and self.__altitude_cache is not None:
            azimuth = self.__azimuth_cache
            altitude = self.__altitude_cache
        else:
            complex_sensor_plane: NDArray[complex] = self.__get_complex_sensor_plane()
            azimuth = flip(angle(z=complex_sensor_plane, deg=True), axis=1)
            altitude = rad2deg(
                self.__apply_conjugation(
                    complex_sensor_plane=complex_sensor_plane,
                    custom_lens_conjugation=custom_lens_conjugation
                )
            )
            if self.__lens_conjugation_type != 'custom':
                self.__azimuth_cache = azimuth
                self.__altitude_cache = altitude

        if altitude_min_clip is not None:
            altitude = altitude.copy()
            altitude = altitude.clip(min=altitude_min_clip)

        return azimuth, altitude
