"""Map sensor pixels to directions in the sensor-local camera chart."""

from collections.abc import Callable
from typing import Any
from warnings import warn

import numpy as np
from numpy import absolute, angle, arcsin, arctan, pi, rad2deg

from pyskylumos._types import ComplexArray, FloatArray
from pyskylumos._validation import (
    UNSET,
    require_integer,
    require_real,
    resolve_deprecated_alias,
)
from pyskylumos.exceptions import ConfigurationError, InputTypeError, InputValidationError


class OpticalConjugator:
    """Convert sensor pixels into sensor-local azimuth/altitude directions.

    The default chart is upward looking, with North at image-right and East at
    image-top. Engine may interpret these directions as world aligned or rotate
    them through an integrated camera pose before evaluating the sky.
    """

    __lens_conjugation_type: str
    __number_pixels_vertical: int
    __number_pixels_horizontal: int
    __lens_focal_length_micrometers: float
    __sensor_pixel_pitch_micrometers: float
    __complex_sensor_plane_cache: ComplexArray | None
    __azimuth_cache: FloatArray | None
    __altitude_cache: FloatArray | None

    def __init__(
        self,
        lens_conjugation_type: Any = UNSET,
        number_pixels_vertical: Any = UNSET,
        number_pixels_horizontal: Any = UNSET,
        lens_focal_length_micrometers: Any = UNSET,
        sensor_pixel_size_square_micrometers: Any = UNSET,
        *,
        sensor_pixel_pitch_micrometers: Any = UNSET,
    ) -> None:
        """Initialize optical conjugation parameters and caches.

        Args:
            lens_conjugation_type: Projection model name.
            number_pixels_vertical: Vertical pixel count of the sensor.
            number_pixels_horizontal: Horizontal pixel count of the sensor.
            lens_focal_length_micrometers: Lens focal length in micrometers.
            sensor_pixel_size_square_micrometers: Deprecated alias for
                ``sensor_pixel_pitch_micrometers``.
            sensor_pixel_pitch_micrometers: Linear centre-to-centre pixel pitch
                in micrometers.
        """
        pitch = resolve_deprecated_alias(
            preferred_name="sensor_pixel_pitch_micrometers",
            preferred_value=sensor_pixel_pitch_micrometers,
            legacy_name="sensor_pixel_size_square_micrometers",
            legacy_value=sensor_pixel_size_square_micrometers,
        )
        if not isinstance(lens_conjugation_type, str):
            raise ConfigurationError("lens_conjugation_type must be a string.")
        allowed = {
            "thin",
            "stereographic",
            "equi_angle",
            "equi_solid_angle",
            "orthogonal",
            "custom",
        }
        if lens_conjugation_type not in allowed:
            raise ConfigurationError(
                f"lens_conjugation_type must be one of {sorted(allowed)}; "
                f"got {lens_conjugation_type!r}."
            )
        self.__lens_conjugation_type = lens_conjugation_type
        self.__number_pixels_vertical = require_integer(
            "number_pixels_vertical", number_pixels_vertical, minimum=1
        )
        self.__number_pixels_horizontal = require_integer(
            "number_pixels_horizontal", number_pixels_horizontal, minimum=1
        )
        self.__lens_focal_length_micrometers = require_real(
            "lens_focal_length_micrometers",
            lens_focal_length_micrometers,
            minimum=0.0,
            minimum_inclusive=False,
        )
        self.__sensor_pixel_pitch_micrometers = require_real(
            "sensor_pixel_pitch_micrometers",
            pitch,
            minimum=0.0,
            minimum_inclusive=False,
        )
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
        """Return the linear pixel pitch under its deprecated name.

        Returns:
            Linear pixel pitch in micrometers.
        """
        warn(
            "sensor_pixel_size_square_micrometers is deprecated; use "
            "sensor_pixel_pitch_micrometers instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.__sensor_pixel_pitch_micrometers

    @property
    def sensor_pixel_pitch_micrometers(self) -> float:
        """Return the linear centre-to-centre pixel pitch in micrometers."""
        return self.__sensor_pixel_pitch_micrometers

    def __get_complex_sensor_plane(self) -> ComplexArray:
        """Return the complex sensor plane for the pixel grid.

        Returns:
            Complex-valued grid representing sensor coordinates.
        """
        if self.__complex_sensor_plane_cache is not None:
            return self.__complex_sensor_plane_cache

        start_x: float = -((self.__number_pixels_horizontal - 1) / 2)
        stop_x: float = -start_x
        x_pixels: FloatArray = np.linspace(
            start=start_x,
            stop=stop_x,
            num=self.__number_pixels_horizontal,
            dtype=np.float64,
        )
        del start_x, stop_x

        start_y: float = (self.__number_pixels_vertical - 1) / 2
        stop_y: float = -start_y
        y_pixels: FloatArray = np.linspace(
            start=start_y,
            stop=stop_y,
            num=self.__number_pixels_vertical,
            dtype=np.float64,
        )
        del start_y, stop_y

        x_micrometers: FloatArray = self.__sensor_pixel_pitch_micrometers * x_pixels
        y_micrometers: FloatArray = self.__sensor_pixel_pitch_micrometers * y_pixels

        real: FloatArray = (
            np.ones(shape=(self.__number_pixels_vertical, 1), dtype=np.float64) * x_micrometers
        )
        imaginary: FloatArray = (
            np.ones(shape=(1, self.__number_pixels_horizontal), dtype=np.float64)
            * y_micrometers.T[:, None]
        )
        complex_plane: ComplexArray = np.asarray(real + 1j * imaginary, dtype=np.complex128)

        self.__complex_sensor_plane_cache = complex_plane
        return complex_plane

    def __apply_conjugation(
        self,
        complex_sensor_plane: ComplexArray,
        custom_lens_conjugation: Callable[..., FloatArray] | None,
    ) -> FloatArray:
        """Apply the configured lens conjugation to sensor plane coordinates.

        Args:
            complex_sensor_plane: Complex sensor plane coordinates.
            custom_lens_conjugation: Optional custom conjugation function.

        Returns:
            Altitude angles (radians) for each sensor coordinate.
        """
        half_pi: float = pi / 2
        match self.__lens_conjugation_type:
            case "thin":
                result = half_pi - arctan(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers
                )
            case "stereographic":
                result = half_pi - 2 * arctan(
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers / 2
                )
            case "equi_angle":
                result = half_pi - (
                    absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers
                )
            case "equi_solid_angle":
                argument = absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers / 2
                if np.any(argument > 1):
                    raise InputValidationError(
                        "equi_solid_angle projection is undefined because the sensor radius "
                        "exceeds twice the focal length."
                    )
                result = half_pi - 2 * arcsin(argument)
            case "orthogonal":
                argument = absolute(complex_sensor_plane) / self.__lens_focal_length_micrometers
                if np.any(argument > 1):
                    raise InputValidationError(
                        "orthogonal projection is undefined because the sensor radius "
                        "exceeds the focal length."
                    )
                result = half_pi - arcsin(argument)
            case "custom":
                if custom_lens_conjugation is None:
                    raise InputValidationError(
                        "Custom lens conjugation type requires a custom_lens_conjugation function."
                    )
                result = custom_lens_conjugation(
                    complex_sensor_plane=complex_sensor_plane,
                    lens_focal_length_micrometers=self.__lens_focal_length_micrometers,
                )
                if not isinstance(result, np.ndarray):
                    raise InputTypeError("custom_lens_conjugation must return a numpy.ndarray.")
                if result.shape != complex_sensor_plane.shape:
                    raise InputValidationError(
                        "custom_lens_conjugation must preserve the sensor-plane shape; "
                        f"expected {complex_sensor_plane.shape}, got {result.shape}."
                    )
                if not np.issubdtype(result.dtype, np.floating):
                    raise InputTypeError(
                        "custom_lens_conjugation must return real floating-point radians."
                    )
                if np.isinf(result).any():
                    raise InputValidationError(
                        "custom_lens_conjugation output must not contain infinity."
                    )
                return np.asarray(result, dtype=np.float64)
            case _:
                raise ConfigurationError("Invalid lens projection type.")
        return np.asarray(result, dtype=np.float64)

    def get_azimuth_altitude(
        self,
        altitude_min_clip: float | None,
        custom_lens_conjugation: Callable[..., FloatArray] | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return sensor-local azimuth and altitude for the pixel grid.

        Args:
            altitude_min_clip: Minimum altitude to keep (degrees).
            custom_lens_conjugation: Optional custom conjugation function.

        Returns:
            Sensor-local azimuth and altitude grids in degrees. Fresh arrays
            are returned on every call, so a caller writing to them cannot
            corrupt the internal cache. No world pose or AOP basis transport is
            applied here.
        """
        if altitude_min_clip is not None:
            altitude_min_clip = require_real(
                "altitude_min_clip", altitude_min_clip, minimum=-90.0, maximum=90.0
            )
        if custom_lens_conjugation is not None and not callable(custom_lens_conjugation):
            raise InputTypeError("custom_lens_conjugation must be callable or None.")
        if (
            self.__lens_conjugation_type != "custom"
            and self.__azimuth_cache is not None
            and self.__altitude_cache is not None
        ):
            azimuth = self.__azimuth_cache
            altitude = self.__altitude_cache
        else:
            complex_sensor_plane: ComplexArray = self.__get_complex_sensor_plane()
            azimuth = np.asarray(angle(complex_sensor_plane, deg=True), dtype=np.float64)
            altitude = np.asarray(
                rad2deg(
                    self.__apply_conjugation(
                        complex_sensor_plane=complex_sensor_plane,
                        custom_lens_conjugation=custom_lens_conjugation,
                    )
                ),
                dtype=np.float64,
            )
            if self.__lens_conjugation_type != "custom":
                self.__azimuth_cache = azimuth
                self.__altitude_cache = altitude

        azimuth = azimuth.copy()
        altitude = altitude.copy()

        if altitude_min_clip is not None:
            altitude = altitude.clip(min=altitude_min_clip)

        return azimuth, altitude
