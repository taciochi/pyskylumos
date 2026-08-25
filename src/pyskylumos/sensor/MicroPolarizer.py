"""Micro-polarizer response in the active sensor/analyzer coordinate frame."""

from typing import Any

import numpy as np
from numpy import cos, deg2rad, float32, isinf, isnan, multiply, nan, zeros, zeros_like
from numpy.random import Generator, default_rng

from pyskylumos._types import RealArray, SensorArray
from pyskylumos._validation import (
    UNSET,
    require_integer,
    require_real,
    require_real_array,
    require_same_shape,
    resolve_deprecated_alias,
)
from pyskylumos.exceptions import ConfigurationError, InputValidationError
from pyskylumos.sensor.SlicingPattern import SlicingPattern
from pyskylumos.sensor.StokesCalculator import StokesCalculator


class MicroPolarizer:
    """Simulate a micro-polarizer array in its physical sensor-plane frame.

    Analyzer angles and input AOP must use the same active sensor frame. This
    class has no world-ray or camera-pose information; Engine is responsible for
    transporting sky AOP into this frame before measurement.
    """

    __extinction_ratio: float
    __polarizer_tolerance_radians: float
    __wire_grid_orientations_slicing: dict[int, SlicingPattern]
    __angle_map_cache: SensorArray | None
    __defects_cache: SensorArray | None
    __angle_map_shape: tuple[int, int] | None
    __defects_shape: tuple[int, int] | None
    __rng: Generator

    def __init__(
        self,
        extinction_ratio: Any = UNSET,
        tolerance: Any = UNSET,
        wire_grid_orientations_slicing: Any = UNSET,
        random_seed: int | None = None,
        *,
        polarizer_tolerance_radians: Any = UNSET,
    ) -> None:
        """Initialize the micro-polarizer with tolerances and slicing patterns.

        Args:
            extinction_ratio: Polarizer extinction ratio (e.g., 0.99).
            tolerance: Deprecated alias for ``polarizer_tolerance_radians``.
            wire_grid_orientations_slicing: Slicing pattern per sensor-plane
                analyzer angle in degrees.
            random_seed: Optional seed for deterministic defect generation.
            polarizer_tolerance_radians: Maximum absolute manufacturing-defect
                angle in radians.
        """
        tolerance_value = resolve_deprecated_alias(
            preferred_name="polarizer_tolerance_radians",
            preferred_value=polarizer_tolerance_radians,
            legacy_name="tolerance",
            legacy_value=tolerance,
        )
        self.__polarizer_tolerance_radians = require_real(
            "polarizer_tolerance_radians",
            tolerance_value,
            minimum=0.0,
            maximum=np.pi / 2,
        )
        self.__extinction_ratio = require_real(
            "extinction_ratio", extinction_ratio, minimum=0.0, maximum=1.0
        )
        if not isinstance(wire_grid_orientations_slicing, dict):
            raise ConfigurationError("wire_grid_orientations_slicing must be a dict.")
        StokesCalculator(wire_grid_orientations_slicing)
        self.__wire_grid_orientations_slicing = wire_grid_orientations_slicing
        self.__angle_map_cache = None
        self.__defects_cache = None
        self.__angle_map_shape = None
        self.__defects_shape = None
        if random_seed is not None:
            random_seed = require_integer("random_seed", random_seed, minimum=0)
        self.__rng = default_rng(random_seed)

    def __get_angle_map(self, row_dim: int, col_dim: int) -> SensorArray:
        """Return the pixel-angle map for the given sensor dimensions.

        Args:
            row_dim: Number of pixel rows.
            col_dim: Number of pixel columns.

        Returns:
            Sensor-frame wire-grid orientation per pixel, in radians.
        """
        if self.__angle_map_cache is not None and self.__angle_map_shape == (row_dim, col_dim):
            return self.__angle_map_cache

        angle_map = zeros((row_dim, col_dim), dtype=float32)
        for orientation_angle_deg, slicing_pattern in self.__wire_grid_orientations_slicing.items():
            orientation_angle_rad = deg2rad(orientation_angle_deg)
            angle_map[
                slicing_pattern.start_row :: slicing_pattern.step,
                slicing_pattern.start_column :: slicing_pattern.step,
            ] = orientation_angle_rad

        self.__angle_map_cache = angle_map
        self.__angle_map_shape = (row_dim, col_dim)
        if self.__defects_shape != self.__angle_map_shape:
            self.__defects_cache = None
            self.__defects_shape = None
        return angle_map

    def __get_defects(self, row_dim: int, col_dim: int) -> SensorArray:
        """Return random angular defects for the given sensor dimensions.

        Args:
            row_dim: Number of pixel rows.
            col_dim: Number of pixel columns.

        Returns:
            Defect map in radians for each pixel.
        """
        if self.__polarizer_tolerance_radians == 0:
            return zeros((row_dim, col_dim), dtype=float32)

        if self.__defects_cache is not None and self.__defects_shape == (row_dim, col_dim):
            return self.__defects_cache

        defects = self.__rng.random((row_dim, col_dim)).astype(float32)
        defects = self.__polarizer_tolerance_radians * (1 - 2 * defects)
        self.__defects_cache = defects
        self.__defects_shape = (row_dim, col_dim)
        return defects

    def get_intensity_on_pixel(
        self,
        degree_of_polarization: RealArray,
        angle_of_polarization: RealArray,
        radiance: RealArray,
    ) -> RealArray:
        """Compute intensity reaching each pixel after polarizer filtering.

        Finite degree-of-polarization values must be physical and finite
        radiance values must be non-negative. ``NaN`` values are preserved as
        masks. Positive-infinite radiance is an explicit over-range sentinel:
        it remains positive infinity wherever the analyzer transmission is
        non-zero, and an exactly extinguished analyzer still receives zero.

        Args:
            degree_of_polarization: Degree of polarization values.
            angle_of_polarization: AOP in radians, referenced to the same active
                sensor-plane axis as the wire-grid orientation map.
            radiance: Radiance values per pixel.

        Returns:
            Intensity on each pixel after polarizer filtering.

        Raises:
            ValueError: If DOP is outside [0, 1], DOP or AOP is infinite, or
                radiance is negative or negative infinity.
        """
        degree_of_polarization = require_real_array(
            "degree_of_polarization", degree_of_polarization
        )
        angle_of_polarization = require_real_array("angle_of_polarization", angle_of_polarization)
        radiance = require_real_array("radiance", radiance)
        require_same_shape(
            degree_of_polarization=degree_of_polarization,
            angle_of_polarization=angle_of_polarization,
            radiance=radiance,
        )
        StokesCalculator(self.__wire_grid_orientations_slicing).validate_sensor_shape(
            radiance.shape[1], radiance.shape[2]
        )
        if isinf(degree_of_polarization).any():
            raise InputValidationError(
                "degree_of_polarization must not contain infinity; use NaN for masks."
            )
        if ((degree_of_polarization < 0) | (degree_of_polarization > 1)).any():
            raise InputValidationError("finite degree_of_polarization values must lie in [0, 1].")
        if isinf(angle_of_polarization).any():
            raise InputValidationError(
                "angle_of_polarization must not contain infinity; use NaN for masks."
            )
        if (radiance < 0).any():
            raise InputValidationError(
                "radiance must be non-negative; positive infinity and NaN masks are allowed."
            )

        _, row_dim, col_dim = radiance.shape
        angle_map = self.__get_angle_map(row_dim=row_dim, col_dim=col_dim)
        defects = self.__get_defects(row_dim=row_dim, col_dim=col_dim)
        angle_map = (angle_map + defects)[None, :, :]

        # 4) Compute intensity: I = 0.5 * radiance * [1 + (extinction_ratio * DoP) * cos(2 * (AoP - angle_map))]
        #    Note AoP and angle_map are in radians.
        transmission = 0.5 * (
            1.0
            + (self.__extinction_ratio * degree_of_polarization)
            * cos(2.0 * (angle_of_polarization - angle_map))
        )

        intensity_on_pixel = zeros_like(transmission)
        multiply(
            radiance,
            transmission,
            out=intensity_on_pixel,
            where=transmission != 0,
        )
        intensity_on_pixel[isnan(radiance) | isnan(transmission)] = nan

        return intensity_on_pixel
