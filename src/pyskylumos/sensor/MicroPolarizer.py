"""Micro-polarizer array model for wire-grid orientation effects."""

from typing import Dict, Optional, Tuple

from numpy.typing import NDArray
from numpy import float32, deg2rad, cos, zeros
from numpy.random import Generator, default_rng

from pyskylumos.sensor.SlicingPattern import SlicingPattern


class MicroPolarizer:
    """Simulate intensity response of a micro-polarizer array."""

    __extinction_ratio: float
    __tolerance: float
    __wire_grid_orientations_slicing: Dict[int, SlicingPattern]
    __angle_map_cache: Optional[NDArray[float32]]
    __defects_cache: Optional[NDArray[float32]]
    __angle_map_shape: Optional[Tuple[int, int]]
    __defects_shape: Optional[Tuple[int, int]]
    __rng: Generator

    def __init__(
            self,
            extinction_ratio: float,
            tolerance: float,
            wire_grid_orientations_slicing: Dict[int, SlicingPattern],
            random_seed: Optional[int] = None
    ) -> None:
        """Initialize the micro-polarizer with tolerances and slicing patterns.

        Args:
            extinction_ratio: Polarizer extinction ratio (e.g., 0.99).
            tolerance: Maximum angular offset for manufacturing defects.
            wire_grid_orientations_slicing: Slicing pattern per orientation angle.
            random_seed: Optional seed for deterministic defect generation.
        """
        self.__tolerance = tolerance
        self.__extinction_ratio = extinction_ratio
        self.__wire_grid_orientations_slicing = wire_grid_orientations_slicing
        self.__angle_map_cache = None
        self.__defects_cache = None
        self.__angle_map_shape = None
        self.__defects_shape = None
        self.__rng = default_rng(random_seed)

    def __get_angle_map(
            self,
            row_dim: int,
            col_dim: int
    ) -> NDArray[float32]:
        """Return the pixel-angle map for the given sensor dimensions.

        Args:
            row_dim: Number of pixel rows.
            col_dim: Number of pixel columns.

        Returns:
            Angle map containing the wire-grid orientation per pixel.
        """
        if self.__angle_map_cache is not None and self.__angle_map_shape == (row_dim, col_dim):
            return self.__angle_map_cache

        angle_map = zeros((row_dim, col_dim), dtype=float32)
        for orientation_angle_deg, slicing_pattern in self.__wire_grid_orientations_slicing.items():
            orientation_angle_rad = deg2rad(orientation_angle_deg)
            angle_map[
            slicing_pattern.start_row::slicing_pattern.step,
            slicing_pattern.start_column::slicing_pattern.step
            ] = orientation_angle_rad

        self.__angle_map_cache = angle_map
        self.__angle_map_shape = (row_dim, col_dim)
        if self.__defects_shape != self.__angle_map_shape:
            self.__defects_cache = None
            self.__defects_shape = None
        return angle_map

    def __get_defects(
            self,
            row_dim: int,
            col_dim: int
    ) -> NDArray[float32]:
        """Return random angular defects for the given sensor dimensions.

        Args:
            row_dim: Number of pixel rows.
            col_dim: Number of pixel columns.

        Returns:
            Defect map in radians for each pixel.
        """
        if self.__tolerance == 0:
            return zeros((row_dim, col_dim), dtype=float32)

        if self.__defects_cache is not None and self.__defects_shape == (row_dim, col_dim):
            return self.__defects_cache

        defects = self.__rng.random((row_dim, col_dim)).astype(float32)
        defects = (self.__tolerance * (1 - 2 * defects))
        self.__defects_cache = defects
        self.__defects_shape = (row_dim, col_dim)
        return defects

    def get_intensity_on_pixel(
            self,
            degree_of_polarization: NDArray[NDArray[float32]],
            angle_of_polarization: NDArray[NDArray[float32]],
            radiance: NDArray[NDArray[float32]]
    ) -> NDArray[float32]:
        """Compute intensity reaching each pixel after polarizer filtering.

        Args:
            degree_of_polarization: Degree of polarization values.
            angle_of_polarization: Angle of polarization values (radians).
            radiance: Radiance values per pixel.

        Returns:
            Intensity on each pixel after polarizer filtering.
        """
        _, row_dim, col_dim = radiance.shape
        angle_map = self.__get_angle_map(row_dim=row_dim, col_dim=col_dim)
        defects = self.__get_defects(row_dim=row_dim, col_dim=col_dim)
        angle_map = (angle_map + defects)[None, :, :]

        # 4) Compute intensity: I = 0.5 * radiance * [1 + (extinction_ratio * DoP) * cos(2 * (AoP - angle_map))]
        #    Note AoP and angle_map are in radians.
        intensity_on_pixel = 0.5 * radiance * (
                1.0 + (
                self.__extinction_ratio * degree_of_polarization
        ) * cos(
            2.0 * (angle_of_polarization - angle_map)
        )
        )

        return intensity_on_pixel
