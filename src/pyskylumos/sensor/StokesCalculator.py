"""Reconstruct sensor-frame Stokes parameters and polarization metrics."""

import numpy as np
from numpy import arctan2, clip, divide, full_like, isinf, nan, sqrt

from pyskylumos._types import SensorArray
from pyskylumos._validation import require_integer, require_real_array
from pyskylumos.exceptions import ConfigurationError, InputValidationError
from pyskylumos.sensor.SlicingPattern import SlicingPattern


class StokesCalculator:
    """Reconstruct polarization in the micro-polarizer's analyzer frame."""

    __wire_grid_orientations_slicing: dict[int, SlicingPattern]

    def __init__(self, wire_grid_orientations_slicing: dict[int, SlicingPattern]) -> None:
        """Initialize the calculator with slicing patterns per orientation.

        Args:
            wire_grid_orientations_slicing: Slicing pattern per orientation angle.
        """
        self.__wire_grid_orientations_slicing = self.__validate_mosaic(
            wire_grid_orientations_slicing
        )

    @staticmethod
    def __validate_mosaic(
        wire_grid_orientations_slicing: dict[int, SlicingPattern],
    ) -> dict[int, SlicingPattern]:
        """Return a defensive copy of a complete, disjoint 2x2 analyzer mosaic."""
        if not isinstance(wire_grid_orientations_slicing, dict):
            raise ConfigurationError("wire_grid_orientations_slicing must be a dict.")
        required = {0, 45, 90, 135}
        actual = set(wire_grid_orientations_slicing)
        if actual != required:
            raise ConfigurationError(
                "wire_grid_orientations_slicing must contain exactly analyzer orientations "
                f"{sorted(required)}; got {sorted(actual)}."
            )

        residues: set[tuple[int, int]] = set()
        for orientation, pattern in wire_grid_orientations_slicing.items():
            if not isinstance(pattern, SlicingPattern):
                raise ConfigurationError(
                    f"Mosaic entry {orientation} must be a SlicingPattern; "
                    f"got {type(pattern).__name__}."
                )
            if pattern.step != 2:
                raise ConfigurationError(
                    f"Mosaic entry {orientation} must use step=2; got {pattern.step}."
                )
            residue = (pattern.start_row, pattern.start_column)
            if pattern.start_row not in (0, 1) or pattern.start_column not in (0, 1):
                raise ConfigurationError(
                    f"Mosaic entry {orientation} offsets must be 0 or 1; got {residue}."
                )
            if residue in residues:
                raise ConfigurationError(f"Mosaic entries overlap at 2x2 tile offset {residue}.")
            residues.add(residue)

        if residues != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ConfigurationError("Mosaic entries must cover every position in the 2x2 tile.")
        return dict(wire_grid_orientations_slicing)

    def validate_sensor_shape(
        self, number_pixels_vertical: int, number_pixels_horizontal: int
    ) -> None:
        """Check that every orientation samples the same number of pixels.

        The Stokes parameters are formed by differencing the sub-images sampled by
        each analyzer orientation, so all four must have identical shapes. That
        holds when each sensor dimension is divisible by the mosaic step; a 5-pixel
        dimension with a step of 2 gives 3 rows for one orientation and 2 for
        another, which cannot be differenced.

        Args:
            number_pixels_vertical: Vertical pixel count of the sensor.
            number_pixels_horizontal: Horizontal pixel count of the sensor.

        Raises:
            ValueError: If the orientations would sample different pixel counts.
        """
        number_pixels_vertical = require_integer(
            "number_pixels_vertical", number_pixels_vertical, minimum=1
        )
        number_pixels_horizontal = require_integer(
            "number_pixels_horizontal", number_pixels_horizontal, minimum=1
        )
        for orientation_angle, pattern in self.__wire_grid_orientations_slicing.items():
            for dimension_name, size, start in (
                ("number_pixels_vertical", number_pixels_vertical, pattern.start_row),
                ("number_pixels_horizontal", number_pixels_horizontal, pattern.start_column),
            ):
                if size % pattern.step:
                    raise ConfigurationError(
                        f"{dimension_name} must be divisible by the micro-polarizer mosaic step "
                        f"{pattern.step} so that every analyzer orientation samples the same "
                        f"number of pixels; got {size}. Orientation {orientation_angle} degrees "
                        f"starts at offset {start}."
                    )

    def __get_orientation_intensity(
        self, bits_intensity: SensorArray, orientation_angle: int
    ) -> SensorArray:
        """Extract the intensity map for a given polarizer orientation.

        Args:
            bits_intensity: Quantized intensity values from the sensor.
            orientation_angle: Orientation angle in degrees.

        Returns:
            Intensity values for the requested orientation.
        """
        pattern: SlicingPattern = self.__wire_grid_orientations_slicing[orientation_angle]
        return bits_intensity[
            :, pattern.start_row :: pattern.step, pattern.start_column :: pattern.step
        ]

    def __compute_stokes_parameters(
        self,
        bits_intensity: SensorArray,
    ) -> tuple[SensorArray, SensorArray, SensorArray]:
        """Compute Stokes parameters S0, S1, and S2 from intensity data.

        Args:
            bits_intensity: Quantized intensity values from the sensor.

        Returns:
            Tuple of S0, S1, and S2 arrays.
        """
        orientation_0_intensity: SensorArray = self.__get_orientation_intensity(
            bits_intensity=bits_intensity, orientation_angle=0
        )

        orientation_45_intensity: SensorArray = self.__get_orientation_intensity(
            bits_intensity=bits_intensity, orientation_angle=45
        )

        orientation_90_intensity: SensorArray = self.__get_orientation_intensity(
            bits_intensity=bits_intensity, orientation_angle=90
        )

        orientation_135_intensity: SensorArray = self.__get_orientation_intensity(
            bits_intensity=bits_intensity, orientation_angle=135
        )

        s0: SensorArray = 0.5 * (
            orientation_0_intensity
            + orientation_45_intensity
            + orientation_90_intensity
            + orientation_135_intensity
        )
        s1: SensorArray = orientation_0_intensity - orientation_90_intensity
        s2: SensorArray = orientation_45_intensity - orientation_135_intensity

        return s0, s1, s2

    @staticmethod
    def __compute_degree_of_polarization(
        s0: SensorArray,
        s1: SensorArray,
        s2: SensorArray,
    ) -> SensorArray:
        """Compute degree of polarization from Stokes parameters.

        Args:
            s0: Stokes parameter S0.
            s1: Stokes parameter S1.
            s2: Stokes parameter S2.

        Returns:
            Degree of polarization values.
        """
        numerator = sqrt(s1**2 + s2**2)
        raw_dop = divide(numerator, s0, out=full_like(s0, nan), where=s0 != 0)
        return clip(raw_dop, 0.0, 1.0)

    @staticmethod
    def __compute_angle_of_polarization(
        s1: SensorArray,
        s2: SensorArray,
    ) -> SensorArray:
        """Compute angle of polarization from Stokes parameters.

        Args:
            s1: Stokes parameter S1.
            s2: Stokes parameter S2.

        Returns:
            AOP values in radians, referenced to the sensor-plane zero-degree
            analyzer axis.
        """
        return np.asarray(0.5 * arctan2(s2, s1), dtype=np.float32)

    def simulate_measurements(
        self,
        bits_intensity: SensorArray,
    ) -> tuple[SensorArray, SensorArray]:
        """Compute degree and angle of polarization from sensor readings.

        The reconstructed DOP is projected onto the physical interval [0, 1].
        This bounds noise- and saturation-driven estimates without changing AOP,
        while zero-intensity and masked measurements retain ``NaN`` DOP.

        Args:
            bits_intensity: Quantized intensity values from the sensor.

        Returns:
            Degree of polarization and sensor-frame AOP arrays.

        Raises:
            ValueError: If finite ADC counts are negative or any count is infinite.
        """
        bits_intensity = require_real_array("bits_intensity", bits_intensity)
        self.validate_sensor_shape(bits_intensity.shape[1], bits_intensity.shape[2])
        if isinf(bits_intensity).any():
            raise InputValidationError(
                "bits_intensity must not contain infinity; use NaN for masks."
            )
        if (bits_intensity < 0).any():
            raise InputValidationError("finite bits_intensity values must be non-negative.")

        s0: SensorArray
        s1: SensorArray
        s2: SensorArray

        s0, s1, s2 = self.__compute_stokes_parameters(bits_intensity=bits_intensity)
        dop: SensorArray = self.__compute_degree_of_polarization(s0=s0, s1=s1, s2=s2)
        aop: SensorArray = self.__compute_angle_of_polarization(s1=s1, s2=s2)

        return (
            np.asarray(dop, dtype=np.float32),
            np.asarray(aop, dtype=np.float32),
        )
