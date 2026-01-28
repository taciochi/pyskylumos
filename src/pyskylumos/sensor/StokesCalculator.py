"""Utilities to compute Stokes parameters and polarization metrics."""

from typing import Dict, Tuple

from numpy.typing import NDArray
from numpy import float32, sqrt, arctan2, divide, full_like, nan

from pyskylumos.sensor.SlicingPattern import SlicingPattern


class StokesCalculator:
    """Compute Stokes parameters and polarization metrics from sensor data."""

    __wire_grid_orientations_slicing: Dict[int, SlicingPattern]

    def __init__(
            self,
            wire_grid_orientations_slicing: Dict[int, SlicingPattern]
    ) -> None:
        """Initialize the calculator with slicing patterns per orientation.

        Args:
            wire_grid_orientations_slicing: Slicing pattern per orientation angle.
        """
        self.__wire_grid_orientations_slicing = wire_grid_orientations_slicing

    def __get_orientation_intensity(
            self,
            bits_intensity: NDArray[NDArray[float32]],
            orientation_angle: int
    ) -> NDArray[NDArray[float32]]:
        """Extract the intensity map for a given polarizer orientation.

        Args:
            bits_intensity: Quantized intensity values from the sensor.
            orientation_angle: Orientation angle in degrees.

        Returns:
            Intensity values for the requested orientation.
        """
        pattern: SlicingPattern = self.__wire_grid_orientations_slicing[orientation_angle]
        return bits_intensity[:, pattern.start_row::pattern.step, pattern.start_column::pattern.step]

    def __compute_stokes_parameters(
            self,
            bits_intensity: NDArray[NDArray[float32]]
    ) -> Tuple[NDArray[NDArray[float32]], NDArray[NDArray[float32]], NDArray[NDArray[float32]]]:
        """Compute Stokes parameters S0, S1, and S2 from intensity data.

        Args:
            bits_intensity: Quantized intensity values from the sensor.

        Returns:
            Tuple of S0, S1, and S2 arrays.
        """
        orientation_0_intensity: NDArray[NDArray[float32]] = self.__get_orientation_intensity(
            bits_intensity=bits_intensity,
            orientation_angle=0
        )

        orientation_45_intensity: NDArray[NDArray[float32]] = self.__get_orientation_intensity(
            bits_intensity=bits_intensity,
            orientation_angle=45
        )

        orientation_90_intensity: NDArray[NDArray[float32]] = self.__get_orientation_intensity(
            bits_intensity=bits_intensity,
            orientation_angle=90
        )

        orientation_135_intensity: NDArray[NDArray[float32]] = self.__get_orientation_intensity(
            bits_intensity=bits_intensity,
            orientation_angle=135
        )

        s0: NDArray[NDArray[float32]] = 0.5 * (
                orientation_0_intensity + orientation_45_intensity +
                orientation_90_intensity + orientation_135_intensity
        )
        s1: NDArray[NDArray[float32]] = orientation_0_intensity - orientation_90_intensity
        s2: NDArray[NDArray[float32]] = orientation_45_intensity - orientation_135_intensity

        return s0, s1, s2

    @staticmethod
    def __compute_degree_of_polarization(
            s0: NDArray[NDArray[float32]],
            s1: NDArray[NDArray[float32]],
            s2: NDArray[NDArray[float32]]
    ) -> NDArray[NDArray[float32]]:
        """Compute degree of polarization from Stokes parameters.

        Args:
            s0: Stokes parameter S0.
            s1: Stokes parameter S1.
            s2: Stokes parameter S2.

        Returns:
            Degree of polarization values.
        """
        numerator = sqrt(s1 ** 2 + s2 ** 2)
        return divide(numerator, s0, out=full_like(s0, nan), where=s0 != 0)

    @staticmethod
    def __compute_angle_of_polarization(
            s1: NDArray[NDArray[float32]],
            s2: NDArray[NDArray[float32]],
    ) -> NDArray[NDArray[float32]]:
        """Compute angle of polarization from Stokes parameters.

        Args:
            s1: Stokes parameter S1.
            s2: Stokes parameter S2.

        Returns:
            Angle of polarization values in radians.
        """
        return 0.5 * arctan2(s2, s1)

    def simulate_measurements(
            self,
            bits_intensity: NDArray[NDArray[float32]]
    ) -> Tuple[NDArray[float32], NDArray[float32]]:
        """Compute degree and angle of polarization from sensor readings.

        Args:
            bits_intensity: Quantized intensity values from the sensor.

        Returns:
            Tuple of degree and angle of polarization arrays.
        """
        s0: NDArray[NDArray[float32]]
        s1: NDArray[NDArray[float32]]
        s2: NDArray[NDArray[float32]]

        s0, s1, s2 = self.__compute_stokes_parameters(bits_intensity=bits_intensity)
        dop: NDArray[NDArray[float32]] = self.__compute_degree_of_polarization(s0=s0, s1=s1, s2=s2)
        aop: NDArray[NDArray[float32]] = self.__compute_angle_of_polarization(s1=s1, s2=s2)

        return dop, aop
