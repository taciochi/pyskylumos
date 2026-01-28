"""Sensor chip model for quantization and noise effects."""

from numpy.random import randn
from numpy.typing import NDArray
from numpy import float32, nanmax, minimum, floor, nan_to_num, isnan


class SensorChip:
    """Simulate sensor quantization with noise and saturation effects."""

    __adc_resolution: float
    __signal_to_noise_ratio: float
    __pixel_saturation_ratio: float

    def __init__(
            self,
            pixel_saturation_ratio: float,
            adc_resolution: float,
            signal_to_noise_ratio: float,
    ) -> None:
        """Initialize the sensor chip configuration.

        Args:
            pixel_saturation_ratio: Ratio of max intensity used before saturation.
            adc_resolution: ADC resolution in bits.
            signal_to_noise_ratio: Desired signal-to-noise ratio.
        """
        self.__adc_resolution = adc_resolution
        self.__pixel_saturation_ratio = pixel_saturation_ratio
        self.__signal_to_noise_ratio = signal_to_noise_ratio

    def get_bits_intensity(
            self,
            intensity_on_pixel: NDArray[NDArray[float32]],
            # maximum_relative_radiance: float
    ) -> NDArray[NDArray[float32]]:
        """Convert intensity on pixels into quantized sensor readings.

        Args:
            intensity_on_pixel: Incoming intensity for each pixel.

        Returns:
            Quantized intensity values in ADC bits.
        """
        mask: NDArray[bool] = isnan(intensity_on_pixel)
        white_noise: NDArray[NDArray[float32]] = (
                (
                        (1 / self.__signal_to_noise_ratio) * nan_to_num(intensity_on_pixel, nan=0)
                        * randn(*intensity_on_pixel.shape)
                ) + nan_to_num(intensity_on_pixel, nan=0)
        ).astype(float32)

        max_pixel_value = 2 ** self.__adc_resolution - 1
        max_incoming_intensity = nanmax(intensity_on_pixel, axis=(1, 2), keepdims=True)

        bits_intensity: NDArray[NDArray[float32]] = minimum(
            floor(
                (
                        max_pixel_value / (max_incoming_intensity * self.__pixel_saturation_ratio)
                ) * white_noise
            ).astype(int),
            max_pixel_value
        )
        bits_intensity[mask] = None

        return bits_intensity
