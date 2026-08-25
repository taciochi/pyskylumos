"""Normalized sensor response with multiplicative noise and ADC quantization."""

from typing import Any

from numpy import clip, divide, float32, floor, isnan, isposinf, nan, nan_to_num, zeros_like
from numpy.random import Generator, default_rng

from pyskylumos._types import BoolArray, RealArray, SensorArray
from pyskylumos._validation import (
    UNSET,
    require_integer,
    require_real,
    require_real_array,
    resolve_deprecated_alias,
)
from pyskylumos.exceptions import InputValidationError


class SensorChip:
    """Simulate sensor quantization with noise and saturation effects."""

    __adc_resolution_bits: int
    __multiplicative_noise_snr: float
    __auto_exposure_saturation_fraction: float
    __rng: Generator

    def __init__(
        self,
        pixel_saturation_ratio: Any = UNSET,
        adc_resolution: Any = UNSET,
        signal_to_noise_ratio: Any = UNSET,
        random_seed: int | None = None,
        *,
        auto_exposure_saturation_fraction: Any = UNSET,
        adc_resolution_bits: Any = UNSET,
        multiplicative_noise_snr: Any = UNSET,
    ) -> None:
        """Initialize the sensor chip configuration.

        Args:
            pixel_saturation_ratio: Deprecated alias for
                ``auto_exposure_saturation_fraction``.
            adc_resolution: Deprecated alias for ``adc_resolution_bits``.
            signal_to_noise_ratio: Deprecated alias for ``multiplicative_noise_snr``.
            random_seed: Optional seed for deterministic multiplicative noise.
            auto_exposure_saturation_fraction: Fraction of the brightest finite
                pre-noise signal mapped to ADC full scale.
            adc_resolution_bits: Unsigned ADC resolution, from 1 through 24 bits.
            multiplicative_noise_snr: Positive signal-to-noise ratio for the
                relative Gaussian perturbation.
        """
        saturation = resolve_deprecated_alias(
            preferred_name="auto_exposure_saturation_fraction",
            preferred_value=auto_exposure_saturation_fraction,
            legacy_name="pixel_saturation_ratio",
            legacy_value=pixel_saturation_ratio,
        )
        bits = resolve_deprecated_alias(
            preferred_name="adc_resolution_bits",
            preferred_value=adc_resolution_bits,
            legacy_name="adc_resolution",
            legacy_value=adc_resolution,
        )
        snr = resolve_deprecated_alias(
            preferred_name="multiplicative_noise_snr",
            preferred_value=multiplicative_noise_snr,
            legacy_name="signal_to_noise_ratio",
            legacy_value=signal_to_noise_ratio,
        )
        self.__auto_exposure_saturation_fraction = require_real(
            "auto_exposure_saturation_fraction",
            saturation,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        self.__adc_resolution_bits = require_integer(
            "adc_resolution_bits", bits, minimum=1, maximum=24
        )
        self.__multiplicative_noise_snr = require_real(
            "multiplicative_noise_snr", snr, minimum=0.0, minimum_inclusive=False
        )
        if random_seed is not None:
            random_seed = require_integer("random_seed", random_seed, minimum=0)
        self.__rng = default_rng(random_seed)

    def get_bits_intensity(
        self,
        intensity_on_pixel: RealArray,
    ) -> SensorArray:
        """Convert intensity on pixels into quantized sensor readings.

        Counts are returned as ``float32`` rather than an integer type so that
        masked pixels can carry ``NaN``; every finite value is integral. Counts
        are clipped to the unsigned ADC range, so multiplicative noise cannot
        produce a negative reading.

        Positive infinity denotes an over-range signal and maps directly to ADC
        full scale. Such pixels are excluded when deriving exposure from the
        largest finite intensity, so they do not suppress finite pixels in the
        same frame. Negative intensity and negative infinity are invalid.

        The full-scale point is ``auto_exposure_saturation_fraction`` of the
        brightest finite intensity in the frame, so the brightest pixels saturate. A frame that is
        entirely dark, or entirely masked, yields zero counts rather than a
        division by zero. Noise is multiplicative Gaussian noise, not additive
        read noise: ``signal * (1 + Normal(0, 1 / SNR))``.

        Args:
            intensity_on_pixel: Incoming intensity for each pixel.

        Returns:
            Quantized intensity values in ADC counts, NaN where the input was NaN.

        Raises:
            ValueError: If any input intensity is negative or negative infinity.
        """
        intensity_on_pixel = require_real_array("intensity_on_pixel", intensity_on_pixel)
        if (intensity_on_pixel < 0).any():
            raise InputValidationError(
                "intensity_on_pixel must be non-negative; positive infinity and NaN masks are allowed."
            )

        mask: BoolArray = isnan(intensity_on_pixel)
        over_range: BoolArray = isposinf(intensity_on_pixel)
        signal: RealArray = nan_to_num(
            intensity_on_pixel,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        noisy_signal: RealArray = signal + (
            (signal / self.__multiplicative_noise_snr) * self.__rng.standard_normal(signal.shape)
        )

        max_pixel_value: float = 2**self.__adc_resolution_bits - 1
        saturation_intensity: RealArray = (
            signal.max(axis=(1, 2), keepdims=True) * self.__auto_exposure_saturation_fraction
        )

        counts_per_unit_intensity: RealArray = divide(
            max_pixel_value,
            saturation_intensity,
            out=zeros_like(saturation_intensity),
            where=saturation_intensity > 0,
        )

        bits_intensity: SensorArray = clip(
            floor(counts_per_unit_intensity * noisy_signal), 0.0, max_pixel_value
        ).astype(float32)
        bits_intensity[over_range] = max_pixel_value
        bits_intensity[mask] = nan

        return bits_intensity
