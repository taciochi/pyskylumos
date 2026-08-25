"""Tests for the sensor chip's quantization, clipping and noise.

Every assertion here covers behaviour that was broken before 0.1.0:
``get_bits_intensity`` raised ``TypeError`` for every input, and once that was
patched out it produced negative counts, divide-by-zero NaNs and all-NaN-slice
warnings.
"""

import warnings

import numpy as np
import pytest

from pyskylumos.sensor.SensorChip import SensorChip

ADC_RESOLUTION = 12
MAX_COUNT = 2**ADC_RESOLUTION - 1


def make_chip(multiplicative_noise_snr=50, random_seed=0, auto_exposure_saturation_fraction=0.9):
    """Return a sensor chip with deterministic noise."""
    return SensorChip(
        auto_exposure_saturation_fraction=auto_exposure_saturation_fraction,
        adc_resolution_bits=ADC_RESOLUTION,
        multiplicative_noise_snr=multiplicative_noise_snr,
        random_seed=random_seed,
    )


def gradient_frame(rows=8, columns=8):
    """Return a frame spanning a wide intensity range."""
    return np.linspace(0.0, 1.0, rows * columns, dtype=np.float32).reshape(1, rows, columns)


# --------------------------------------------------------------------------- #
# A1 -- it runs at all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "frame",
    [
        np.ones((1, 4, 4), dtype=np.float32),
        np.zeros((1, 4, 4), dtype=np.float32),
        np.full((1, 4, 4), np.nan, dtype=np.float32),
    ],
    ids=["clean", "all-zero", "all-nan"],
)
def test_get_bits_intensity_does_not_raise(frame):
    assert make_chip().get_bits_intensity(intensity_on_pixel=frame) is not None


def test_partially_masked_frame_is_accepted():
    frame = np.ones((1, 4, 4), dtype=np.float32)
    frame[0, 0, 0] = np.nan

    counts = make_chip().get_bits_intensity(intensity_on_pixel=frame)

    assert np.isnan(counts[0, 0, 0])
    assert np.isfinite(counts[0, 1:, :]).all()


# --------------------------------------------------------------------------- #
# A2 -- the returned contract
# --------------------------------------------------------------------------- #


def test_counts_are_float32_and_integral():
    counts = make_chip().get_bits_intensity(intensity_on_pixel=gradient_frame())

    assert counts.dtype == np.float32
    finite = counts[np.isfinite(counts)]
    np.testing.assert_array_equal(finite, np.floor(finite))


def test_output_shape_matches_input():
    frame = gradient_frame(rows=6, columns=10)

    assert make_chip().get_bits_intensity(intensity_on_pixel=frame).shape == frame.shape


# --------------------------------------------------------------------------- #
# A3 -- counts stay inside the unsigned ADC range
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("multiplicative_noise_snr", [0.5, 2, 10, 50, 1000])
def test_counts_stay_within_the_adc_range(multiplicative_noise_snr):
    counts = make_chip(multiplicative_noise_snr=multiplicative_noise_snr).get_bits_intensity(
        intensity_on_pixel=gradient_frame(rows=32, columns=32)
    )

    finite = counts[np.isfinite(counts)]
    assert finite.min() >= 0.0
    assert finite.max() <= MAX_COUNT


def test_a_dim_pixel_under_heavy_noise_cannot_go_negative():
    """Before 0.1.0 this configuration produced counts of about -4322."""
    frame = np.full((1, 64, 64), 1.0, dtype=np.float32)
    frame[0, 0, 0] = 1e-4

    counts = make_chip(multiplicative_noise_snr=2).get_bits_intensity(intensity_on_pixel=frame)

    assert np.nanmin(counts) >= 0.0


# --------------------------------------------------------------------------- #
# A5-A6 -- degenerate frames are handled quietly
# --------------------------------------------------------------------------- #


def test_all_zero_intensity_yields_zero_counts_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        counts = make_chip().get_bits_intensity(
            intensity_on_pixel=np.zeros((1, 4, 4), dtype=np.float32)
        )

    np.testing.assert_array_equal(counts, np.zeros((1, 4, 4), dtype=np.float32))


def test_fully_masked_frame_yields_all_nan_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        counts = make_chip().get_bits_intensity(
            intensity_on_pixel=np.full((1, 4, 4), np.nan, dtype=np.float32)
        )

    assert np.isnan(counts).all()


def test_each_frame_is_scaled_independently():
    """A dark frame beside a bright one must not borrow the bright one's scale."""
    frame = np.zeros((2, 4, 4), dtype=np.float32)
    frame[1] = 1.0

    counts = make_chip().get_bits_intensity(intensity_on_pixel=frame)

    np.testing.assert_array_equal(counts[0], np.zeros((4, 4), dtype=np.float32))
    assert counts[1].max() == MAX_COUNT


def test_positive_infinity_saturates_without_suppressing_finite_pixels():
    frame = np.array([[[0.25, np.inf], [0.5, 1.0]]], dtype=np.float32)
    finite_reference = frame.copy()
    finite_reference[np.isposinf(finite_reference)] = 0.0

    actual = make_chip(random_seed=7).get_bits_intensity(frame)
    expected_finite = make_chip(random_seed=7).get_bits_intensity(finite_reference)

    assert actual[0, 0, 1] == MAX_COUNT
    finite_mask = np.isfinite(frame)
    np.testing.assert_array_equal(actual[finite_mask], expected_finite[finite_mask])


def test_all_positive_infinite_intensity_saturates_quietly():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        counts = make_chip().get_bits_intensity(np.full((2, 4, 4), np.inf, dtype=np.float32))

    np.testing.assert_array_equal(counts, np.full((2, 4, 4), MAX_COUNT, dtype=np.float32))


@pytest.mark.parametrize("invalid_intensity", [-0.01, -np.inf])
def test_negative_input_intensity_is_rejected(invalid_intensity):
    frame = np.ones((1, 4, 4), dtype=np.float32)
    frame[0, 0, 0] = invalid_intensity

    with pytest.raises(ValueError, match="intensity_on_pixel must be non-negative"):
        make_chip().get_bits_intensity(frame)


def test_infinity_and_nan_handling_does_not_mutate_the_input_frame():
    frame = np.array([[[0.25, np.inf], [np.nan, 1.0]]], dtype=np.float32)
    original = frame.copy()

    make_chip().get_bits_intensity(frame)

    np.testing.assert_array_equal(frame, original)


# --------------------------------------------------------------------------- #
# A7 -- the mask survives quantization exactly
# --------------------------------------------------------------------------- #


def test_nan_positions_are_preserved_exactly():
    frame = gradient_frame(rows=8, columns=8)
    mask = np.zeros(frame.shape, dtype=bool)
    mask[0, ::3, ::2] = True
    frame[mask] = np.nan

    counts = make_chip().get_bits_intensity(intensity_on_pixel=frame)

    np.testing.assert_array_equal(np.isnan(counts), mask)


# --------------------------------------------------------------------------- #
# A8 -- determinism
# --------------------------------------------------------------------------- #


def test_identical_seeds_give_identical_counts():
    frame = gradient_frame()

    first = make_chip(random_seed=7).get_bits_intensity(intensity_on_pixel=frame)
    second = make_chip(random_seed=7).get_bits_intensity(intensity_on_pixel=frame)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_counts():
    frame = gradient_frame(rows=32, columns=32)

    first = make_chip(random_seed=7).get_bits_intensity(intensity_on_pixel=frame)
    second = make_chip(random_seed=8).get_bits_intensity(intensity_on_pixel=frame)

    assert not np.array_equal(first, second)


def test_an_unseeded_chip_still_works():
    counts = SensorChip(
        auto_exposure_saturation_fraction=0.9,
        adc_resolution_bits=ADC_RESOLUTION,
        multiplicative_noise_snr=50,
    ).get_bits_intensity(intensity_on_pixel=gradient_frame())

    assert np.isfinite(counts).all()


# --------------------------------------------------------------------------- #
# A9 -- saturation behaviour is unchanged
# --------------------------------------------------------------------------- #


def test_the_brightest_pixel_saturates():
    """Full scale sits at auto_exposure_saturation_fraction of the peak, so the peak clips."""
    frame = gradient_frame(rows=16, columns=16)

    counts = make_chip(multiplicative_noise_snr=100000).get_bits_intensity(intensity_on_pixel=frame)

    assert counts.max() == MAX_COUNT


def test_a_lower_saturation_ratio_saturates_more_pixels():
    frame = gradient_frame(rows=16, columns=16)

    generous = make_chip(multiplicative_noise_snr=100000, auto_exposure_saturation_fraction=0.99)
    aggressive = make_chip(multiplicative_noise_snr=100000, auto_exposure_saturation_fraction=0.5)

    generous_counts = generous.get_bits_intensity(intensity_on_pixel=frame)
    aggressive_counts = aggressive.get_bits_intensity(intensity_on_pixel=frame)

    assert (aggressive_counts == MAX_COUNT).sum() > (generous_counts == MAX_COUNT).sum()


@pytest.mark.parametrize("adc_resolution_bits", [8, 10, 12, 14, 16])
def test_adc_resolution_bits_sets_the_full_scale_value(adc_resolution_bits):
    chip = SensorChip(
        auto_exposure_saturation_fraction=0.9,
        adc_resolution_bits=adc_resolution_bits,
        multiplicative_noise_snr=100000,
        random_seed=0,
    )

    counts = chip.get_bits_intensity(intensity_on_pixel=gradient_frame(rows=16, columns=16))

    assert counts.max() == 2**adc_resolution_bits - 1
