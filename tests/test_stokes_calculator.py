"""Tests for Stokes parameter recovery and sensor shape validation."""

import numpy as np
import pytest

from pyskylumos.sensor.SlicingPattern import SlicingPattern
from pyskylumos.sensor.StokesCalculator import StokesCalculator

MOSAIC_2X2 = {
    0: SlicingPattern(start_row=0, start_column=0, step=2),
    45: SlicingPattern(start_row=0, start_column=1, step=2),
    90: SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}


def make_calculator(wire_grid_orientations_slicing=None):
    """Return a calculator for the given mosaic, defaulting to a 2x2 layout."""
    return StokesCalculator(
        wire_grid_orientations_slicing=wire_grid_orientations_slicing or MOSAIC_2X2
    )


def test_stokes_calculator_recovers_simple_signal():
    bits_intensity = np.array(
        [
            [
                [4.0, 2.0],
                [0.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )

    dop, aop = make_calculator().simulate_measurements(bits_intensity=bits_intensity)

    assert np.allclose(dop, 1.0)
    assert np.allclose(aop, 0.0)


def test_unpolarized_signal_gives_zero_degree_of_polarization():
    bits_intensity = np.full((1, 2, 2), 3.0, dtype=np.float32)

    dop, _ = make_calculator().simulate_measurements(bits_intensity=bits_intensity)

    assert np.allclose(dop, 0.0)


def test_zero_intensity_has_undefined_degree_of_polarization():
    dop, _ = make_calculator().simulate_measurements(
        bits_intensity=np.zeros((1, 2, 2), dtype=np.float32)
    )

    assert np.isnan(dop).all()


# --------------------------------------------------------------------------- #
# A10 -- sensor shape validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("vertical, horizontal", [(2, 2), (4, 4), (4, 6), (16, 64), (64, 16)])
def test_valid_sensor_shapes_are_accepted(vertical, horizontal):
    make_calculator().validate_sensor_shape(
        number_pixels_vertical=vertical,
        number_pixels_horizontal=horizontal,
    )


@pytest.mark.parametrize(
    "vertical, horizontal, offending",
    [
        (5, 5, "number_pixels_vertical"),
        (5, 4, "number_pixels_vertical"),
        (4, 5, "number_pixels_horizontal"),
        (63, 64, "number_pixels_vertical"),
    ],
)
def test_odd_sensor_shapes_are_rejected_by_name(vertical, horizontal, offending):
    with pytest.raises(ValueError, match=offending):
        make_calculator().validate_sensor_shape(
            number_pixels_vertical=vertical,
            number_pixels_horizontal=horizontal,
        )


def test_rejection_message_names_the_mosaic_step():
    with pytest.raises(ValueError, match="mosaic step 2"):
        make_calculator().validate_sensor_shape(
            number_pixels_vertical=5,
            number_pixels_horizontal=4,
        )


def test_validation_rejects_non_two_by_two_mosaics():
    """The four-channel Stokes contract requires complete 2x2 coverage."""
    mosaic_4x4 = {
        angle: SlicingPattern(start_row=index // 4, start_column=index % 4, step=4)
        for index, angle in enumerate([0, 45, 90, 135])
    }
    with pytest.raises(ValueError, match="step=2"):
        make_calculator(mosaic_4x4)


def test_the_shapes_validation_rejects_exactly_what_would_fail_at_runtime():
    """Whatever validation rejects must be what actually breaks, and vice versa."""
    calculator = make_calculator()

    for vertical, horizontal in [(4, 4), (5, 5), (4, 5), (6, 6)]:
        try:
            calculator.validate_sensor_shape(vertical, horizontal)
            accepted = True
        except ValueError:
            accepted = False

        try:
            calculator.simulate_measurements(
                bits_intensity=np.ones((1, vertical, horizontal), dtype=np.float32)
            )
            runs = True
        except ValueError:
            runs = False

        assert accepted == runs, f"validation disagrees with runtime for {vertical}x{horizontal}"


# --------------------------------------------------------------------------- #
# A4, A7 -- physical bounds and mask propagation
# --------------------------------------------------------------------------- #


def test_masked_counts_propagate_to_masked_measurements():
    bits_intensity = np.ones((1, 4, 4), dtype=np.float32)
    bits_intensity[0, 0, 0] = np.nan

    dop, aop = make_calculator().simulate_measurements(bits_intensity=bits_intensity)

    assert np.isnan(dop[0, 0, 0])
    assert np.isnan(aop[0, 0, 0])
    assert np.isfinite(dop[0, 1:, 1:]).all()


def malus_mosaic(degree_of_polarization, angle_of_polarization, radiance=1000.0):
    """Return a 2x2 mosaic of counts produced by one polarization state.

    Applies Malus's law at each analyzer angle, so the four sub-images are
    mutually consistent the way a real micro-polarizer array's are.
    """
    tile = np.empty((1, 2, 2), dtype=np.float32)
    for orientation_angle, pattern in MOSAIC_2X2.items():
        tile[0, pattern.start_row, pattern.start_column] = (
            0.5
            * radiance
            * (
                1.0
                + degree_of_polarization
                * np.cos(2.0 * (angle_of_polarization - np.deg2rad(orientation_angle)))
            )
        )
    return tile


@pytest.mark.parametrize("degree_of_polarization", [0.0, 0.25, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("angle_of_polarization_deg", [-80.0, -30.0, 0.0, 37.0, 89.0])
def test_round_trips_a_physical_polarization_state(
    degree_of_polarization, angle_of_polarization_deg
):
    """A state encoded by Malus's law must come back out unchanged."""
    angle_of_polarization = np.deg2rad(angle_of_polarization_deg)
    counts = malus_mosaic(degree_of_polarization, angle_of_polarization)

    dop, aop = make_calculator().simulate_measurements(bits_intensity=counts)

    assert dop.ravel()[0] == pytest.approx(degree_of_polarization, abs=1e-5)
    assert dop.ravel()[0] <= 1.0 + 1e-6
    if degree_of_polarization > 0:
        recovered = (aop.ravel()[0] + np.pi / 2) % np.pi - np.pi / 2
        expected = (angle_of_polarization + np.pi / 2) % np.pi - np.pi / 2
        assert recovered == pytest.approx(expected, abs=1e-5)


def test_angle_of_polarization_is_always_axial():
    rng = np.random.default_rng(0)
    bits_intensity = rng.integers(0, 4096, size=(4, 32, 32)).astype(np.float32)

    _, aop = make_calculator().simulate_measurements(bits_intensity=bits_intensity)

    assert np.all(np.abs(aop[np.isfinite(aop)]) <= np.pi / 2 + 1e-6)


def test_unphysical_raw_stokes_ratio_is_clipped_without_changing_aop():
    """One active analyzer gives raw DOP 2, whose cone projection has DOP 1."""
    bits_intensity = np.array(
        [
            [
                [4.0, 0.0],
                [0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    dop, aop = make_calculator().simulate_measurements(bits_intensity=bits_intensity)

    assert dop.ravel()[0] == 1.0
    assert aop.ravel()[0] == 0.0


def test_negative_counts_are_rejected():
    bits_intensity = np.array(
        [
            [
                [4.0, 2.0],
                [-3000.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match=r"bits_intensity.*non-negative"):
        make_calculator().simulate_measurements(bits_intensity=bits_intensity)


@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_infinite_counts_are_rejected(infinity):
    bits_intensity = np.ones((1, 2, 2), dtype=np.float32)
    bits_intensity[0, 0, 0] = infinity

    with pytest.raises(ValueError, match="bits_intensity must not contain infinity"):
        make_calculator().simulate_measurements(bits_intensity=bits_intensity)


def test_stokes_reconstruction_does_not_mutate_counts_with_nan_masks():
    bits_intensity = np.ones((1, 4, 4), dtype=np.float32)
    bits_intensity[0, 0, 0] = np.nan
    original = bits_intensity.copy()

    make_calculator().simulate_measurements(bits_intensity=bits_intensity)

    np.testing.assert_array_equal(bits_intensity, original)
