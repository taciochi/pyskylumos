import numpy as np
import pytest

from pyskylumos.sensor.MicroPolarizer import MicroPolarizer
from pyskylumos.sensor.SlicingPattern import SlicingPattern


def test_micro_polarizer_intensity_pattern():
    wire_grid_orientations_slicing = {
        0: SlicingPattern(start_row=0, start_column=0, step=2),
        45: SlicingPattern(start_row=0, start_column=1, step=2),
        90: SlicingPattern(start_row=1, start_column=0, step=2),
        135: SlicingPattern(start_row=1, start_column=1, step=2),
    }
    polarizer = MicroPolarizer(
        extinction_ratio=1.0,
        polarizer_tolerance_radians=0.0,
        wire_grid_orientations_slicing=wire_grid_orientations_slicing,
    )

    radiance = np.ones((1, 2, 2), dtype=np.float32)
    dop = np.ones_like(radiance)
    aop = np.zeros_like(radiance)

    intensity = polarizer.get_intensity_on_pixel(
        degree_of_polarization=dop, angle_of_polarization=aop, radiance=radiance
    )

    expected = np.array(
        [
            [
                [1.0, 0.5],
                [0.0, 0.5],
            ]
        ],
        dtype=np.float32,
    )

    assert np.allclose(intensity, expected)


def test_micro_polarizer_deterministic_defects():
    wire_grid_orientations_slicing = {
        0: SlicingPattern(start_row=0, start_column=0, step=2),
        45: SlicingPattern(start_row=0, start_column=1, step=2),
        90: SlicingPattern(start_row=1, start_column=0, step=2),
        135: SlicingPattern(start_row=1, start_column=1, step=2),
    }
    polarizer = MicroPolarizer(
        extinction_ratio=0.95,
        polarizer_tolerance_radians=0.1,
        wire_grid_orientations_slicing=wire_grid_orientations_slicing,
        random_seed=42,
    )

    radiance = np.ones((1, 2, 2), dtype=np.float32)
    dop = np.ones_like(radiance)
    aop = np.zeros_like(radiance)

    first = polarizer.get_intensity_on_pixel(dop, aop, radiance)
    second = polarizer.get_intensity_on_pixel(dop, aop, radiance)

    assert np.allclose(first, second)


def make_perfect_polarizer():
    """Return a defect-free ideal 2x2 analyzer mosaic."""
    return MicroPolarizer(
        extinction_ratio=1.0,
        polarizer_tolerance_radians=0.0,
        wire_grid_orientations_slicing={
            0: SlicingPattern(start_row=0, start_column=0, step=2),
            45: SlicingPattern(start_row=0, start_column=1, step=2),
            90: SlicingPattern(start_row=1, start_column=0, step=2),
            135: SlicingPattern(start_row=1, start_column=1, step=2),
        },
    )


@pytest.mark.parametrize("invalid_dop", [-0.01, 1.01])
def test_nonphysical_degree_of_polarization_is_rejected(invalid_dop):
    values = np.full((1, 2, 2), invalid_dop, dtype=np.float32)

    with pytest.raises(ValueError, match=r"degree_of_polarization.*\[0, 1\]"):
        make_perfect_polarizer().get_intensity_on_pixel(
            degree_of_polarization=values,
            angle_of_polarization=np.zeros_like(values),
            radiance=np.ones_like(values),
        )


@pytest.mark.parametrize("parameter", ["degree_of_polarization", "angle_of_polarization"])
@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_infinite_polarization_inputs_are_rejected(parameter, infinity):
    arguments = {
        "degree_of_polarization": np.full((1, 2, 2), 0.5, dtype=np.float32),
        "angle_of_polarization": np.zeros((1, 2, 2), dtype=np.float32),
        "radiance": np.ones((1, 2, 2), dtype=np.float32),
    }
    arguments[parameter][0, 0, 0] = infinity

    with pytest.raises(ValueError, match=parameter):
        make_perfect_polarizer().get_intensity_on_pixel(**arguments)


@pytest.mark.parametrize("invalid_radiance", [-0.01, -np.inf])
def test_negative_radiance_is_rejected(invalid_radiance):
    radiance = np.ones((1, 2, 2), dtype=np.float32)
    radiance[0, 0, 0] = invalid_radiance

    with pytest.raises(ValueError, match="radiance must be non-negative"):
        make_perfect_polarizer().get_intensity_on_pixel(
            degree_of_polarization=np.full_like(radiance, 0.5),
            angle_of_polarization=np.zeros_like(radiance),
            radiance=radiance,
        )


def test_positive_infinite_radiance_respects_exact_extinction_without_nan():
    radiance = np.full((1, 2, 2), np.inf, dtype=np.float32)

    with np.errstate(invalid="raise"):
        intensity = make_perfect_polarizer().get_intensity_on_pixel(
            degree_of_polarization=np.ones_like(radiance),
            angle_of_polarization=np.zeros_like(radiance),
            radiance=radiance,
        )

    assert np.isposinf(intensity[0, 0, 0])
    assert np.isposinf(intensity[0, 0, 1])
    assert intensity[0, 1, 0] == 0.0
    assert np.isposinf(intensity[0, 1, 1])
    assert not np.isnan(intensity).any()


def test_nan_masks_propagate_without_mutating_sensor_state_inputs():
    dop = np.full((1, 2, 2), 0.5, dtype=np.float32)
    aop = np.zeros_like(dop)
    radiance = np.ones_like(dop)
    dop[0, 0, 0] = np.nan
    aop[0, 0, 1] = np.nan
    radiance[0, 1, 0] = np.nan
    originals = [array.copy() for array in (dop, aop, radiance)]

    intensity = make_perfect_polarizer().get_intensity_on_pixel(dop, aop, radiance)

    expected_mask = np.isnan(dop) | np.isnan(aop) | np.isnan(radiance)
    np.testing.assert_array_equal(np.isnan(intensity), expected_mask)
    assert np.isfinite(intensity[~expected_mask]).all()
    for actual, original in zip((dop, aop, radiance), originals, strict=False):
        np.testing.assert_array_equal(actual, original)
