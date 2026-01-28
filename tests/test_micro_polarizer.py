import numpy as np

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
        tolerance=0.0,
        wire_grid_orientations_slicing=wire_grid_orientations_slicing
    )

    radiance = np.ones((1, 2, 2), dtype=np.float32)
    dop = np.ones_like(radiance)
    aop = np.zeros_like(radiance)

    intensity = polarizer.get_intensity_on_pixel(
        degree_of_polarization=dop,
        angle_of_polarization=aop,
        radiance=radiance
    )

    expected = np.array([[
        [1.0, 0.5],
        [0.0, 0.5],
    ]], dtype=np.float32)

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
        tolerance=0.1,
        wire_grid_orientations_slicing=wire_grid_orientations_slicing,
        random_seed=42,
    )

    radiance = np.ones((1, 2, 2), dtype=np.float32)
    dop = np.ones_like(radiance)
    aop = np.zeros_like(radiance)

    first = polarizer.get_intensity_on_pixel(dop, aop, radiance)
    second = polarizer.get_intensity_on_pixel(dop, aop, radiance)

    assert np.allclose(first, second)
