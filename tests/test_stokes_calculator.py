import numpy as np

from pyskylumos.sensor.SlicingPattern import SlicingPattern
from pyskylumos.sensor.StokesCalculator import StokesCalculator


def test_stokes_calculator_recovers_simple_signal():
    wire_grid_orientations_slicing = {
        0: SlicingPattern(start_row=0, start_column=0, step=2),
        45: SlicingPattern(start_row=0, start_column=1, step=2),
        90: SlicingPattern(start_row=1, start_column=0, step=2),
        135: SlicingPattern(start_row=1, start_column=1, step=2),
    }
    calculator = StokesCalculator(wire_grid_orientations_slicing=wire_grid_orientations_slicing)

    bits_intensity = np.array([[
        [4.0, 2.0],
        [0.0, 2.0],
    ]], dtype=np.float32)

    dop, aop = calculator.simulate_measurements(bits_intensity=bits_intensity)

    assert np.allclose(dop, 1.0)
    assert np.allclose(aop, 0.0)
