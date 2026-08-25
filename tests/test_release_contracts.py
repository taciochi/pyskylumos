"""Release-hardening contracts for AUD-007 through AUD-013."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

import pyskylumos
from pyskylumos.engine import Engine
from pyskylumos.exceptions import ConfigurationError, InputTypeError, InputValidationError
from pyskylumos.sensor import SlicingPattern, StokesCalculator
from pyskylumos.sky_models import Rayleigh

WIRE_GRID = {
    0: SlicingPattern(0, 0, 2),
    45: SlicingPattern(0, 1, 2),
    90: SlicingPattern(1, 0, 2),
    135: SlicingPattern(1, 1, 2),
}
TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
SUN = SkyCoord(az=[137.0] * deg, alt=[33.0] * deg, frame=AltAz(obstime=TIMES, location=LOCATION))


def make_engine(**overrides):
    options = {
        "sensor_pixel_pitch_micrometers": 2.2,
        "lens_conjugation_type": "thin",
        "number_pixels_vertical": 4,
        "number_pixels_horizontal": 4,
        "lens_focal_length_micrometers": 3500,
        "polarizer_tolerance_radians": 0.0,
        "extinction_ratio": 0.99,
        "auto_exposure_saturation_fraction": 0.9,
        "adc_resolution_bits": 12,
        "multiplicative_noise_snr": 50,
        "wire_grid_orientations_slicing": WIRE_GRID,
        "random_seed": 4,
    }
    options.update(overrides)
    return Engine(**options)


def simulate(engine, model="RAYLEIGH", **overrides):
    azimuths, altitudes = engine.get_initial_azimuth_altitude(0)
    options = {
        "sky_model": model,
        "observation_location": LOCATION,
        "times": TIMES,
        "cie_sky_type": 4,
        "altitudes": altitudes,
        "azimuths": azimuths,
        "sun_position": SUN,
    }
    options.update(overrides)
    return engine.simulate_sky_polarization(**options)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"sensor_pixel_pitch_micrometers": 0}, "sensor_pixel_pitch_micrometers"),
        ({"lens_focal_length_micrometers": np.inf}, "lens_focal_length_micrometers"),
        ({"number_pixels_vertical": 4.5}, "number_pixels_vertical.*integer"),
        ({"number_pixels_horizontal": True}, "number_pixels_horizontal.*integer"),
        ({"polarizer_tolerance_radians": -0.1}, "polarizer_tolerance_radians"),
        ({"polarizer_tolerance_radians": np.pi}, "polarizer_tolerance_radians"),
        ({"extinction_ratio": 1.1}, "extinction_ratio"),
        ({"auto_exposure_saturation_fraction": 0}, "auto_exposure_saturation_fraction"),
        ({"adc_resolution_bits": 12.5}, "adc_resolution_bits.*integer"),
        ({"adc_resolution_bits": 25}, "adc_resolution_bits"),
        ({"multiplicative_noise_snr": 0}, "multiplicative_noise_snr"),
        ({"multiplicative_noise_snr": np.nan}, "multiplicative_noise_snr"),
        ({"random_seed": -1}, "random_seed"),
    ],
)
def test_engine_rejects_invalid_configuration(override, message):
    with pytest.raises(ConfigurationError, match=message):
        make_engine(**override)


def test_preferred_and_legacy_sensor_names_are_bit_identical():
    preferred = make_engine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = Engine(
            sensor_pixel_size_square_micrometers=2.2,
            lens_conjugation_type="thin",
            number_pixels_vertical=4,
            number_pixels_horizontal=4,
            lens_focal_length_micrometers=3500,
            tolerance=0.0,
            extinction_ratio=0.99,
            pixel_saturation_ratio=0.9,
            adc_resolution=12,
            signal_to_noise_ratio=50,
            wire_grid_orientations_slicing=WIRE_GRID,
            random_seed=4,
        )

    shape = (1, 4, 4)
    inputs = {
        "degree_of_polarization": np.full(shape, 0.7),
        "angle_of_polarization": np.full(shape, 0.3),
        "radiance": np.linspace(0.1, 1.0, 16).reshape(shape),
    }
    preferred_result = preferred.simulate_measurement(**inputs)
    legacy_result = legacy.simulate_measurement(**inputs)

    assert len([item for item in caught if issubclass(item.category, DeprecationWarning)]) == 5
    np.testing.assert_array_equal(preferred_result["dop"], legacy_result["dop"])
    np.testing.assert_array_equal(preferred_result["aop"], legacy_result["aop"])


def test_conflicting_preferred_and_legacy_aliases_are_rejected():
    with pytest.raises(ConfigurationError, match="Pass only 'sensor_pixel_pitch_micrometers'"):
        make_engine(sensor_pixel_size_square_micrometers=2.2)


def test_mosaic_must_be_complete_disjoint_and_two_by_two():
    missing = {key: value for key, value in WIRE_GRID.items() if key != 135}
    overlapping = dict(WIRE_GRID)
    overlapping[45] = SlicingPattern(0, 0, 2)

    with pytest.raises(ConfigurationError, match="exactly analyzer orientations"):
        StokesCalculator(missing)
    with pytest.raises(ConfigurationError, match="overlap"):
        StokesCalculator(overlapping)
    with pytest.raises(ConfigurationError, match="step"):
        SlicingPattern(0, 0, 0)


def test_measurement_rank_shape_and_sensor_dimensions_are_targeted():
    engine = make_engine()
    valid = np.ones((1, 4, 4))

    with pytest.raises(InputValidationError, match="rank 3"):
        engine.simulate_measurement(valid[0], valid, valid)
    with pytest.raises(InputValidationError, match="identical shapes"):
        engine.simulate_measurement(valid, valid[:, :, :-1], valid)
    with pytest.raises(InputValidationError, match="configured sensor dimensions"):
        engine.simulate_measurement(np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.ones((1, 2, 2)))


def test_nan_masks_from_different_inputs_propagate_as_their_union():
    shape = (1, 4, 4)
    dop = np.full(shape, 0.5)
    aop = np.zeros(shape)
    radiance = np.ones(shape)
    dop[0, 0, 0] = np.nan
    aop[0, 0, 1] = np.nan
    radiance[0, 1, 0] = np.nan

    measured = make_engine().simulate_measurement(dop, aop, radiance)

    assert np.isnan(measured["dop"][0, 0, 0])
    assert np.isnan(measured["aop"][0, 0, 0])


def test_model_cie_and_rotation_inputs_have_stable_errors():
    engine = make_engine()
    with pytest.raises(InputTypeError, match="sky_model must be a string"):
        simulate(engine, model=None)
    with pytest.raises(InputValidationError, match="1 to 15"):
        simulate(engine, cie_sky_type=16)
    with pytest.raises(InputValidationError, match="finite"):
        simulate(engine, azimuth_rotation_angle=np.inf)


@pytest.mark.parametrize("rotation", [720.0, -720.0, 1e9])
def test_aop_rotation_uses_true_modular_wrapping(rotation):
    values, _ = simulate(make_engine(), azimuth_rotation_angle=rotation)
    finite = values[1][np.isfinite(values[1])]
    assert np.all(finite >= -np.pi / 2)
    assert np.all(finite < np.pi / 2)


def test_parameter_names_are_immutable_and_all_sky_outputs_are_float64():
    engine = make_engine()
    for model in (
        "RAYLEIGH",
        "DEPOLARIZED_RAYLEIGH",
        "ASYMMETRIC",
        "BERRY",
        "PAN",
        "QUEEN",
    ):
        values, names = simulate(engine, model=model)
        assert isinstance(names, tuple)
        assert all(value.dtype == np.float64 for value in values)


def test_rayleigh_neutralities_are_nan_and_singular_grid_is_warning_free():
    azimuths = np.array([[137.0, -43.0, 90.0]])
    altitudes = np.array([[33.0, -33.0, 0.0]])
    model = Rayleigh(TIMES, LOCATION, altitudes=altitudes, azimuths=azimuths)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values = model.simulate_sky(cie_sky_type=4, sun_position=SUN)

    singular = values[0] <= 1e-15
    assert singular.sum() == 2
    assert np.isnan(values[1][singular]).all()
    assert np.isfinite(values[1][~singular]).all()


def test_public_version_and_exception_exports_exist():
    assert pyskylumos.__version__ == "0.1.1"
    assert issubclass(ConfigurationError, ValueError)
    assert issubclass(InputValidationError, ValueError)
    assert issubclass(InputTypeError, TypeError)
