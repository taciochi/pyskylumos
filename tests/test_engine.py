"""Tests for Engine construction, argument hygiene and reproducibility."""

import warnings

import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.engine import Engine
from pyskylumos.exceptions import ConfigurationError, InputValidationError
from pyskylumos.sensor import OpticalConjugator, SlicingPattern
from pyskylumos.sky_models import NeutralPointRangeWarning, Pan, PanFidelityWarning

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
SUN = SkyCoord(az=[137.0] * deg, alt=[33.0] * deg, frame=AltAz(obstime=TIMES, location=LOCATION))

WIRE_GRID = {
    0: SlicingPattern(start_row=0, start_column=0, step=2),
    45: SlicingPattern(start_row=0, start_column=1, step=2),
    90: SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}


def make_engine(
    vertical=16,
    horizontal=16,
    random_seed=None,
    multiplicative_noise_snr=50,
    lens_conjugation_type="thin",
):
    """Return an engine with the standard 2x2 mosaic."""
    return Engine(
        sensor_pixel_pitch_micrometers=2.2,
        lens_conjugation_type=lens_conjugation_type,
        number_pixels_vertical=vertical,
        number_pixels_horizontal=horizontal,
        lens_focal_length_micrometers=3500,
        polarizer_tolerance_radians=0.05,
        extinction_ratio=0.99,
        auto_exposure_saturation_fraction=0.9,
        adc_resolution_bits=12,
        multiplicative_noise_snr=multiplicative_noise_snr,
        wire_grid_orientations_slicing=WIRE_GRID,
        random_seed=random_seed,
    )


def direction_from_degrees(azimuths, altitudes):
    """Return independent Cartesian unit vectors for degree-valued directions."""
    azimuths_rad = np.deg2rad(azimuths)
    altitudes_rad = np.deg2rad(altitudes)
    return np.stack(
        (
            np.cos(altitudes_rad) * np.cos(azimuths_rad),
            np.cos(altitudes_rad) * np.sin(azimuths_rad),
            np.sin(altitudes_rad),
        ),
        axis=-1,
    )


def rodrigues_tilt_oracle(azimuths, altitudes, azimuthal_tilt, tilt_angle):
    """Rotate directions independently with the public tilt convention."""
    vectors = direction_from_degrees(azimuths, altitudes)
    axis = np.array(
        [np.cos(azimuthal_tilt), -np.sin(azimuthal_tilt), 0.0],
        dtype=np.float64,
    )
    return (
        vectors * np.cos(tilt_angle)
        + np.cross(axis, vectors) * np.sin(tilt_angle)
        + axis * np.sum(vectors * axis, axis=-1, keepdims=True) * (1.0 - np.cos(tilt_angle))
    )


def measure(engine, sky_model="QUEEN"):
    """Run a full sky simulation and measurement through the engine."""
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        values, names = engine.simulate_sky_polarization(
            sky_model=sky_model,
            observation_location=LOCATION,
            times=TIMES,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=SUN,
        )
    sky = dict(zip(names, values, strict=False))
    return engine.simulate_measurement(
        degree_of_polarization=sky["degree of polarization"],
        angle_of_polarization=sky["angle of polarization"],
        radiance=sky["radiance"],
    )


# --------------------------------------------------------------------------- #
# A11 -- sensor geometry is validated at construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("vertical, horizontal", [(2, 2), (16, 16), (16, 64), (64, 16)])
def test_valid_sensor_geometry_is_accepted(vertical, horizontal):
    assert make_engine(vertical, horizontal) is not None


@pytest.mark.parametrize(
    "vertical, horizontal, offending",
    [
        (5, 16, "number_pixels_vertical"),
        (16, 5, "number_pixels_horizontal"),
        (15, 15, "number_pixels_vertical"),
    ],
)
def test_odd_sensor_geometry_is_rejected_at_construction(vertical, horizontal, offending):
    with pytest.raises(ValueError, match=offending):
        make_engine(vertical, horizontal)


def test_geometry_error_arrives_before_any_simulation():
    """The message must name the constructor argument, not a downstream shape."""
    with pytest.raises(ValueError, match="divisible by the micro-polarizer mosaic step"):
        make_engine(vertical=15, horizontal=16)


# --------------------------------------------------------------------------- #
# A12 -- the engine never writes to a caller's array
# --------------------------------------------------------------------------- #

AUDIT_AZIMUTHS = np.array([[0.0, 45.0, 90.0], [-30.0, 120.0, -170.0]])
AUDIT_ALTITUDES = np.array([[15.0, 45.0, 80.0], [5.0, 30.0, 60.0]])


@pytest.mark.parametrize(
    "dtype, tolerance_deg",
    [(np.float64, 1e-12), (np.float32, 1e-5)],
)
def test_zero_tilt_is_the_identity(dtype, tolerance_deg):
    azimuths = AUDIT_AZIMUTHS.astype(dtype)
    altitudes = AUDIT_ALTITUDES.astype(dtype)

    tilted_azimuths, tilted_altitudes = make_engine().tilt_sensor(
        azimuths=azimuths,
        altitudes=altitudes,
        azimuthal_tilt=0.0,
        tilt_angle=0.0,
    )

    azimuth_error = (tilted_azimuths - azimuths + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(azimuth_error, 0.0, atol=tolerance_deg)
    np.testing.assert_allclose(tilted_altitudes, altitudes, atol=tolerance_deg)


@pytest.mark.parametrize(
    "azimuthal_tilt, tilt_angle",
    [(0.0, 0.37), (0.73, -0.41), (-1.2, np.pi / 2)],
)
def test_tilt_matches_an_independent_rodrigues_oracle(azimuthal_tilt, tilt_angle):
    actual_azimuths, actual_altitudes = make_engine().tilt_sensor(
        azimuths=AUDIT_AZIMUTHS,
        altitudes=AUDIT_ALTITUDES,
        azimuthal_tilt=azimuthal_tilt,
        tilt_angle=tilt_angle,
    )

    actual_vectors = direction_from_degrees(actual_azimuths, actual_altitudes)
    expected_vectors = rodrigues_tilt_oracle(
        AUDIT_AZIMUTHS,
        AUDIT_ALTITUDES,
        azimuthal_tilt,
        tilt_angle,
    )
    np.testing.assert_allclose(actual_vectors, expected_vectors, atol=1e-12)


def test_positive_north_axis_tilt_preserves_the_existing_handedness():
    azimuths = np.array([0.0, 90.0, 0.0])
    altitudes = np.array([0.0, 0.0, 90.0])

    tilted_azimuths, tilted_altitudes = make_engine().tilt_sensor(
        azimuths=azimuths,
        altitudes=altitudes,
        azimuthal_tilt=0.0,
        tilt_angle=np.pi / 2,
    )

    actual_vectors = direction_from_degrees(tilted_azimuths, tilted_altitudes)
    expected_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(actual_vectors, expected_vectors, atol=1e-12)


def test_tilt_followed_by_its_inverse_recovers_the_directions():
    engine = make_engine()
    azimuthal_tilt = 0.63
    tilt_angle = 0.37

    tilted_azimuths, tilted_altitudes = engine.tilt_sensor(
        AUDIT_AZIMUTHS,
        AUDIT_ALTITUDES,
        azimuthal_tilt,
        tilt_angle,
    )
    recovered_azimuths, recovered_altitudes = engine.tilt_sensor(
        tilted_azimuths,
        tilted_altitudes,
        azimuthal_tilt,
        -tilt_angle,
    )

    azimuth_error = (recovered_azimuths - AUDIT_AZIMUTHS + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(azimuth_error, 0.0, atol=1e-12)
    np.testing.assert_allclose(recovered_altitudes, AUDIT_ALTITUDES, atol=1e-12)


def test_tilt_preserves_pairwise_angular_separations():
    initial_vectors = direction_from_degrees(AUDIT_AZIMUTHS, AUDIT_ALTITUDES).reshape(-1, 3)
    tilted_azimuths, tilted_altitudes = make_engine().tilt_sensor(
        AUDIT_AZIMUTHS,
        AUDIT_ALTITUDES,
        azimuthal_tilt=-0.92,
        tilt_angle=0.48,
    )
    tilted_vectors = direction_from_degrees(tilted_azimuths, tilted_altitudes).reshape(-1, 3)

    np.testing.assert_allclose(
        tilted_vectors @ tilted_vectors.T,
        initial_vectors @ initial_vectors.T,
        atol=1e-12,
    )


def test_tilt_preserves_shape_inputs_and_nan_mask():
    azimuths = np.array([[0.0, np.nan], [45.0, 90.0]])
    altitudes = np.array([[10.0, 20.0], [np.nan, 40.0]])
    original_azimuths = azimuths.copy()
    original_altitudes = altitudes.copy()

    tilted_azimuths, tilted_altitudes = make_engine().tilt_sensor(
        azimuths,
        altitudes,
        azimuthal_tilt=0.3,
        tilt_angle=-0.2,
    )

    expected_mask = np.isnan(azimuths) | np.isnan(altitudes)
    assert tilted_azimuths.shape == azimuths.shape
    assert tilted_altitudes.shape == altitudes.shape
    np.testing.assert_array_equal(np.isnan(tilted_azimuths), expected_mask)
    np.testing.assert_array_equal(np.isnan(tilted_altitudes), expected_mask)
    assert np.isfinite(tilted_azimuths[~expected_mask]).all()
    assert np.isfinite(tilted_altitudes[~expected_mask]).all()
    np.testing.assert_array_equal(azimuths, original_azimuths)
    np.testing.assert_array_equal(altitudes, original_altitudes)


def test_local_meridian_aop_conversion_handles_cardinal_azimuths_and_axial_wrap():
    aop = np.zeros((1, 5))
    azimuths = np.array([[0.0, 90.0, 180.0, 270.0, 360.0]])

    converted = Engine.convert_local_meridian_aop_to_sensor(aop, azimuths)

    np.testing.assert_allclose(
        converted,
        [[0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0]],
        atol=1e-15,
    )
    assert np.all(converted >= -np.pi / 2)
    assert np.all(converted < np.pi / 2)


def test_local_meridian_aop_conversion_broadcasts_preserves_nans_and_inputs():
    aop = np.array([[0.2], [np.nan]])
    azimuths = np.array([[10.0, 170.0, 350.0]])
    original_aop = aop.copy()
    original_azimuths = azimuths.copy()
    expected = (aop + np.deg2rad(azimuths) + np.pi / 2) % np.pi - np.pi / 2

    converted = Engine.convert_local_meridian_aop_to_sensor(aop, azimuths)

    assert converted.shape == (2, 3)
    np.testing.assert_allclose(converted, expected, atol=1e-15, equal_nan=True)
    np.testing.assert_array_equal(aop, original_aop)
    np.testing.assert_array_equal(azimuths, original_azimuths)


def test_rotate_sensor_leaves_the_callers_array_untouched():
    azimuths = np.array([[10.0, 20.0], [-170.0, 175.0]])
    original = azimuths.copy()

    rotated = Engine.rotate_sensor(azimuths=azimuths, rotation_angle=30.0)

    np.testing.assert_array_equal(azimuths, original)
    np.testing.assert_allclose(rotated, [[40.0, 50.0], [-140.0, -155.0]])


def test_rotate_sensor_wraps_into_the_half_open_range():
    rotated = Engine.rotate_sensor(azimuths=np.array([170.0, -170.0]), rotation_angle=20.0)

    assert np.all(rotated >= -180.0)
    assert np.all(rotated < 180.0)


def test_rotate_sensor_without_an_angle_is_a_no_op():
    azimuths = np.array([[10.0, 20.0]])

    assert Engine.rotate_sensor(azimuths=azimuths, rotation_angle=None) is azimuths
    assert Engine.rotate_sensor(azimuths=azimuths, rotation_angle=0) is azimuths


def test_azimuth_rotation_leaves_the_sky_models_array_untouched():
    engine = make_engine()
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)

    unrotated, _names = engine.simulate_sky_polarization(
        sky_model="QUEEN",
        observation_location=LOCATION,
        times=TIMES,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=SUN,
    )
    baseline = unrotated[1].copy()

    rotated, _ = engine.simulate_sky_polarization(
        sky_model="QUEEN",
        observation_location=LOCATION,
        times=TIMES,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=SUN,
        azimuth_rotation_angle=25.0,
    )

    np.testing.assert_array_equal(unrotated[1], baseline)
    assert not np.allclose(rotated[1], baseline, equal_nan=True)


def test_pan_engine_measurement_matches_independently_converted_direct_pan():
    engine_path = make_engine(random_seed=4)
    oracle_path = make_engine(random_seed=4)
    azimuths, altitudes = engine_path.get_initial_azimuth_altitude(altitude_min_clip=0)

    engine_values, names = engine_path.simulate_sky_polarization(
        sky_model="PAN",
        observation_location=LOCATION,
        times=TIMES,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=SUN,
    )
    direct_model = Pan(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        direct_values = direct_model.simulate_sky(cie_sky_type=4, sun_position=SUN)

    engine_sky = dict(zip(names, engine_values, strict=False))
    expected_aop = (direct_values[1] + np.deg2rad(azimuths) + np.pi / 2) % np.pi - np.pi / 2
    engine_measurement = engine_path.simulate_measurement(
        degree_of_polarization=engine_sky["degree of polarization"],
        angle_of_polarization=engine_sky["angle of polarization"],
        radiance=engine_sky["radiance"],
    )
    oracle_measurement = oracle_path.simulate_measurement(
        degree_of_polarization=direct_values[0],
        angle_of_polarization=expected_aop,
        radiance=direct_values[2],
    )

    np.testing.assert_array_equal(engine_measurement["dop"], oracle_measurement["dop"])
    np.testing.assert_array_equal(engine_measurement["aop"], oracle_measurement["aop"])


def test_get_initial_azimuth_altitude_does_not_hand_out_shared_state():
    engine = make_engine()

    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=None)
    azimuths[0, 0] = 999.0
    altitudes[0, 0] = -999.0

    fresh_azimuths, fresh_altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=None)

    assert fresh_azimuths[0, 0] != 999.0
    assert fresh_altitudes[0, 0] != -999.0


# --------------------------------------------------------------------------- #
# A14, A16 -- measurement works and is reproducible
# --------------------------------------------------------------------------- #


def test_measurement_completes_and_has_the_expected_shape():
    measurement = measure(make_engine(random_seed=0))

    assert measurement["dop"].shape == (1, 8, 8)
    assert measurement["aop"].shape == (1, 8, 8)


def test_measured_values_are_physical():
    measurement = measure(make_engine(random_seed=0))

    dop, aop = measurement["dop"], measurement["aop"]
    finite = np.isfinite(dop)

    assert finite.any()
    assert np.all(dop[finite] >= 0.0)
    assert np.all(dop[finite] <= 1.0)
    assert np.all(np.abs(aop[np.isfinite(aop)]) <= np.pi / 2 + 1e-6)


@pytest.mark.parametrize("multiplicative_noise_snr", [0.5, 2])
def test_low_snr_measurements_are_bounded_and_seeded(multiplicative_noise_snr):
    shape = (1, 64, 64)
    dop = np.full(shape, 0.9, dtype=np.float32)
    aop = np.full(shape, np.deg2rad(37.0), dtype=np.float32)
    radiance = np.ones(shape, dtype=np.float32)

    first = make_engine(
        vertical=64,
        horizontal=64,
        random_seed=0,
        multiplicative_noise_snr=multiplicative_noise_snr,
    ).simulate_measurement(dop, aop, radiance)
    second = make_engine(
        vertical=64,
        horizontal=64,
        random_seed=0,
        multiplicative_noise_snr=multiplicative_noise_snr,
    ).simulate_measurement(dop, aop, radiance)

    finite = first["dop"][np.isfinite(first["dop"])]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
    assert np.any(finite == 1.0)
    np.testing.assert_array_equal(first["dop"], second["dop"])
    np.testing.assert_array_equal(first["aop"], second["aop"])


def test_positive_infinite_radiance_is_accepted_as_full_frame_over_range():
    shape = (1, 16, 16)

    measurement = make_engine(random_seed=0).simulate_measurement(
        degree_of_polarization=np.full(shape, 0.5, dtype=np.float32),
        angle_of_polarization=np.zeros(shape, dtype=np.float32),
        radiance=np.full(shape, np.inf, dtype=np.float32),
    )

    np.testing.assert_array_equal(measurement["dop"], np.zeros((1, 8, 8), dtype=np.float32))
    assert np.isfinite(measurement["aop"]).all()


@pytest.mark.parametrize(
    "parameter, invalid_value, message",
    [
        ("degree_of_polarization", -0.1, r"degree_of_polarization.*\[0, 1\]"),
        ("degree_of_polarization", 1.1, r"degree_of_polarization.*\[0, 1\]"),
        ("degree_of_polarization", np.inf, "degree_of_polarization.*infinity"),
        ("angle_of_polarization", np.inf, "angle_of_polarization.*infinity"),
        ("radiance", -0.1, "radiance must be non-negative"),
        ("radiance", -np.inf, "radiance must be non-negative"),
    ],
)
def test_measurement_rejects_nonphysical_sensor_states(parameter, invalid_value, message):
    inputs = {
        "degree_of_polarization": np.full((1, 16, 16), 0.5, dtype=np.float32),
        "angle_of_polarization": np.zeros((1, 16, 16), dtype=np.float32),
        "radiance": np.ones((1, 16, 16), dtype=np.float32),
    }
    inputs[parameter][0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match=message):
        make_engine(random_seed=0).simulate_measurement(**inputs)


def test_identical_seeds_reproduce_a_measurement():
    first = measure(make_engine(random_seed=11))
    second = measure(make_engine(random_seed=11))

    np.testing.assert_array_equal(first["dop"], second["dop"])
    np.testing.assert_array_equal(first["aop"], second["aop"])


def test_different_seeds_produce_different_measurements():
    first = measure(make_engine(random_seed=11))
    second = measure(make_engine(random_seed=12))

    assert not np.array_equal(first["dop"], second["dop"])


def test_an_unseeded_engine_still_measures():
    measurement = measure(make_engine())

    assert np.isfinite(measurement["dop"]).any()


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #


def test_times_must_be_an_astropy_time():
    engine = make_engine()
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)

    with pytest.raises(TypeError, match=r"times must be an astropy\.time\.Time"):
        engine.simulate_sky_polarization(
            sky_model="QUEEN",
            observation_location=LOCATION,
            times="2026-06-21T12:00:00",
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
        )


# --------------------------------------------------------------------------- #
# custom lens conjugation: the current name and its deprecated alias
# --------------------------------------------------------------------------- #


def constant_projection(*, complex_sensor_plane, lens_focal_length_micrometers):
    """Return a hook that maps every pixel to a 45 degree zenith angle."""
    assert lens_focal_length_micrometers == 3500.0
    return np.full(complex_sensor_plane.shape, np.pi / 4, dtype=np.float64)


def make_custom_engine():
    """Return an engine whose conjugator expects a custom projection hook."""
    return make_engine(vertical=4, horizontal=6, lens_conjugation_type="custom")


def test_custom_lens_conjugation_hook_is_forwarded_to_the_conjugator():
    azimuths, altitudes = make_custom_engine().get_initial_azimuth_altitude(
        altitude_min_clip=None, custom_lens_conjugation=constant_projection
    )

    conjugator = OpticalConjugator(
        lens_conjugation_type="custom",
        number_pixels_vertical=4,
        number_pixels_horizontal=6,
        lens_focal_length_micrometers=3500,
        sensor_pixel_pitch_micrometers=2.2,
    )
    expected_azimuths, expected_altitudes = conjugator.get_azimuth_altitude(
        altitude_min_clip=None, custom_lens_conjugation=constant_projection
    )

    np.testing.assert_array_equal(azimuths, expected_azimuths)
    np.testing.assert_array_equal(altitudes, expected_altitudes)
    np.testing.assert_allclose(altitudes, 45.0)


def test_custom_lens_conjugation_hook_is_accepted_positionally_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        azimuths, altitudes = make_custom_engine().get_initial_azimuth_altitude(
            None, constant_projection
        )

    keyword_azimuths, keyword_altitudes = make_custom_engine().get_initial_azimuth_altitude(
        altitude_min_clip=None, custom_lens_conjugation=constant_projection
    )
    np.testing.assert_array_equal(azimuths, keyword_azimuths)
    np.testing.assert_array_equal(altitudes, keyword_altitudes)


def test_deprecated_custom_lens_conjugation_type_still_works_and_warns():
    with pytest.warns(DeprecationWarning, match="custom_lens_conjugation_type is deprecated"):
        legacy_azimuths, legacy_altitudes = make_custom_engine().get_initial_azimuth_altitude(
            altitude_min_clip=None, custom_lens_conjugation_type=constant_projection
        )

    current_azimuths, current_altitudes = make_custom_engine().get_initial_azimuth_altitude(
        altitude_min_clip=None, custom_lens_conjugation=constant_projection
    )
    np.testing.assert_array_equal(legacy_azimuths, current_azimuths)
    np.testing.assert_array_equal(legacy_altitudes, current_altitudes)


def test_conflicting_custom_lens_conjugation_names_are_rejected():
    with pytest.raises(ConfigurationError, match="Pass only 'custom_lens_conjugation'"):
        make_custom_engine().get_initial_azimuth_altitude(
            altitude_min_clip=None,
            custom_lens_conjugation=constant_projection,
            custom_lens_conjugation_type=constant_projection,
        )


def test_omitting_both_custom_lens_conjugation_names_passes_no_hook():
    """Neither name supplied must resolve to None, not to the UNSET sentinel."""
    with pytest.raises(InputValidationError, match="requires a custom_lens_conjugation function"):
        make_custom_engine().get_initial_azimuth_altitude(altitude_min_clip=None)


def test_non_custom_conjugation_ignores_the_absent_hook():
    engine = make_engine(vertical=4, horizontal=6)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)

    assert azimuths.shape == (4, 6)
    assert np.isfinite(altitudes).all()
