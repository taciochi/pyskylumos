"""Tests for the optical conjugator's pixel-to-sky mapping and orientation."""

import numpy as np
import pytest

from pyskylumos.sensor.OpticalConjugator import OpticalConjugator

PIXELS = 65
CENTRE = PIXELS // 2

CONJUGATION_TYPES = ["thin", "stereographic", "equi_angle", "equi_solid_angle", "orthogonal"]


def make_conjugator(
    conjugation_type="thin", vertical=PIXELS, horizontal=PIXELS, focal_length=3500.0
):
    """Return a conjugator with a square, odd-sized pixel grid."""
    return OpticalConjugator(
        lens_conjugation_type=conjugation_type,
        number_pixels_vertical=vertical,
        number_pixels_horizontal=horizontal,
        lens_focal_length_micrometers=focal_length,
        sensor_pixel_pitch_micrometers=2.2,
    )


def test_optical_conjugator_outputs_shapes():
    conjugator = make_conjugator(vertical=4, horizontal=6)

    azimuths, altitudes = conjugator.get_azimuth_altitude(altitude_min_clip=None)

    assert azimuths.shape == (4, 6)
    assert altitudes.shape == (4, 6)
    assert np.isfinite(azimuths).all()
    assert np.isfinite(altitudes).all()


# --------------------------------------------------------------------------- #
# C1-C4 -- cardinal orientation: North right, East top, South left, West bottom
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "row, column, expected_azimuth_deg, cardinal",
    [
        (CENTRE, PIXELS - 1, 0.0, "North at image right"),
        (0, CENTRE, 90.0, "East at image top"),
        (CENTRE, 0, 180.0, "South at image left"),
        (PIXELS - 1, CENTRE, -90.0, "West at image bottom"),
    ],
)
def test_cardinal_directions_sit_where_the_convention_says(
    row, column, expected_azimuth_deg, cardinal
):
    """PySkyLumos renders the sky with North at the right and East at the top.

    Note that this is the mirror image of Pan Eq. (20), whose azimuth runs
    clockwise from image-up. Angles of polarization cannot be carried between the
    two conventions without a reflection, which flips their sign.
    """
    azimuths, _ = make_conjugator().get_azimuth_altitude(altitude_min_clip=None)

    assert azimuths[row, column] == pytest.approx(expected_azimuth_deg, abs=1e-4), cardinal


def test_azimuth_increases_counter_clockwise_in_display_order():
    """Walking North -> East -> South, the azimuth rises monotonically."""
    azimuths, _ = make_conjugator().get_azimuth_altitude(altitude_min_clip=None)

    north = azimuths[CENTRE, PIXELS - 1]
    east = azimuths[0, CENTRE]
    south = azimuths[CENTRE, 0]

    assert north < east < south

    upper_right_quadrant = azimuths[:CENTRE, CENTRE + 1 :]
    assert np.all(upper_right_quadrant > 0.0)
    assert np.all(upper_right_quadrant < 90.0)


def test_azimuth_is_a_four_quadrant_angle():
    azimuths, _ = make_conjugator().get_azimuth_altitude(altitude_min_clip=None)

    assert azimuths.min() >= -180.0
    assert azimuths.max() <= 180.0
    assert np.any(azimuths < 0.0)
    assert np.any(azimuths > 0.0)


# --------------------------------------------------------------------------- #
# C6 -- altitude falls monotonically from the zenith
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("conjugation_type", CONJUGATION_TYPES)
def test_altitude_peaks_at_the_zenith_and_falls_with_radius(conjugation_type):
    focal_length = 3500.0 if conjugation_type != "equi_angle" else 2.2 * CENTRE / np.deg2rad(70.0)
    _, altitudes = make_conjugator(
        conjugation_type, focal_length=focal_length
    ).get_azimuth_altitude(altitude_min_clip=None)

    assert altitudes[CENTRE, CENTRE] == pytest.approx(90.0, abs=1e-4)

    row = altitudes[CENTRE, CENTRE:]
    assert np.all(np.diff(row) < 0.0)
    assert np.isfinite(altitudes).all()


def test_pan_equidistant_camera_mapping_is_the_equi_angle_conjugation():
    """Pan Eqs. (18)-(19) describe an equidistant fisheye.

    Read as printed, Eq. (19) is missing its normalisation by ``n / 2``, and the
    symbol called "the field of the camera view" has to be the *half* field of
    view for the zenith angle to reach it at the image edge. With those two
    readings, the conjugator's ``equi_angle`` mode reproduces the mapping.
    """
    half_field_of_view_deg = 70.0
    pixel_size = 2.2
    focal_length = pixel_size * CENTRE / np.deg2rad(half_field_of_view_deg)

    _, altitudes = make_conjugator("equi_angle", focal_length=focal_length).get_azimuth_altitude(
        altitude_min_clip=None
    )

    columns = np.arange(PIXELS) - CENTRE
    rows = CENTRE - np.arange(PIXELS)
    radius = np.hypot(*np.meshgrid(columns, rows))
    expected_zenith_deg = half_field_of_view_deg * radius / CENTRE

    np.testing.assert_allclose(90.0 - altitudes, expected_zenith_deg, atol=1e-4)


# --------------------------------------------------------------------------- #
# C7 -- the conjugator clips, while the sky models mask
# --------------------------------------------------------------------------- #


def make_wide_field_conjugator():
    """Return an equidistant conjugator whose corners fall below the horizon.

    A 70 degree half field of view across the half-width puts the diagonal at
    about 99 degrees of zenith angle, i.e. roughly 9 degrees below the horizon.
    """
    return make_conjugator("equi_angle", focal_length=2.2 * CENTRE / np.deg2rad(70.0))


def test_altitude_min_clip_clips_rather_than_masking():
    """A deliberate asymmetry: the conjugator clips, the sky models write NaN.

    The conjugator must return a usable direction for every pixel, so it clamps
    below-horizon altitudes to the floor. The sky models then mask the same
    pixels with NaN, which is what downstream consumers test for.
    """
    conjugator = make_wide_field_conjugator()
    _, unclipped = conjugator.get_azimuth_altitude(altitude_min_clip=None)
    _, clipped = conjugator.get_azimuth_altitude(altitude_min_clip=0.0)

    assert np.any(unclipped < 0.0)
    assert np.all(clipped >= 0.0)
    assert not np.any(np.isnan(clipped))
    np.testing.assert_allclose(clipped, np.clip(unclipped, 0.0, None), atol=1e-6)


def test_clipping_does_not_corrupt_the_cache():
    conjugator = make_wide_field_conjugator()

    _, clipped = conjugator.get_azimuth_altitude(altitude_min_clip=0.0)
    _, unclipped = conjugator.get_azimuth_altitude(altitude_min_clip=None)

    assert np.any(unclipped < 0.0)
    assert np.all(clipped >= 0.0)


# --------------------------------------------------------------------------- #
# the cache is never handed out by reference
# --------------------------------------------------------------------------- #


def test_returned_arrays_are_copies_not_the_cache():
    conjugator = make_conjugator()

    first_azimuth, first_altitude = conjugator.get_azimuth_altitude(altitude_min_clip=None)
    second_azimuth, second_altitude = conjugator.get_azimuth_altitude(altitude_min_clip=None)

    np.testing.assert_array_equal(first_azimuth, second_azimuth)
    np.testing.assert_array_equal(first_altitude, second_altitude)
    assert first_azimuth is not second_azimuth
    assert first_altitude is not second_altitude


def test_writing_to_a_returned_array_cannot_corrupt_later_calls():
    conjugator = make_conjugator()

    azimuth, altitude = conjugator.get_azimuth_altitude(altitude_min_clip=None)
    expected_azimuth = azimuth.copy()
    expected_altitude = altitude.copy()

    azimuth[:] = 999.0
    altitude[:] = -999.0

    fresh_azimuth, fresh_altitude = conjugator.get_azimuth_altitude(altitude_min_clip=None)

    np.testing.assert_array_equal(fresh_azimuth, expected_azimuth)
    np.testing.assert_array_equal(fresh_altitude, expected_altitude)


def test_unknown_conjugation_type_is_rejected():
    with pytest.raises(ValueError, match="lens_conjugation_type must be one of"):
        make_conjugator("fisheye")


def test_custom_conjugation_requires_a_callable():
    with pytest.raises(ValueError, match="requires a custom_lens_conjugation function"):
        make_conjugator("custom").get_azimuth_altitude(altitude_min_clip=None)


def test_custom_conjugation_hook_is_validated_and_used():
    conjugator = make_conjugator("custom", vertical=4, horizontal=6)

    def projection(*, complex_sensor_plane, lens_focal_length_micrometers):
        assert lens_focal_length_micrometers == 3500.0
        return np.full(complex_sensor_plane.shape, np.pi / 4, dtype=np.float32)

    azimuths, altitudes = conjugator.get_azimuth_altitude(
        altitude_min_clip=None, custom_lens_conjugation=projection
    )

    assert azimuths.dtype == np.float64
    assert altitudes.dtype == np.float64
    np.testing.assert_allclose(altitudes, 45.0)


@pytest.mark.parametrize(
    "hook, error_type, message",
    [
        (1, TypeError, "must be callable"),
        (lambda **_: [0.0], TypeError, "must return a numpy.ndarray"),
        (lambda **_: np.zeros((1, 1)), ValueError, "must preserve the sensor-plane shape"),
        (
            lambda *, complex_sensor_plane, **_: np.zeros_like(complex_sensor_plane),
            TypeError,
            "must return real floating-point radians",
        ),
        (
            lambda *, complex_sensor_plane, **_: np.full(complex_sensor_plane.shape, np.inf),
            ValueError,
            "must not contain infinity",
        ),
    ],
)
def test_custom_conjugation_rejects_invalid_hooks(hook, error_type, message):
    with pytest.raises(error_type, match=message):
        make_conjugator("custom", vertical=4, horizontal=6).get_azimuth_altitude(
            altitude_min_clip=None, custom_lens_conjugation=hook
        )


@pytest.mark.parametrize(
    "projection, message",
    [
        ("equi_solid_angle", "exceeds twice the focal length"),
        ("orthogonal", "exceeds the focal length"),
    ],
)
def test_projection_domain_failures_are_targeted(projection, message):
    with pytest.raises(ValueError, match=message):
        make_conjugator(
            projection, vertical=4, horizontal=4, focal_length=0.1
        ).get_azimuth_altitude(altitude_min_clip=None)


def test_preferred_and_deprecated_pitch_properties_match():
    conjugator = make_conjugator()

    assert conjugator.sensor_pixel_pitch_micrometers == 2.2
    with pytest.warns(DeprecationWarning, match="sensor_pixel_pitch_micrometers"):
        assert conjugator.sensor_pixel_size_square_micrometers == 2.2
