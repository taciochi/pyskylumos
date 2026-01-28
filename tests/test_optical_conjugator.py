import numpy as np

from pyskylumos.sensor.OpticalConjugator import OpticalConjugator


def test_optical_conjugator_outputs_shapes():
    conjugator = OpticalConjugator(
        lens_conjugation_type="thin",
        number_pixels_vertical=4,
        number_pixels_horizontal=6,
        lens_focal_length_micrometers=3500,
        sensor_pixel_size_square_micrometers=2.2,
    )

    azimuths, altitudes = conjugator.get_azimuth_altitude(altitude_min_clip=None)

    assert azimuths.shape == (4, 6)
    assert altitudes.shape == (4, 6)
    assert np.isfinite(azimuths).all()
    assert np.isfinite(altitudes).all()
