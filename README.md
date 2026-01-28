# PySkyLumos

A Python package for simulating skylight polarization sensor recordings using advanced polarization models (including Pan, Berry, and Rayleigh). Designed for researchers and engineers, this tool enables the generation of synthetic datasets for biomimetic navigation, machine learning, computer vision, and atmospheric optics. Ideal for developing and testing bio-inspired sensors, building training data for AI models, and exploring applications in robotics, remote sensing, and environmental monitoring.

## Quick start

Below is a minimal example that builds an `Engine`, generates a sky polarization field, and simulates a sensor
measurement. Angles are in degrees unless otherwise stated, while AoP values are radians.

```python
from astropy.time import Time
from astropy.coordinates import EarthLocation

from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern

wire_grid_orientations_slicing = {
    0: SlicingPattern(start_row=0, start_column=0, step=2),
    45: SlicingPattern(start_row=0, start_column=1, step=2),
    90: SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}

engine = Engine(
    sensor_pixel_size_square_micrometers=2.2,
    lens_conjugation_type="thin",
    number_pixels_vertical=64,
    number_pixels_horizontal=64,
    lens_focal_length_micrometers=3500,
    tolerance=0.0,
    extinction_ratio=0.99,
    pixel_saturation_ratio=0.9,
    adc_resolution=12,
    signal_to_noise_ratio=50,
    wire_grid_orientations_slicing=wire_grid_orientations_slicing,
)

azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
observation_location = EarthLocation(lat=53.4, lon=-2.96, height=50)

sky_parameters, names = engine.simulate_sky_polarization(
    sky_model="rayleigh",
    observation_location=observation_location,
    times=Time("2024-07-01T12:00:00"),
    cie_sky_type=4,
    altitudes=altitudes,
    azimuths=azimuths,
)

sky = dict(zip(names, sky_parameters))
measurement = engine.simulate_measurement(
    degree_of_polarization=sky["degree of polarization"],
    angle_of_polarization=sky["angle of polarization"],
    radiance=sky["radiance"],
)

print(measurement["dop"].shape, measurement["aop"].shape)
```
