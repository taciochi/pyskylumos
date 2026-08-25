"""Exercise the README quick start for every documented model name.

Run this script from outside the source tree with a clean wheel installation.
It intentionally uses the README's ephemeris-derived 2024 Sun position rather
than the explicit Sun override used by ``clean_install_smoke.py``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern

WIRE_GRID = {
    0: SlicingPattern(start_row=0, start_column=0, step=2),
    45: SlicingPattern(start_row=0, start_column=1, step=2),
    90: SlicingPattern(start_row=1, start_column=0, step=2),
    135: SlicingPattern(start_row=1, start_column=1, step=2),
}


def make_engine() -> Engine:
    return Engine(
        sensor_pixel_pitch_micrometers=2.2,
        lens_conjugation_type="thin",
        number_pixels_vertical=64,
        number_pixels_horizontal=64,
        lens_focal_length_micrometers=3500,
        polarizer_tolerance_radians=0.0,
        extinction_ratio=0.99,
        auto_exposure_saturation_fraction=0.9,
        adc_resolution_bits=12,
        multiplicative_noise_snr=50,
        wire_grid_orientations_slicing=WIRE_GRID,
        random_seed=0,
    )


def main() -> None:
    results = {}
    for model in (
        "RAYLEIGH",
        "DEPOLARIZED_RAYLEIGH",
        "ASYMMETRIC",
        "BERRY",
        "PAN",
        "QUEEN",
    ):
        engine = make_engine()
        azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values, names = engine.simulate_sky_polarization(
                sky_model=model,
                observation_location=EarthLocation(lat=53.4, lon=-2.96, height=50),
                times=Time(["2024-07-01T12:00:00"]),
                cie_sky_type=4,
                altitudes=altitudes,
                azimuths=azimuths,
                model_options=(
                    {"dop_max": 1.0, "out_of_range": "ignore"}
                    if model in ("ASYMMETRIC", "QUEEN")
                    else {"depolarization_ratio": 0.0279}
                    if model == "DEPOLARIZED_RAYLEIGH"
                    else None
                ),
            )
        sky = dict(zip(names, values, strict=False))
        measurement = engine.simulate_measurement(
            degree_of_polarization=sky["degree of polarization"],
            angle_of_polarization=sky["angle of polarization"],
            radiance=sky["radiance"],
        )
        results[model] = {
            "measurement_shape": list(measurement["dop"].shape),
            "measurement_finite": bool(np.isfinite(measurement["dop"]).all()),
            "warnings": [type(item.message).__name__ for item in caught],
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
