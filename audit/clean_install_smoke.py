"""Public-API smoke test intended to run outside the source tree."""

from __future__ import annotations

import json
import warnings
from importlib import metadata, resources

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

import pyskylumos
from pyskylumos.engine import Engine
from pyskylumos.sensor import SlicingPattern

WIRE_GRID = {
    0: SlicingPattern(0, 0, 2),
    45: SlicingPattern(0, 1, 2),
    90: SlicingPattern(1, 0, 2),
    135: SlicingPattern(1, 1, 2),
}


def make_engine(seed: int = 0) -> Engine:
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
        random_seed=seed,
    )


def main() -> None:
    installed_version = metadata.version("pyskylumos")
    if pyskylumos.__version__ != installed_version:
        raise RuntimeError(
            f"Version mismatch: import={pyskylumos.__version__}, metadata={installed_version}."
        )
    if not resources.files("pyskylumos").joinpath("py.typed").is_file():
        raise RuntimeError("The installed distribution does not contain py.typed.")

    time = Time(["2026-06-21T12:00:00"])
    location = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
    sun = SkyCoord(az=[137.0] * deg, alt=[33.0] * deg, frame=AltAz(obstime=time, location=location))
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, names = engine.simulate_sky_polarization(
                sky_model=model,
                observation_location=location,
                times=time,
                cie_sky_type=4,
                altitudes=altitudes,
                azimuths=azimuths,
                sun_position=sun,
                sensor_azimuthal_tilt_radians=0.37,
                sensor_tilt_angle_radians=0.18,
            )
        sky = dict(zip(names, values, strict=False))
        measurement = engine.simulate_measurement(
            degree_of_polarization=sky["degree of polarization"],
            angle_of_polarization=sky["angle of polarization"],
            radiance=sky["radiance"],
        )
        results[model] = {
            "sky_shape": list(sky["degree of polarization"].shape),
            "measurement_shape": list(measurement["dop"].shape),
            "sky_finite": int(np.isfinite(sky["degree of polarization"]).sum()),
            "measurement_finite": int(np.isfinite(measurement["dop"]).sum()),
            "measurement_dop_min": float(np.nanmin(measurement["dop"])),
            "measurement_dop_max": float(np.nanmax(measurement["dop"])),
        }
    print(
        json.dumps(
            {"package_version": installed_version, "models": results},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
