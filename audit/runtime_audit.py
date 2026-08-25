"""Targeted runtime probes for public API and sensor-pipeline failure modes."""

from __future__ import annotations

import json
import warnings

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.engine import Engine
from pyskylumos.sensor import SensorChip, SlicingPattern, StokesCalculator
from pyskylumos.sky_models import Pan

WIRE_GRID = {
    0: SlicingPattern(0, 0, 2),
    45: SlicingPattern(0, 1, 2),
    90: SlicingPattern(1, 0, 2),
    135: SlicingPattern(1, 1, 2),
}


def make_engine(seed: int = 0, snr: float = 50) -> Engine:
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
        multiplicative_noise_snr=snr,
        wire_grid_orientations_slicing=WIRE_GRID,
        random_seed=seed,
    )


def captured_exception(callable_) -> dict[str, str] | None:
    try:
        callable_()
    except Exception as error:  # audit probe: exact public failure is evidence
        return {"type": type(error).__name__, "message": str(error)}
    return None


def main() -> None:
    results: dict[str, object] = {}

    engine = make_engine()
    input_azimuth = np.array([[0.0, 45.0, 90.0], [-30.0, 120.0, -170.0]])
    input_altitude = np.array([[15.0, 45.0, 80.0], [5.0, 30.0, 60.0]])
    tilted_azimuth, tilted_altitude = engine.tilt_sensor(
        input_azimuth, input_altitude, azimuthal_tilt=0.0, tilt_angle=0.0
    )
    results["zero_tilt_max_azimuth_error_deg"] = float(
        np.max(np.abs((tilted_azimuth - input_azimuth + 180) % 360 - 180))
    )
    results["zero_tilt_max_altitude_error_deg"] = float(
        np.max(np.abs(tilted_altitude - input_altitude))
    )

    time = Time(["2026-06-21T12:00:00"])
    location = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
    sun = SkyCoord(az=[137.0] * deg, alt=[33.0] * deg, frame=AltAz(obstime=time, location=location))
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=0)

    tilt_axis = 0.61
    tilt_angle = 0.29
    world_azimuths, world_altitudes = engine.tilt_sensor(
        azimuths,
        altitudes,
        azimuthal_tilt=tilt_axis,
        tilt_angle=tilt_angle,
    )
    recovered_azimuths, recovered_altitudes = engine.tilt_sensor(
        world_azimuths,
        world_altitudes,
        azimuthal_tilt=tilt_axis,
        tilt_angle=-tilt_angle,
    )
    results["nonzero_tilt_inverse_direction_max_error_deg"] = {
        "azimuth": float(
            np.nanmax(np.abs((recovered_azimuths - azimuths + 180.0) % 360.0 - 180.0))
        ),
        "altitude": float(np.nanmax(np.abs(recovered_altitudes - altitudes))),
    }

    world_values, _ = engine.simulate_sky_polarization(
        sky_model="RAYLEIGH",
        observation_location=location,
        times=time,
        cie_sky_type=4,
        altitudes=world_altitudes,
        azimuths=world_azimuths,
        sun_position=sun,
    )
    tilted_values, _ = engine.simulate_sky_polarization(
        sky_model="RAYLEIGH",
        observation_location=location,
        times=time,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=sun,
        sensor_azimuthal_tilt_radians=tilt_axis,
        sensor_tilt_angle_radians=tilt_angle,
    )
    tilt_aop_correction = 0.5 * np.arctan2(
        np.sin(2.0 * (tilted_values[1] - world_values[1])),
        np.cos(2.0 * (tilted_values[1] - world_values[1])),
    )
    results["nonzero_tilt_world_sampling_max_difference"] = {
        "dop": float(np.nanmax(np.abs(tilted_values[0] - world_values[0]))),
        "radiance": float(np.nanmax(np.abs(tilted_values[2] - world_values[2]))),
    }
    results["nonzero_tilt_aop_basis_correction_rad"] = {
        "min": float(np.nanmin(tilt_aop_correction)),
        "max": float(np.nanmax(tilt_aop_correction)),
        "standard_deviation": float(np.nanstd(tilt_aop_correction)),
    }

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        pan_values, names = engine.simulate_sky_polarization(
            sky_model="PAN",
            observation_location=location,
            times=time,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=sun,
        )
    pan = dict(zip(names, pan_values, strict=False))
    results["pan_engine_warnings"] = [type(item.message).__name__ for item in recorded]

    direct_pan = Pan(
        times=time,
        observation_location=location,
        azimuths=azimuths,
        altitudes=altitudes,
    )
    with warnings.catch_warnings(record=True) as direct_recorded:
        warnings.simplefilter("always")
        direct_values = direct_pan.simulate_sky(cie_sky_type=4, sun_position=sun)
    direct_frame_aop = (direct_values[1] + np.deg2rad(azimuths) + np.pi / 2) % np.pi - np.pi / 2
    results["pan_direct_warnings"] = [type(item.message).__name__ for item in direct_recorded]
    engine_vs_conversion = 0.5 * np.arctan2(
        np.sin(2 * (pan["angle of polarization"] - direct_frame_aop)),
        np.cos(2 * (pan["angle of polarization"] - direct_frame_aop)),
    )
    results["pan_engine_vs_explicit_conversion_aop_max_axial_difference_rad"] = float(
        np.nanmax(np.abs(engine_vs_conversion[direct_values[0] >= 1e-8]))
    )

    engine_measurement = make_engine(seed=4).simulate_measurement(
        pan["degree of polarization"], pan["angle of polarization"], pan["radiance"]
    )
    explicit_measurement = make_engine(seed=4).simulate_measurement(
        direct_values[0], direct_frame_aop, direct_values[2]
    )
    results["pan_engine_vs_explicit_conversion_measurement_aop_max_axial_difference_rad"] = float(
        np.nanmax(
            np.abs(
                0.5
                * np.arctan2(
                    np.sin(2 * (engine_measurement["aop"] - explicit_measurement["aop"])),
                    np.cos(2 * (engine_measurement["aop"] - explicit_measurement["aop"])),
                )
            )
        )
    )

    results["invalid_cie_type"] = captured_exception(
        lambda: engine.simulate_sky_polarization(
            sky_model="RAYLEIGH",
            observation_location=location,
            times=time,
            cie_sky_type=16,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=sun,
        )
    )
    results["none_sky_model"] = captured_exception(
        lambda: engine.simulate_sky_polarization(
            sky_model=None,
            observation_location=location,
            times=time,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            sun_position=sun,
        )
    )

    results["derived_sun_standard"] = captured_exception(
        lambda: engine.simulate_sky_polarization(
            sky_model="RAYLEIGH",
            observation_location=location,
            times=time,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            accuracy=False,
        )
    )
    results["derived_sun_jpl"] = captured_exception(
        lambda: engine.simulate_sky_polarization(
            sky_model="RAYLEIGH",
            observation_location=location,
            times=time,
            cie_sky_type=4,
            altitudes=altitudes,
            azimuths=azimuths,
            accuracy=True,
        )
    )

    rotated_values, _ = engine.simulate_sky_polarization(
        sky_model="RAYLEIGH",
        observation_location=location,
        times=time,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=sun,
        azimuth_rotation_angle=720.0,
    )
    results["aop_after_720_degree_rotation"] = {
        "min": float(np.nanmin(rotated_values[1])),
        "max": float(np.nanmax(rotated_values[1])),
        "outside_axial_range": int(np.sum(np.abs(rotated_values[1]) > np.pi / 2)),
    }

    first_values, immutable_names = engine.simulate_sky_polarization(
        sky_model="RAYLEIGH",
        observation_location=location,
        times=time,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=sun,
    )
    del first_values
    names_mutation = captured_exception(lambda: immutable_names.__setitem__(0, "caller mutation"))
    _, subsequent_names = engine.simulate_sky_polarization(
        sky_model="RAYLEIGH",
        observation_location=location,
        times=time,
        cie_sky_type=4,
        altitudes=altitudes,
        azimuths=azimuths,
        sun_position=sun,
    )
    results["parameter_names_are_immutable"] = (
        isinstance(immutable_names, tuple)
        and names_mutation is not None
        and subsequent_names[-1] != "caller mutation"
    )

    results["zero_snr_sensor"] = captured_exception(
        lambda: SensorChip(
            auto_exposure_saturation_fraction=0.9,
            adc_resolution_bits=12,
            multiplicative_noise_snr=0,
            random_seed=0,
        )
    )
    results["fractional_adc_resolution"] = captured_exception(
        lambda: SensorChip(
            auto_exposure_saturation_fraction=0.9,
            adc_resolution_bits=12.5,
            multiplicative_noise_snr=50,
            random_seed=0,
        )
    )
    noisy_dop_max = {}
    constant_radiance = np.ones((1, 64, 64), dtype=np.float32)
    constant_dop = np.full_like(constant_radiance, 0.9)
    constant_aop = np.full_like(constant_radiance, np.deg2rad(37.0))
    for snr in (0.5, 2, 10, 50, 1000):
        measurement = make_engine(seed=0, snr=snr).simulate_measurement(
            constant_dop, constant_aop, constant_radiance
        )
        noisy_dop_max[str(snr)] = {
            "min": float(np.nanmin(measurement["dop"])),
            "max": float(np.nanmax(measurement["dop"])),
            "count_above_one": int(np.sum(measurement["dop"] > 1.0)),
        }
    results["constant_state_measured_dop_by_snr"] = noisy_dop_max

    roundtrip_engine = Engine(
        sensor_pixel_pitch_micrometers=2.2,
        lens_conjugation_type="thin",
        number_pixels_vertical=64,
        number_pixels_horizontal=64,
        lens_focal_length_micrometers=3500,
        polarizer_tolerance_radians=0.0,
        extinction_ratio=1.0,
        auto_exposure_saturation_fraction=0.999999,
        adc_resolution_bits=24,
        multiplicative_noise_snr=1e12,
        wire_grid_orientations_slicing=WIRE_GRID,
        random_seed=0,
    )
    roundtrip = roundtrip_engine.simulate_measurement(constant_dop, constant_aop, constant_radiance)
    results["noise_free_roundtrip"] = {
        "dop_max_error": float(np.nanmax(np.abs(roundtrip["dop"] - 0.9))),
        "aop_max_axial_error_rad": float(
            np.nanmax(
                np.abs(
                    0.5
                    * np.arctan2(
                        np.sin(2 * (roundtrip["aop"] - np.deg2rad(37.0))),
                        np.cos(2 * (roundtrip["aop"] - np.deg2rad(37.0))),
                    )
                )
            )
        ),
    }

    immutable_dop = constant_dop.copy()
    immutable_aop = constant_aop.copy()
    immutable_radiance = constant_radiance.copy()
    make_engine(seed=0).simulate_measurement(immutable_dop, immutable_aop, immutable_radiance)
    results["measurement_inputs_unchanged"] = {
        "dop": bool(np.array_equal(immutable_dop, constant_dop, equal_nan=True)),
        "aop": bool(np.array_equal(immutable_aop, constant_aop, equal_nan=True)),
        "radiance": bool(np.array_equal(immutable_radiance, constant_radiance, equal_nan=True)),
    }

    results["mismatched_measurement_shapes"] = captured_exception(
        lambda: make_engine(seed=0).simulate_measurement(
            constant_dop,
            constant_aop[:, :, :-1],
            constant_radiance,
        )
    )

    nonphysical_results = {}
    for name, dop_input, radiance_input in (
        ("dop_above_one", np.full_like(constant_dop, 1.5), constant_radiance),
        ("negative_dop", np.full_like(constant_dop, -0.2), constant_radiance),
        ("negative_radiance", constant_dop, np.full_like(constant_radiance, -1.0)),
    ):
        try:
            measurement = make_engine(seed=0).simulate_measurement(
                dop_input, constant_aop, radiance_input
            )
            nonphysical_results[name] = {
                "accepted": True,
                "dop_finite": int(np.isfinite(measurement["dop"]).sum()),
                "dop_min": float(np.nanmin(measurement["dop"])),
                "dop_max": float(np.nanmax(measurement["dop"])),
            }
        except Exception as error:  # audit probe: exact public failure is evidence
            nonphysical_results[name] = {
                "accepted": False,
                "type": type(error).__name__,
                "message": str(error),
            }
    results["nonphysical_measurement_inputs"] = nonphysical_results

    malformed_mosaic_factories = {
        "missing_135": lambda: {key: value for key, value in WIRE_GRID.items() if key != 135},
        "overlapping": lambda: {
            0: SlicingPattern(0, 0, 2),
            45: SlicingPattern(0, 0, 2),
            90: SlicingPattern(1, 0, 2),
            135: SlicingPattern(1, 1, 2),
        },
        "zero_step": lambda: {
            0: SlicingPattern(0, 0, 0),
            45: SlicingPattern(0, 1, 2),
            90: SlicingPattern(1, 0, 2),
            135: SlicingPattern(1, 1, 2),
        },
    }
    mosaic_results = {}
    for name, factory in malformed_mosaic_factories.items():
        mosaic_results[name] = captured_exception(
            lambda f=factory: StokesCalculator(f()).validate_sensor_shape(4, 4)
        )
    results["malformed_mosaic_validation"] = mosaic_results

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
