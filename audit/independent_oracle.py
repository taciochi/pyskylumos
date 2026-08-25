"""Independent numerical audit of the six PySkyLumos sky models.

The oracle below deliberately does not import any implementation helpers from
``pyskylumos.sky_models``.  It restates the scalar/vector equations from the
primary sources and compares them with the public model classes.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.sky_models import (
    AsymmetricQuartic,
    Berry,
    DepolarizedRayleigh,
    NeutralPointRangeWarning,
    Pan,
    PanFidelityWarning,
    QuEEN,
    Rayleigh,
)

TIMES = Time(["2026-06-21T12:00:00"])
LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
SUN_AZIMUTHS_DEG = (0.0, 37.0, 137.0, 275.0)
SUN_ELEVATIONS_DEG = (-5.0, 0.0, 26.999999, 27.0, 27.000001, 45.0, 64.0, 75.9464, 80.0)

CIE_PARAMETERS = {
    1: (4.0, -0.7, 0.0, -1.0, 0.0),
    2: (4.0, -0.7, 2.0, -1.5, 0.15),
    3: (1.1, -0.8, 0.0, -1.0, 0.0),
    4: (1.1, -0.8, 2.0, -1.5, 0.15),
    5: (0.0, -1.0, 0.0, -1.0, 0.0),
    6: (0.0, -1.0, 2.0, -1.5, 0.15),
    7: (0.0, -1.0, 5.0, -2.5, 0.30),
    8: (0.0, -1.0, 10.0, -3.0, 0.45),
    9: (-1.0, -0.55, 2.0, -1.5, 0.15),
    10: (-1.0, -0.55, 5.0, -2.5, 0.30),
    11: (-1.0, -0.55, 10.0, -3.0, 0.45),
    12: (-1.0, -0.32, 10.0, -3.0, 0.45),
    13: (-1.0, -0.32, 16.0, -3.0, 0.30),
    14: (-1.0, -0.15, 16.0, -3.0, 0.30),
    15: (-1.0, -0.15, 24.0, -2.8, 0.15),
}


def axial_residual(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the signed axial-angle difference on [-pi/2, pi/2)."""
    return 0.5 * np.arctan2(np.sin(2 * (first - second)), np.cos(2 * (first - second)))


def direction(altitude: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Return unit directions in the package's North/East/Up basis."""
    return np.stack(
        (
            np.cos(altitude) * np.cos(azimuth),
            np.cos(altitude) * np.sin(azimuth),
            np.sin(altitude),
        ),
        axis=-1,
    )


def stereographic_from_vector(vector: np.ndarray) -> np.ndarray:
    """Project unit vectors from the sphere to the zenith-centred plane."""
    vector = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
    altitude = np.arcsin(np.clip(vector[..., 2], -1.0, 1.0))
    azimuth = np.arctan2(vector[..., 1], vector[..., 0])
    return np.tan((np.pi / 2 - altitude) / 2) * np.exp(1j * azimuth)


def rayleigh_oracle(
    observed_altitude: np.ndarray,
    observed_azimuth: np.ndarray,
    sun_altitude: float,
    sun_azimuth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Rayleigh geometry and projected E-vector orientation independently."""
    point = direction(observed_altitude, observed_azimuth)
    sun = direction(np.asarray(sun_altitude), np.asarray(sun_azimuth))
    cosine = np.clip(np.sum(point * sun, axis=-1), -1.0, 1.0)
    scattering = np.arccos(cosine)
    dop = np.sin(scattering) ** 2 / (1 + np.cos(scattering) ** 2)

    # The electric vector is tangent to the sphere and normal to the scattering
    # plane.  Move an infinitesimal great-circle step in that direction, then
    # read its orientation after stereographic projection.  This avoids using
    # the production model's trigonometric AOP expression.
    electric = np.cross(sun, point)
    electric_norm = np.linalg.norm(electric, axis=-1, keepdims=True)
    electric = np.divide(
        electric, electric_norm, out=np.zeros_like(electric), where=electric_norm > 1e-14
    )
    epsilon = 1e-7
    displaced = point * np.cos(epsilon) + electric * np.sin(epsilon)
    aop = np.angle(stereographic_from_vector(displaced) - stereographic_from_vector(point))
    return scattering, dop, aop


def depolarized_rayleigh_dop(
    scattering: np.ndarray, depolarization_ratio: float = 0.0279
) -> np.ndarray:
    """Evaluate Wu et al. (2014) Eqs. (3)-(5) independently."""
    king_factor = (1.0 - depolarization_ratio) / (1.0 + 0.5 * depolarization_ratio)
    anisotropy = 4.0 * (1.0 - king_factor) / (3.0 * king_factor)
    return np.sin(scattering) ** 2 / (1.0 + np.cos(scattering) ** 2 + anisotropy)


def project(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    return np.tan(zenith / 2) * np.exp(1j * azimuth)


def quartic_field(
    observed_zenith: np.ndarray,
    observed_azimuth: np.ndarray,
    sun_zenith: float,
    sun_azimuth: float,
    below_offset: float,
    above_offset: float,
) -> tuple[np.ndarray, tuple[complex, complex, complex, complex]]:
    """Evaluate Berry Eq. (4.2) using roots placed directly by sky angles."""
    observed = project(observed_zenith, observed_azimuth)
    below = project(np.asarray(sun_zenith + below_offset), np.asarray(sun_azimuth))
    above = project(np.asarray(sun_zenith - above_offset), np.asarray(sun_azimuth))
    above_anti = -1 / np.conjugate(below)
    below_anti = -1 / np.conjugate(above)
    numerator = (
        -4
        * (observed - below)
        * (observed - above)
        * (observed - above_anti)
        * (observed - below_anti)
    )
    denominator = (
        (1 + np.abs(observed) ** 2) ** 2 * np.abs(below - above_anti) * np.abs(above - below_anti)
    )
    return numerator / denominator, (below, above, above_anti, below_anti)


def asymmetric_shape(
    observed_zenith: np.ndarray,
    observed_azimuth: np.ndarray,
    signed_offsets: tuple[float, float, float, float],
    sun_zenith: float,
    sun_azimuth: float,
) -> np.ndarray:
    """Evaluate the independent four-root half-angle modulus."""
    azimuth_difference = np.cos(observed_azimuth - sun_azimuth)
    shape = np.ones_like(observed_zenith, dtype=np.float64)
    for signed_offset in signed_offsets:
        meridian_angle = sun_zenith + signed_offset
        cosine_separation = (
            np.cos(observed_zenith) * np.cos(meridian_angle)
            + np.sin(observed_zenith) * np.sin(meridian_angle) * azimuth_difference
        )
        shape *= np.sqrt(np.clip((1 - cosine_separation) / 2, 0.0, None))
    return shape


def asymmetric_peak_scale(
    signed_offsets: tuple[float, float, float, float], sun_azimuth: float
) -> float:
    """Independently reproduce the deterministic full-sphere peak scan."""
    zenith = np.linspace(1e-6, np.pi - 1e-6, 257)
    azimuth = np.linspace(0.0, 2 * np.pi, 513)[:-1]
    best = 0.0
    for refinement in range(4):
        grid_zenith, grid_azimuth = np.meshgrid(zenith, azimuth, indexing="ij")
        values = asymmetric_shape(
            grid_zenith,
            grid_azimuth,
            signed_offsets,
            0.0,
            sun_azimuth,
        )
        row, column = np.unravel_index(int(values.argmax()), values.shape)
        best = float(values[row, column])
        if refinement == 3:
            break
        zenith_step = (zenith[-1] - zenith[0]) / (zenith.size - 1)
        azimuth_step = (azimuth[-1] - azimuth[0]) / (azimuth.size - 1)
        zenith = np.linspace(
            max(1e-9, zenith[row] - 2 * zenith_step),
            min(np.pi - 1e-9, zenith[row] + 2 * zenith_step),
            65,
        )
        azimuth = np.linspace(
            azimuth[column] - 2 * azimuth_step,
            azimuth[column] + 2 * azimuth_step,
            65,
        )
    return 1.0 / best


def asymmetric_field(
    observed_zenith: np.ndarray,
    observed_azimuth: np.ndarray,
    sun_zenith: float,
    sun_azimuth: float,
    below_offset: float,
    above_offset: float,
) -> tuple[np.ndarray, tuple[complex, complex, complex, complex]]:
    """Evaluate the default asymmetric field without production helpers."""
    signed_offsets = (
        below_offset,
        -above_offset,
        -np.pi + below_offset,
        -np.pi - below_offset,
    )
    meridian_angles = tuple(sun_zenith + offset for offset in signed_offsets)
    roots = tuple(
        complex(np.tan(angle / 2) * np.exp(1j * sun_azimuth)) for angle in meridian_angles
    )
    observed = project(observed_zenith, observed_azimuth)
    analytic = (
        -4
        * (observed - roots[0])
        * (observed - roots[1])
        * (observed - roots[2])
        * (observed - roots[3])
        / (1 + np.abs(observed) ** 2) ** 2
    )
    modulus = asymmetric_shape(
        observed_zenith,
        observed_azimuth,
        signed_offsets,
        sun_zenith,
        sun_azimuth,
    ) * asymmetric_peak_scale(signed_offsets, sun_azimuth)
    return modulus * np.exp(1j * np.angle(analytic)), roots


def cie_oracle(
    zenith: np.ndarray, sun_zenith: float, scattering: np.ndarray, sky_type: int
) -> np.ndarray:
    a, b, c, d, e = CIE_PARAMETERS[sky_type]
    gradation = (1 + a * np.exp(b / np.cos(zenith))) / (1 + a * np.exp(b))
    indicatrix = (
        1 + c * (np.exp(d * scattering) - np.exp(d * np.pi / 2)) + e * np.cos(scattering) ** 2
    )
    zenith_indicatrix = (
        1 + c * (np.exp(d * sun_zenith) - np.exp(d * np.pi / 2)) + e * np.cos(sun_zenith) ** 2
    )
    return gradation * indicatrix / zenith_indicatrix


def offsets(model_name: str, sun_elevation_deg: float) -> tuple[float, float]:
    if model_name == "berry":
        return np.deg2rad(15.0), np.deg2rad(15.0)
    brewster = (
        37.34 + 0.49 * sun_elevation_deg
        if sun_elevation_deg <= 27
        else 56.84 - 0.25 * sun_elevation_deg
    )
    babinet = 42.53 - 0.56 * sun_elevation_deg
    return np.deg2rad(brewster), np.deg2rad(babinet)


def simulate(
    model_class,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    azimuths: np.ndarray,
    altitudes: np.ndarray,
    cie_type: int = 4,
):
    sun = SkyCoord(
        az=[sun_azimuth_deg] * deg,
        alt=[sun_elevation_deg] * deg,
        frame=AltAz(obstime=TIMES, location=LOCATION),
    )
    kwargs = {"out_of_range": "ignore"} if model_class in (AsymmetricQuartic, Pan, QuEEN) else {}
    model = model_class(
        times=TIMES,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PanFidelityWarning)
        warnings.simplefilter("ignore", NeutralPointRangeWarning)
        return model, model.simulate_sky(cie_sky_type=cie_type, sun_position=sun)


def main() -> None:
    rows, columns = 17, 29
    azimuths_deg = np.tile(np.linspace(-179.0, 179.0, columns), (rows, 1))
    altitudes_deg = np.tile(np.linspace(1.0, 89.0, rows)[:, None], (1, columns))
    azimuths = np.deg2rad(azimuths_deg)
    altitudes = np.deg2rad(altitudes_deg)
    zenith = np.pi / 2 - altitudes

    metrics: dict[str, float | int | str] = {
        "oracle_cases": 0,
        "oracle_rtol": 1e-6,
        "oracle_atol": 1e-8,
        "aop_tolerance_rad": 1e-6,
        "neutral_point_tolerance_deg": 1e-6,
    }
    maxima = {
        "rayleigh_scattering_abs_error": 0.0,
        "rayleigh_dop_abs_error": 0.0,
        "rayleigh_aop_axial_error_rad": 0.0,
        "depolarized_rayleigh_dop_abs_error": 0.0,
        "depolarized_rayleigh_aop_axial_error_rad": 0.0,
        "berry_published_modulus_abs_error": 0.0,
        "berry_vs_legacy_policy_abs_difference": 0.0,
        "berry_aop_axial_error_rad": 0.0,
        "pan_dop_abs_error": 0.0,
        "pan_aop_axial_error_rad": 0.0,
        "queen_dop_abs_error": 0.0,
        "queen_aop_axial_error_rad": 0.0,
        "asymmetric_dop_abs_error": 0.0,
        "asymmetric_aop_axial_error_rad": 0.0,
        "cie_radiance_abs_error": 0.0,
        "neutral_metadata_error_deg": 0.0,
    }
    sky_dop_min = np.inf
    sky_dop_max = -np.inf
    minimum_defined_radiance = np.inf
    nonfinite_defined_radiance = 0

    classes = {"berry": Berry, "pan": Pan, "queen": QuEEN}
    for sun_elevation_deg in SUN_ELEVATIONS_DEG:
        for sun_azimuth_deg in SUN_AZIMUTHS_DEG:
            sun_altitude = np.deg2rad(sun_elevation_deg)
            sun_azimuth = np.deg2rad(sun_azimuth_deg)
            sun_zenith = np.pi / 2 - sun_altitude

            _, rayleigh = simulate(
                Rayleigh, sun_elevation_deg, sun_azimuth_deg, azimuths_deg, altitudes_deg
            )
            expected_scattering, expected_dop, expected_aop = rayleigh_oracle(
                altitudes, azimuths, sun_altitude, sun_azimuth
            )
            maxima["rayleigh_scattering_abs_error"] = max(
                maxima["rayleigh_scattering_abs_error"],
                float(np.max(np.abs(rayleigh[3][0] - expected_scattering))),
            )
            maxima["rayleigh_dop_abs_error"] = max(
                maxima["rayleigh_dop_abs_error"],
                float(np.max(np.abs(rayleigh[0][0] - expected_dop))),
            )
            sky_dop_min = min(sky_dop_min, float(np.min(rayleigh[0][0])))
            sky_dop_max = max(sky_dop_max, float(np.max(rayleigh[0][0])))
            rayleigh_valid = expected_dop > 1e-8
            maxima["rayleigh_aop_axial_error_rad"] = max(
                maxima["rayleigh_aop_axial_error_rad"],
                float(
                    np.max(
                        np.abs(
                            axial_residual(
                                rayleigh[1][0][rayleigh_valid], expected_aop[rayleigh_valid]
                            )
                        )
                    )
                ),
            )

            _, depolarized = simulate(
                DepolarizedRayleigh,
                sun_elevation_deg,
                sun_azimuth_deg,
                azimuths_deg,
                altitudes_deg,
            )
            expected_depolarized_dop = depolarized_rayleigh_dop(expected_scattering)
            maxima["depolarized_rayleigh_dop_abs_error"] = max(
                maxima["depolarized_rayleigh_dop_abs_error"],
                float(np.max(np.abs(depolarized[0][0] - expected_depolarized_dop))),
            )
            depolarized_valid = expected_depolarized_dop > 1e-8
            maxima["depolarized_rayleigh_aop_axial_error_rad"] = max(
                maxima["depolarized_rayleigh_aop_axial_error_rad"],
                float(
                    np.max(
                        np.abs(
                            axial_residual(
                                depolarized[1][0][depolarized_valid],
                                expected_aop[depolarized_valid],
                            )
                        )
                    )
                ),
            )
            sky_dop_min = min(sky_dop_min, float(np.min(depolarized[0][0])))
            sky_dop_max = max(sky_dop_max, float(np.max(depolarized[0][0])))

            for model_name, model_class in classes.items():
                _, actual = simulate(
                    model_class, sun_elevation_deg, sun_azimuth_deg, azimuths_deg, altitudes_deg
                )
                below_offset, above_offset = offsets(model_name, sun_elevation_deg)
                field, roots = quartic_field(
                    zenith, azimuths, sun_zenith, sun_azimuth, below_offset, above_offset
                )
                modulus = np.abs(field)
                frame_aop = 0.5 * np.angle(field * np.exp(-2j * sun_azimuth))
                if model_name == "berry":
                    expected_model_dop = modulus
                    legacy_policy_dop = modulus / (2 - modulus)
                    expected_model_aop = frame_aop
                    maxima["berry_published_modulus_abs_error"] = max(
                        maxima["berry_published_modulus_abs_error"],
                        float(np.max(np.abs(actual[0][0] - expected_model_dop))),
                    )
                    maxima["berry_vs_legacy_policy_abs_difference"] = max(
                        maxima["berry_vs_legacy_policy_abs_difference"],
                        float(np.max(np.abs(actual[0][0] - legacy_policy_dop))),
                    )
                    aop_key = "berry_aop_axial_error_rad"
                elif model_name == "pan":
                    expected_model_dop = modulus
                    expected_model_aop = (frame_aop - azimuths + np.pi / 2) % np.pi - np.pi / 2
                    maxima["pan_dop_abs_error"] = max(
                        maxima["pan_dop_abs_error"],
                        float(np.max(np.abs(actual[0][0] - expected_model_dop))),
                    )
                    aop_key = "pan_aop_axial_error_rad"
                else:
                    expected_model_dop = modulus / (2 - modulus)
                    expected_model_aop = frame_aop
                    maxima["queen_dop_abs_error"] = max(
                        maxima["queen_dop_abs_error"],
                        float(np.max(np.abs(actual[0][0] - expected_model_dop))),
                    )
                    aop_key = "queen_aop_axial_error_rad"

                valid = modulus > 1e-8
                sky_dop_min = min(sky_dop_min, float(np.min(actual[0][0])))
                sky_dop_max = max(sky_dop_max, float(np.max(actual[0][0])))
                maxima[aop_key] = max(
                    maxima[aop_key],
                    float(
                        np.max(
                            np.abs(axial_residual(actual[1][0][valid], expected_model_aop[valid]))
                        )
                    ),
                )

                # Metadata entries correspond to above sun, below sun, above anti,
                # below anti. Compare those directions with independently built roots.
                ordered_roots = (roots[1], roots[0], roots[2], roots[3])
                for root, az_index, alt_index in zip(
                    ordered_roots, (6, 8, 12, 14), (7, 9, 13, 15), strict=False
                ):
                    expected_root_zenith = 2 * np.arctan(np.abs(root))
                    expected_root_altitude = np.pi / 2 - expected_root_zenith
                    expected_root_azimuth = np.angle(root) % (2 * np.pi)
                    actual_azimuth = float(np.asarray(actual[az_index]).ravel()[0])
                    actual_altitude = float(np.asarray(actual[alt_index]).ravel()[0])
                    separation_cosine = np.sin(expected_root_altitude) * np.sin(
                        actual_altitude
                    ) + np.cos(expected_root_altitude) * np.cos(actual_altitude) * np.cos(
                        expected_root_azimuth - actual_azimuth
                    )
                    separation_deg = np.rad2deg(np.arccos(np.clip(separation_cosine, -1.0, 1.0)))
                    maxima["neutral_metadata_error_deg"] = max(
                        maxima["neutral_metadata_error_deg"], float(separation_deg)
                    )

                metrics["oracle_cases"] = int(metrics["oracle_cases"]) + 1

            _, actual_asymmetric = simulate(
                AsymmetricQuartic,
                sun_elevation_deg,
                sun_azimuth_deg,
                azimuths_deg,
                altitudes_deg,
            )
            below_offset, above_offset = offsets("queen", sun_elevation_deg)
            expected_field, asymmetric_roots = asymmetric_field(
                zenith,
                azimuths,
                sun_zenith,
                sun_azimuth,
                below_offset,
                above_offset,
            )
            expected_modulus = np.abs(expected_field)
            expected_dop = expected_modulus / (2 - expected_modulus)
            expected_aop = 0.5 * np.angle(expected_field * np.exp(-2j * sun_azimuth))
            maxima["asymmetric_dop_abs_error"] = max(
                maxima["asymmetric_dop_abs_error"],
                float(np.max(np.abs(actual_asymmetric[0][0] - expected_dop))),
            )
            asymmetric_valid = expected_modulus > 1e-8
            maxima["asymmetric_aop_axial_error_rad"] = max(
                maxima["asymmetric_aop_axial_error_rad"],
                float(
                    np.max(
                        np.abs(
                            axial_residual(
                                actual_asymmetric[1][0][asymmetric_valid],
                                expected_aop[asymmetric_valid],
                            )
                        )
                    )
                ),
            )
            sky_dop_min = min(sky_dop_min, float(np.min(actual_asymmetric[0][0])))
            sky_dop_max = max(sky_dop_max, float(np.max(actual_asymmetric[0][0])))

            ordered_asymmetric_roots = (
                asymmetric_roots[1],
                asymmetric_roots[0],
                asymmetric_roots[2],
                asymmetric_roots[3],
            )
            for root, az_index, alt_index in zip(
                ordered_asymmetric_roots, (6, 8, 12, 14), (7, 9, 13, 15), strict=False
            ):
                expected_root_zenith = 2 * np.arctan(np.abs(root))
                expected_root_altitude = np.pi / 2 - expected_root_zenith
                expected_root_azimuth = np.angle(root) % (2 * np.pi)
                actual_azimuth = float(np.asarray(actual_asymmetric[az_index]).ravel()[0])
                actual_altitude = float(np.asarray(actual_asymmetric[alt_index]).ravel()[0])
                separation_cosine = np.sin(expected_root_altitude) * np.sin(
                    actual_altitude
                ) + np.cos(expected_root_altitude) * np.cos(actual_altitude) * np.cos(
                    expected_root_azimuth - actual_azimuth
                )
                separation_deg = np.rad2deg(np.arccos(np.clip(separation_cosine, -1.0, 1.0)))
                maxima["neutral_metadata_error_deg"] = max(
                    maxima["neutral_metadata_error_deg"], float(separation_deg)
                )
            metrics["oracle_cases"] = int(metrics["oracle_cases"]) + 1

    # CIE validation for all 15 types, including the published worked example's
    # broad geometry rather than relying only on zenith normalization.
    sun_elevation_deg, sun_azimuth_deg = 38.02, 147.67
    _, reference = simulate(
        Rayleigh, sun_elevation_deg, sun_azimuth_deg, azimuths_deg, altitudes_deg, cie_type=1
    )
    expected_scattering = np.asarray(reference[3][0])
    sun_zenith = np.deg2rad(90.0 - sun_elevation_deg)
    for sky_type in CIE_PARAMETERS:
        _, actual = simulate(
            Rayleigh,
            sun_elevation_deg,
            sun_azimuth_deg,
            azimuths_deg,
            altitudes_deg,
            cie_type=sky_type,
        )
        expected = cie_oracle(zenith, sun_zenith, expected_scattering, sky_type)
        maxima["cie_radiance_abs_error"] = max(
            maxima["cie_radiance_abs_error"], float(np.max(np.abs(actual[2][0] - expected)))
        )
        minimum_defined_radiance = min(minimum_defined_radiance, float(np.min(actual[2][0])))
        nonfinite_defined_radiance += int(np.size(actual[2][0]) - np.isfinite(actual[2][0]).sum())

    metrics.update(maxima)
    metrics["sky_dop_min"] = sky_dop_min
    metrics["sky_dop_max"] = sky_dop_max
    metrics["minimum_defined_cie_radiance"] = minimum_defined_radiance
    metrics["nonfinite_defined_cie_radiance"] = nonfinite_defined_radiance
    metrics["scientific_oracle_status"] = (
        "PASS"
        if (
            maxima["rayleigh_scattering_abs_error"] <= 1e-8
            and maxima["rayleigh_dop_abs_error"] <= 1e-6
            and maxima["rayleigh_aop_axial_error_rad"] <= 1e-6
            and maxima["depolarized_rayleigh_dop_abs_error"] <= 1e-8
            and maxima["depolarized_rayleigh_aop_axial_error_rad"] <= 1e-6
            and maxima["berry_published_modulus_abs_error"] <= 1e-8
            and maxima["berry_aop_axial_error_rad"] <= 1e-6
            and maxima["pan_dop_abs_error"] <= 1e-8
            and maxima["pan_aop_axial_error_rad"] <= 1e-6
            and maxima["queen_dop_abs_error"] <= 1e-8
            and maxima["queen_aop_axial_error_rad"] <= 1e-6
            and maxima["asymmetric_dop_abs_error"] <= 1e-8
            and maxima["asymmetric_aop_axial_error_rad"] <= 1e-6
            and maxima["cie_radiance_abs_error"] <= 1e-8
            and maxima["neutral_metadata_error_deg"] <= 1e-6
            and sky_dop_min >= 0.0
            and sky_dop_max <= 1.0 + 1e-9
            and minimum_defined_radiance > 0.0
            and nonfinite_defined_radiance == 0
        )
        else "FAIL"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
