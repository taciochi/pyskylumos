"""Covariant quartic sky model with independently placed anti-solar roots.

The solar Brewster and Babinet distances use Pan et al.'s measured solar-elevation
fits. The Arago and fourth-point distances are configured independently along the
same signed solar meridian. This is PySkyLumos's covariant explicit-root
realization of the asymmetric extension discussed in OpenSky, retaining Berry's
quartic phase and OpenSky's intensity-to-DoLP conversion.

No published solar-elevation regressions exist for the Arago or fourth-point
distances. Their defaults are literature-motivated modelling choices, not fitted
laws. In particular, the default fourth-point distance follows only Horvath and
Varju's qualitative observation that it is about as far below the anti-sun as the
Brewster point is below the sun.

The default peak normalization is numerical rather than closed form: it performs a
deterministic full-sphere scan followed by three local refinements. Analytic and
literature-anchored validation establishes the construction's internal properties.
The repository-local single-capture comparison is exploratory and does not
establish that this model fits real skies better than QuEEN in general.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy import absolute, angle, complex128, exp, pi

from pyskylumos._types import ComplexArray, FloatArray, RealArray
from pyskylumos._validation import require_real
from pyskylumos.exceptions import ConfigurationError
from pyskylumos.sky_models.NeutralPointOffsets import RANGE_POLICIES, pan_offsets
from pyskylumos.sky_models.QuarticSkyModel import QuarticSkyModel
from pyskylumos.sky_models.StereographicQuartic import (
    half_angle_modulus,
    omega_from_roots,
    place_on_solar_meridian,
)

OffsetSelector = Literal["brewster", "babinet"] | float
Normalisation = Literal["peak", "berry"]
RootTuple = tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]
OffsetTuple = tuple[FloatArray, FloatArray, FloatArray, FloatArray]

_TWO_PI: float = 2 * pi
_SCAN_ZENITH: FloatArray = np.linspace(1e-6, pi - 1e-6, 257)
_SCAN_AZIMUTH: FloatArray = np.linspace(0.0, _TWO_PI, 513)[:-1]


def _half_angle_shape(
    observed_zenith: FloatArray,
    observed_azimuth: FloatArray,
    meridian_angles: tuple[float, float, float, float],
    sun_azimuth: float,
) -> FloatArray:
    """Evaluate the half-angle shape for the peak-normalization scan."""
    delta_azimuth: FloatArray = np.cos(observed_azimuth - sun_azimuth)
    shape: FloatArray = np.ones_like(np.asarray(observed_zenith, dtype=np.float64))
    for meridian_angle in meridian_angles:
        cos_gamma: FloatArray = (
            np.cos(observed_zenith) * np.cos(meridian_angle)
            + np.sin(observed_zenith) * np.sin(meridian_angle) * delta_azimuth
        )
        shape = shape * np.sqrt(np.clip((1 - cos_gamma) / 2, 0.0, None))
    return shape


def _berry_scale(roots: RootTuple) -> FloatArray:
    """Return Berry's closed-form scale generalized to four explicit roots."""
    chord: FloatArray = absolute(roots[0] - roots[2]) * absolute(roots[1] - roots[3])
    conformal: FloatArray = np.ones_like(chord)
    for root in roots:
        conformal = conformal * np.sqrt(1 + absolute(root) ** 2)
    return 4.0 * conformal / chord


def _peak_scale(
    meridian_angles: OffsetTuple, sun_azimuth: FloatArray, refine: int = 3
) -> FloatArray:
    """Return the inverse full-sphere peak from a deterministic refined scan."""
    angles_flat = [np.reshape(np.asarray(angle, dtype=np.float64), -1) for angle in meridian_angles]
    azimuth_flat = np.reshape(np.asarray(sun_azimuth, dtype=np.float64), -1)
    peaks = np.empty(angles_flat[0].shape, dtype=np.float64)

    for index in range(angles_flat[0].size):
        angles_i = (
            float(angles_flat[0][index]),
            float(angles_flat[1][index]),
            float(angles_flat[2][index]),
            float(angles_flat[3][index]),
        )
        azimuth_i = float(azimuth_flat[index % azimuth_flat.size])
        zenith, azimuth = _SCAN_ZENITH, _SCAN_AZIMUTH
        best = 0.0
        for step in range(refine + 1):
            grid_zenith, grid_azimuth = np.meshgrid(zenith, azimuth, indexing="ij")
            values = _half_angle_shape(grid_zenith, grid_azimuth, angles_i, azimuth_i)
            row, column = np.unravel_index(int(values.argmax()), values.shape)
            best = float(values[row, column])
            if step == refine:
                break
            zenith_step = (zenith[-1] - zenith[0]) / (zenith.size - 1)
            azimuth_step = (azimuth[-1] - azimuth[0]) / (azimuth.size - 1)
            zenith = np.linspace(
                max(1e-9, zenith[row] - 2 * zenith_step),
                min(pi - 1e-9, zenith[row] + 2 * zenith_step),
                65,
            )
            azimuth = np.linspace(
                azimuth[column] - 2 * azimuth_step,
                azimuth[column] + 2 * azimuth_step,
                65,
            )
        peaks[index] = best

    return np.reshape(1.0 / peaks, np.shape(meridian_angles[0]))


class AsymmetricQuartic(QuarticSkyModel):
    """Simulate a covariant quartic field with four meridian-positioned roots."""

    def __init__(
        self,
        times: Time,
        observation_location: EarthLocation,
        azimuths: RealArray,
        altitudes: RealArray,
        arago_offset: Literal["brewster", "babinet"] | float = "brewster",
        fourth_offset: Literal["brewster", "babinet"] | float = "brewster",
        normalisation: Literal["peak", "berry"] = "peak",
        dop_max: float = 1.0,
        out_of_range: str = "warn",
    ) -> None:
        """Initialize the asymmetric quartic model.

        Args:
            times: Observation times for each simulation step.
            observation_location: Location of the observer on Earth.
            azimuths: Grid of world azimuths in degrees for sky sampling.
            altitudes: Grid of world altitudes in degrees for sky sampling.
            arago_offset: Arago distance as ``"brewster"``, ``"babinet"`` or
                a finite constant in degrees.
            fourth_offset: Fourth-point distance under the same selector policy.
            normalisation: ``"peak"`` for the numerical full-sphere scale or
                ``"berry"`` for the generalized closed-form Berry scale.
            dop_max: Peak DoLP scale on ``(0, 1]``.
            out_of_range: Policy for solar elevations outside Pan's measured range.

        Raises:
            ValueError: If any model option is invalid.
        """
        self.__arago_offset = self._validate_offset("arago_offset", arago_offset)
        self.__fourth_offset = self._validate_offset("fourth_offset", fourth_offset)
        if not isinstance(normalisation, str) or normalisation not in ("peak", "berry"):
            raise ConfigurationError(
                f"normalisation must be one of ('peak', 'berry'), got {normalisation!r}."
            )
        if out_of_range not in RANGE_POLICIES:
            raise ConfigurationError(
                f"out_of_range must be one of {RANGE_POLICIES}, got {out_of_range!r}."
            )
        self.__normalisation: Normalisation = normalisation
        self.__dop_max = require_real(
            "dop_max",
            dop_max,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        self.__out_of_range = out_of_range

        super().__init__(
            times=times,
            observation_location=observation_location,
            azimuths=azimuths,
            altitudes=altitudes,
        )

    @staticmethod
    def _validate_offset(name: str, value: OffsetSelector) -> OffsetSelector:
        """Validate one named anti-solar offset selector."""
        if isinstance(value, str):
            if value == "brewster" or value == "babinet":
                return value
            raise ConfigurationError(
                f"{name} must be 'brewster', 'babinet' or a finite real number of degrees; "
                f"got {value!r}."
            )
        return require_real(name, value)

    def _neutral_point_offsets(
        self, sun_altitudes_deg: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return Pan's fitted Brewster and Babinet distances in radians."""
        return pan_offsets(sun_altitudes_deg, policy=self.__out_of_range)

    def _resolve_offset(
        self,
        selector: OffsetSelector,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> FloatArray:
        """Resolve a fitted selector or constant degree value to radians."""
        if selector == "brewster":
            return below_sun_offset
        if selector == "babinet":
            return above_sun_offset
        return np.full_like(below_sun_offset, np.deg2rad(selector), dtype=np.float64)

    def _signed_offsets(
        self, below_sun_offset: FloatArray, above_sun_offset: FloatArray
    ) -> OffsetTuple:
        """Return signed offsets ordered Brewster, Babinet, Arago and fourth."""
        arago = self._resolve_offset(self.__arago_offset, below_sun_offset, above_sun_offset)
        fourth = self._resolve_offset(self.__fourth_offset, below_sun_offset, above_sun_offset)
        return (
            below_sun_offset,
            -above_sun_offset,
            -pi + arago,
            -pi - fourth,
        )

    def _roots(
        self,
        sun_zenith_angle: FloatArray,
        sun_azimuth: FloatArray,
        signed_offsets: OffsetTuple,
    ) -> RootTuple:
        """Place all four roots on the signed solar meridian."""
        return (
            place_on_solar_meridian(sun_zenith_angle, sun_azimuth, signed_offsets[0]),
            place_on_solar_meridian(sun_zenith_angle, sun_azimuth, signed_offsets[1]),
            place_on_solar_meridian(sun_zenith_angle, sun_azimuth, signed_offsets[2]),
            place_on_solar_meridian(sun_zenith_angle, sun_azimuth, signed_offsets[3]),
        )

    def _scale(
        self, roots: RootTuple, signed_offsets: OffsetTuple, sun_azimuth: FloatArray
    ) -> FloatArray:
        """Return the selected normalization scale."""
        if self.__normalisation == "berry":
            return _berry_scale(roots)
        return _peak_scale(signed_offsets, sun_azimuth)

    def _build_field(
        self,
        observed_point_projection: ComplexArray,
        observed_point_zenith_angle: FloatArray,
        observed_point_azimuth: FloatArray,
        sun_zenith_angle: FloatArray,
        sun_azimuth: FloatArray,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> tuple[ComplexArray, tuple[ComplexArray, ...]]:
        """Build the normalized asymmetric field and its four explicit roots."""
        signed_offsets = self._signed_offsets(below_sun_offset, above_sun_offset)
        roots = self._roots(sun_zenith_angle, sun_azimuth, signed_offsets)
        phase = omega_from_roots(observed_point_projection, roots)
        shape = half_angle_modulus(
            observed_point_zenith_angle,
            observed_point_azimuth,
            signed_offsets,
            sun_zenith_angle,
            sun_azimuth,
        )
        modulus = shape * self._scale(roots, signed_offsets, sun_azimuth)
        return (modulus * exp(1j * angle(phase))).astype(complex128), roots

    def _get_dop(self, field: ComplexArray) -> FloatArray:
        """Apply OpenSky's intensity-to-DoLP remap to the normalized modulus."""
        modulus: FloatArray = absolute(field)
        return self.__dop_max * modulus / (2 - modulus)

    def _get_aop(
        self,
        field: ComplexArray,
        sun_azimuth: FloatArray,
        observed_point_azimuth: FloatArray,
    ) -> FloatArray:
        """Return AOP in the fixed world stereographic chart."""
        return 0.5 * angle(field * exp(-2j * sun_azimuth))

    def _singularity_metadata(
        self,
        sun_position: SkyCoord,
        anti_sun_position: SkyCoord,
        below_sun_projection: ComplexArray,
        above_sun_projection: ComplexArray,
        below_sun_offset: FloatArray,
        above_sun_offset: FloatArray,
    ) -> list[FloatArray]:
        """Return metadata for the exact four roots used by this configuration."""
        sun_zenith_angle: FloatArray = np.asarray(
            np.pi / 2 - sun_position.alt.radian, dtype=np.float64
        )
        sun_azimuth: FloatArray = np.asarray(sun_position.az.radian, dtype=np.float64)
        signed_offsets = self._signed_offsets(below_sun_offset, above_sun_offset)
        roots = self._roots(sun_zenith_angle, sun_azimuth, signed_offsets)
        return self._metadata_from_root_tuple((roots[1], roots[0], roots[2], roots[3]))
