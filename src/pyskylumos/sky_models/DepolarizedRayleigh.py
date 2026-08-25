"""Rayleigh polarization for anisotropic molecules, after Wu et al. (2014).

Ideal Rayleigh theory treats air molecules as isotropic, which makes skylight
perfectly polarized at 90 degrees from the sun. Real molecules are anisotropic:
their induced dipole is not exactly parallel to the incident field, so a small
depolarized component survives even at the polarization maximum. The effect is
described by a single scalar, the molecular depolarization ratio ``delta``.

The polarization law is Wu et al. (2014), Eqs. (3)-(5). Writing the King factor
as ``Delta = (1 - delta) / (1 + delta / 2)``, the phase matrix of an anisotropic
molecule gives, for unpolarized incident sunlight::

    DoP(gamma) = sin^2(gamma) / [1 + cos^2(gamma) + 4 (1 - Delta) / (3 Delta)]

At ``delta = 0`` this reduces algebraically to the ideal Rayleigh law of
:class:`~pyskylumos.sky_models.Rayleigh.Rayleigh`, which is asserted by the test
suite element for element.

Only the degree of polarization changes. The angle of polarization, the CIE
radiance, the scattering angle and the neutral-point structure are inherited
unchanged, so this model still places its two neutral points exactly at the sun
and the anti-sun and still reports the fixed-world-chart AOP.

References:
    Wu L-H, Zhang J, Fan Z-G and Gao J 2014 *An analytical model for skylight
    polarization pattern with multiple scattering* Acta Phys. Sin. **63** 114201
    -- Eqs. (3)-(5). The paper's later second-scattering approximation requires
    an aerosol Mueller term and constants without a general published mapping,
    and is deliberately not implemented.

    Bodhaine B A, Wood N B, Dutton E G and Slusser J R 1999 *On Rayleigh Optical
    Depth Calculations* J. Atmos. Ocean. Technol. **16** 1854 -- provenance for
    the conventional dry-air depolarization ratio used as the default.
"""

from typing import ClassVar, Final

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from numpy import cos, sin

from pyskylumos._types import FloatArray, RealArray
from pyskylumos._validation import require_real
from pyskylumos.sky_models.Rayleigh import Rayleigh

#: Conventional dry-air molecular depolarization ratio.
DEFAULT_DEPOLARIZATION_RATIO: Final[float] = 0.0279

#: Upper bound of the range over which Wu et al. validate the phase matrix.
MAXIMUM_DEPOLARIZATION_RATIO: Final[float] = 0.5


class DepolarizedRayleigh(Rayleigh):
    """Simulate Rayleigh polarization with molecular anisotropy included."""

    MODEL_FAMILY: ClassVar[str] = "single_scattering_molecular"
    HAS_SPLIT_NEUTRAL_POINTS: ClassVar[bool] = False
    SUPPORTS_WAVELENGTH: ClassVar[bool] = False
    SUPPORTS_TURBIDITY: ClassVar[bool] = False
    SUPPORTS_GROUND_ALBEDO: ClassVar[bool] = False
    REFERENCE_DOI: ClassVar[str] = "10.7498/aps.63.114201"

    def __init__(
        self,
        times: Time,
        observation_location: EarthLocation,
        altitudes: RealArray,
        azimuths: RealArray,
        depolarization_ratio: float = DEFAULT_DEPOLARIZATION_RATIO,
    ) -> None:
        """Initialize the simulator with observation geometry and molecular anisotropy.

        The positional order places ``altitudes`` before ``azimuths`` to match
        :class:`~pyskylumos.sky_models.Rayleigh.Rayleigh`, which inverts the order
        used by :class:`~pyskylumos.sky_models.SkySimulator.SkySimulator`. Engine
        constructs every model by keyword, so the inversion is invisible there.

        Args:
            times: Observation times for each simulation step.
            observation_location: Location of the observer on Earth.
            altitudes: Grid of world altitudes in degrees for sky sampling.
            azimuths: Grid of world azimuths in degrees for sky sampling.
            depolarization_ratio: Molecular depolarization ratio on ``[0, 0.5]``.
                Zero recovers ideal Rayleigh scattering exactly.

        Raises:
            ConfigurationError: If the depolarization ratio is not a finite real
                number on ``[0, 0.5]``.
        """
        super().__init__(
            times=times,
            observation_location=observation_location,
            altitudes=altitudes,
            azimuths=azimuths,
        )

        ratio: float = require_real(
            "depolarization_ratio",
            depolarization_ratio,
            minimum=0.0,
            maximum=MAXIMUM_DEPOLARIZATION_RATIO,
        )
        king_factor: float = (1.0 - ratio) / (1.0 + 0.5 * ratio)

        self.__depolarization_ratio: float = ratio
        self.__anisotropy: float = 4.0 * (1.0 - king_factor) / (3.0 * king_factor)

    @property
    def depolarization_ratio(self) -> float:
        """Return the configured molecular depolarization ratio.

        Returns:
            The depolarization ratio this simulator was constructed with.
        """
        return self.__depolarization_ratio

    def _dop_from_scattering_angle(self, scattering_angle: FloatArray) -> FloatArray:
        """Compute the degree of polarization for anisotropic molecules.

        Implements Wu et al. (2014) Eqs. (3)-(5). The anisotropy term is a
        non-negative constant, so the result is bounded on ``[0, 1]`` by
        construction and needs no clipping; a clip here would conceal a
        regression rather than prevent one.

        Args:
            scattering_angle: Scattering angle between sun and observation point,
                in radians.

        Returns:
            Degree of polarization for each sampled point.
        """
        dop = sin(scattering_angle) ** 2 / (1 + cos(scattering_angle) ** 2 + self.__anisotropy)
        return np.asarray(dop, dtype=np.float64)
