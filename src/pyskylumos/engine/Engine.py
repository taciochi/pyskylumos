"""Sky, camera-pose, polarization-basis, and sensor-pipeline orchestration."""

from collections.abc import Sequence
from inspect import signature
from math import pi
from numbers import Real
from typing import Any, ClassVar
from warnings import catch_warnings, simplefilter

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy import arctan2, cos, deg2rad, sin, sqrt

from pyskylumos._types import FloatArray, ParameterNames, RealArray, SensorArray
from pyskylumos._validation import (
    UNSET,
    require_argument,
    require_integer,
    require_real_array,
    require_same_shape,
    resolve_deprecated_alias,
)
from pyskylumos.exceptions import InputTypeError, InputValidationError
from pyskylumos.sensor.MicroPolarizer import MicroPolarizer
from pyskylumos.sensor.OpticalConjugator import OpticalConjugator
from pyskylumos.sensor.SensorChip import SensorChip
from pyskylumos.sensor.StokesCalculator import StokesCalculator
from pyskylumos.sky_models.AsymmetricQuartic import AsymmetricQuartic
from pyskylumos.sky_models.Berry import Berry
from pyskylumos.sky_models.DepolarizedRayleigh import DepolarizedRayleigh
from pyskylumos.sky_models.Pan import Pan, PanFidelityWarning
from pyskylumos.sky_models.QuEEN import QuEEN
from pyskylumos.sky_models.Rayleigh import Rayleigh
from pyskylumos.sky_models.SkySimulator import SkySimulator

from ._geometry import (
    directions_from_degrees,
    rotate_directions,
    sensor_tilt_rotation,
    transport_stereographic_aop_to_sensor,
)


class Engine:
    """Orchestrate sky simulation, camera pose, and sensor measurement.

    Direct sky models evaluate world-coordinate fields. This high-level class
    can additionally interpret an optical grid as sensor-local, rotate its rays
    into the world, and transport world-chart AOP into the active analyzer frame
    before passing it to the measurement pipeline.
    """

    __micro_polarizer: MicroPolarizer
    __sensor_chip: SensorChip
    __stokes_calculator: StokesCalculator
    __optical_conjugator: OpticalConjugator
    __number_pixels_vertical: int
    __number_pixels_horizontal: int

    def __init__(
        self,
        sensor_pixel_size_square_micrometers: Any = UNSET,
        lens_conjugation_type: Any = UNSET,
        number_pixels_vertical: Any = UNSET,
        number_pixels_horizontal: Any = UNSET,
        lens_focal_length_micrometers: Any = UNSET,
        tolerance: Any = UNSET,
        extinction_ratio: Any = UNSET,
        pixel_saturation_ratio: Any = UNSET,
        adc_resolution: Any = UNSET,
        signal_to_noise_ratio: Any = UNSET,
        wire_grid_orientations_slicing: Any = UNSET,
        random_seed: int | None = None,
        *,
        sensor_pixel_pitch_micrometers: Any = UNSET,
        polarizer_tolerance_radians: Any = UNSET,
        auto_exposure_saturation_fraction: Any = UNSET,
        adc_resolution_bits: Any = UNSET,
        multiplicative_noise_snr: Any = UNSET,
    ) -> None:
        """Initialize the engine with optics, sensor, and polarizer parameters.

        Args:
            sensor_pixel_size_square_micrometers: Deprecated alias for
                ``sensor_pixel_pitch_micrometers``.
            lens_conjugation_type: Lens conjugation model name.
            number_pixels_vertical: Vertical pixel count of the sensor.
            number_pixels_horizontal: Horizontal pixel count of the sensor.
            lens_focal_length_micrometers: Lens focal length in micrometers.
            tolerance: Deprecated alias for ``polarizer_tolerance_radians``.
            extinction_ratio: Polarizer extinction ratio.
            pixel_saturation_ratio: Deprecated alias for
                ``auto_exposure_saturation_fraction``.
            adc_resolution: Deprecated alias for ``adc_resolution_bits``.
            signal_to_noise_ratio: Deprecated alias for ``multiplicative_noise_snr``.
            wire_grid_orientations_slicing: Slicing pattern per polarizer orientation.
            random_seed: Optional seed shared by the micro-polarizer defects and
                multiplicative sensor noise, making a measurement reproducible.
            sensor_pixel_pitch_micrometers: Linear pixel pitch in micrometers.
            polarizer_tolerance_radians: Maximum polarizer defect in radians.
            auto_exposure_saturation_fraction: Fraction of the brightest finite
                signal mapped to ADC full scale.
            adc_resolution_bits: Unsigned ADC resolution from 1 through 24 bits.
            multiplicative_noise_snr: Positive SNR for relative Gaussian noise.

        Raises:
            ValueError: If the sensor dimensions are incompatible with the
                micro-polarizer mosaic.
        """
        pitch = resolve_deprecated_alias(
            preferred_name="sensor_pixel_pitch_micrometers",
            preferred_value=sensor_pixel_pitch_micrometers,
            legacy_name="sensor_pixel_size_square_micrometers",
            legacy_value=sensor_pixel_size_square_micrometers,
        )
        polarizer_tolerance = resolve_deprecated_alias(
            preferred_name="polarizer_tolerance_radians",
            preferred_value=polarizer_tolerance_radians,
            legacy_name="tolerance",
            legacy_value=tolerance,
        )
        saturation_fraction = resolve_deprecated_alias(
            preferred_name="auto_exposure_saturation_fraction",
            preferred_value=auto_exposure_saturation_fraction,
            legacy_name="pixel_saturation_ratio",
            legacy_value=pixel_saturation_ratio,
        )
        bits = resolve_deprecated_alias(
            preferred_name="adc_resolution_bits",
            preferred_value=adc_resolution_bits,
            legacy_name="adc_resolution",
            legacy_value=adc_resolution,
        )
        noise_snr = resolve_deprecated_alias(
            preferred_name="multiplicative_noise_snr",
            preferred_value=multiplicative_noise_snr,
            legacy_name="signal_to_noise_ratio",
            legacy_value=signal_to_noise_ratio,
        )
        lens_conjugation_type = require_argument("lens_conjugation_type", lens_conjugation_type)
        lens_focal_length_micrometers = require_argument(
            "lens_focal_length_micrometers", lens_focal_length_micrometers
        )
        extinction_ratio = require_argument("extinction_ratio", extinction_ratio)
        wire_grid_orientations_slicing = require_argument(
            "wire_grid_orientations_slicing", wire_grid_orientations_slicing
        )
        self.__number_pixels_vertical = require_integer(
            "number_pixels_vertical", number_pixels_vertical, minimum=1
        )
        self.__number_pixels_horizontal = require_integer(
            "number_pixels_horizontal", number_pixels_horizontal, minimum=1
        )

        self.__stokes_calculator = StokesCalculator(
            wire_grid_orientations_slicing=wire_grid_orientations_slicing
        )

        self.__stokes_calculator.validate_sensor_shape(
            number_pixels_vertical=self.__number_pixels_vertical,
            number_pixels_horizontal=self.__number_pixels_horizontal,
        )

        self.__optical_conjugator = OpticalConjugator(
            lens_conjugation_type=lens_conjugation_type,
            number_pixels_vertical=self.__number_pixels_vertical,
            number_pixels_horizontal=self.__number_pixels_horizontal,
            lens_focal_length_micrometers=lens_focal_length_micrometers,
            sensor_pixel_pitch_micrometers=pitch,
        )

        self.__micro_polarizer = MicroPolarizer(
            polarizer_tolerance_radians=polarizer_tolerance,
            extinction_ratio=extinction_ratio,
            wire_grid_orientations_slicing=wire_grid_orientations_slicing,
            random_seed=random_seed,
        )

        self.__sensor_chip = SensorChip(
            auto_exposure_saturation_fraction=saturation_fraction,
            adc_resolution_bits=bits,
            multiplicative_noise_snr=noise_snr,
            random_seed=random_seed,
        )

    @staticmethod
    def __cartesian_to_spherical(
        x: FloatArray, y: FloatArray, z: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Convert cartesian coordinates to azimuth/elevation angles.

        Args:
            x: X-coordinate array.
            y: Y-coordinate array.
            z: Z-coordinate array.

        Returns:
            Tuple of azimuth and altitude arrays in radians.
        """
        return (
            np.asarray(arctan2(y, x), dtype=np.float64),
            np.asarray(arctan2(z, sqrt(x**2 + y**2)), dtype=np.float64),
        )

    @staticmethod
    def __spherical_to_cartesian(
        azimuths: FloatArray, altitudes: FloatArray, r: int = 1
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Convert azimuth/elevation angles to cartesian coordinates.

        Args:
            azimuths: Azimuth angles in radians.
            altitudes: Altitude angles in radians.
            r: Radius for the conversion.

        Returns:
            Tuple of x, y, z coordinate arrays.
        """
        return (
            np.asarray(r * cos(altitudes) * cos(azimuths), dtype=np.float64),
            np.asarray(r * cos(altitudes) * sin(azimuths), dtype=np.float64),
            np.asarray(r * sin(altitudes), dtype=np.float64),
        )

    @staticmethod
    def __wrap_aop(aop: FloatArray, x: float) -> FloatArray:
        """Wrap angle of polarization to the [-pi/2, pi/2] interval.

        Args:
            aop: Angle of polarization values in radians.
            x: Rotation angle in degrees to apply before wrapping.

        Returns:
            Wrapped angle of polarization values. The input array is not modified.
        """
        if not np.isfinite(x):
            raise InputValidationError("AOP rotation angle must be finite.")
        rotated = aop + deg2rad(x)
        return np.asarray((rotated + pi / 2) % pi - pi / 2, dtype=np.float64)

    @staticmethod
    def convert_local_meridian_aop_to_sensor(
        angle_of_polarization: RealArray, observed_azimuths: RealArray
    ) -> FloatArray:
        """Convert local-meridian AOP to the fixed world stereographic frame.

        Pan's published angle is measured relative to the local meridian at
        each sampled world direction. Adding the observed-point world azimuth
        produces an angle measured from the fixed azimuth-zero chart axis;
        polarization angles are axial, so the result is wrapped onto
        ``[-pi/2, pi/2)``.

        This method does not know the camera pose and therefore does not perform
        the spatially varying world-to-sensor basis transport required by a
        tilted analyzer. It can feed :meth:`simulate_measurement` directly only
        when the sensor and world charts are aligned. For tilted cameras, use
        the integrated tilt arguments of :meth:`simulate_sky_polarization`.

        Args:
            angle_of_polarization: Local-meridian AOP values in radians.
            observed_azimuths: Observed-point azimuths in degrees.

        Returns:
            Fixed-world-chart AOP values in radians. Inputs are broadcast
            without being modified, and NaNs propagate through the conversion.
        """
        sensor_aop = angle_of_polarization + deg2rad(observed_azimuths)
        return np.asarray((sensor_aop + pi / 2) % pi - pi / 2, dtype=np.float64)

    def get_initial_azimuth_altitude(
        self,
        altitude_min_clip: float | None = None,
        custom_lens_conjugation: Any = UNSET,
        *,
        custom_lens_conjugation_type: Any = UNSET,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the optical conjugator's sensor-local direction grid.

        For the default upward-looking pose these degree-valued directions are
        also world AltAz coordinates. When integrated sensor tilt is supplied
        to :meth:`simulate_sky_polarization`, the same arrays remain local to
        the sensor and Engine rotates them into world coordinates internally.

        Args:
            altitude_min_clip: Minimum altitude to keep (degrees).
            custom_lens_conjugation: Optional custom conjugation function, used
                when the conjugator was built with the ``"custom"`` lens
                conjugation type. It is the same argument, under the same name,
                that
                :meth:`~pyskylumos.sensor.OpticalConjugator.OpticalConjugator.get_azimuth_altitude`
                accepts.
            custom_lens_conjugation_type: Deprecated alias for
                ``custom_lens_conjugation``.

        Returns:
            Fresh azimuth and altitude grids in degrees, expressed in the
            sensor-local chart.

        Raises:
            ConfigurationError: If both the current and the deprecated name are
                supplied for the same hook.
        """
        conjugation = resolve_deprecated_alias(
            preferred_name="custom_lens_conjugation",
            preferred_value=custom_lens_conjugation,
            legacy_name="custom_lens_conjugation_type",
            legacy_value=custom_lens_conjugation_type,
            default=None,
        )
        return self.__optical_conjugator.get_azimuth_altitude(
            altitude_min_clip=altitude_min_clip,
            custom_lens_conjugation=conjugation,
        )

    def tilt_sensor(
        self,
        azimuths: RealArray,
        altitudes: RealArray,
        azimuthal_tilt: float,
        tilt_angle: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Rotate sensor-local sampling directions into world coordinates.

        Sky coordinates cross this API boundary in degrees, while the two tilt
        parameters are radians. The rotation keeps its original right-handed
        convention: when ``azimuthal_tilt == 0``, North is the rotation axis;
        positive tilt moves the East horizon towards the zenith and the zenith
        towards the West horizon.

        This compatibility helper rotates directions only. It cannot transform
        an AOP field because no polarization values are supplied. Use
        :meth:`simulate_sky_polarization` with both integrated tilt arguments
        when the returned sky will enter the sensor pipeline.

        Args:
            azimuths: Input azimuth angles in degrees.
            altitudes: Input altitude angles in degrees.
            azimuthal_tilt: Horizontal rotation-axis parameter in radians. Its
                Cartesian axis is ``(cos(a), -sin(a), 0)``; zero is North and
                increasing values turn the axis toward West.
            tilt_angle: Right-handed tilt angle in radians.

        Returns:
            World azimuth and altitude arrays in degrees. Inputs are not
            modified and their combined NaN mask is preserved.

        Raises:
            TypeError: If arrays or tilt scalars have invalid types.
            ValueError: If shapes differ or values violate the finite-input
                contract.
        """
        azimuths = require_real_array("azimuths", azimuths, rank=None)
        altitudes = require_real_array("altitudes", altitudes, rank=None)
        require_same_shape(azimuths=azimuths, altitudes=altitudes)
        if np.isinf(azimuths).any() or np.isinf(altitudes).any():
            raise InputValidationError("azimuths and altitudes must not contain infinity.")
        if isinstance(azimuthal_tilt, bool) or not isinstance(azimuthal_tilt, (int, float)):
            raise InputTypeError("azimuthal_tilt must be a finite real number in radians.")
        if isinstance(tilt_angle, bool) or not isinstance(tilt_angle, (int, float)):
            raise InputTypeError("tilt_angle must be a finite real number in radians.")
        if not np.isfinite(azimuthal_tilt) or not np.isfinite(tilt_angle):
            raise InputValidationError("azimuthal_tilt and tilt_angle must be finite.")
        rotation = sensor_tilt_rotation(azimuthal_tilt, tilt_angle)
        return rotate_directions(
            np.asarray(azimuths, dtype=np.float64),
            np.asarray(altitudes, dtype=np.float64),
            rotation,
        )

    @staticmethod
    def rotate_sensor(azimuths: RealArray, rotation_angle: float | None = None) -> RealArray:
        """Add a degree-valued yaw to a sampling-azimuth array.

        This direction-only helper does not alter AOP. It is distinct from
        ``azimuth_rotation_angle`` on :meth:`simulate_sky_polarization`, which
        represents a uniform analyzer-frame offset and does not rotate rays.

        Args:
            azimuths: Input azimuth values in degrees.
            rotation_angle: Angle in degrees to apply.

        Returns:
            Rotated azimuth values in degrees. The input array is not modified.
        """
        azimuths = require_real_array("azimuths", azimuths, rank=None)
        if np.isinf(azimuths).any():
            raise InputValidationError("azimuths must not contain infinity.")
        if rotation_angle is None or rotation_angle == 0:
            return azimuths
        if isinstance(rotation_angle, bool) or not isinstance(rotation_angle, (int, float)):
            raise InputTypeError("rotation_angle must be a finite real number or None.")
        if not np.isfinite(rotation_angle):
            raise InputValidationError("rotation_angle must be finite.")

        azimuths = azimuths + rotation_angle
        azimuths = (azimuths + 180) % 360 - 180

        return np.asarray(azimuths, dtype=np.float64)

    #: Sky model names accepted by :meth:`simulate_sky_polarization`.
    #:
    #: ``PAN`` selects Pan et al. (2023) as published. Up to version 0.0.6 that
    #: name selected the model now called ``QUEEN``; see the project README migration note.
    #: ``QEN`` is an alias of ``QUEEN`` kept for the naming used in early drafts.
    __SKY_MODELS: ClassVar[dict[str, type[SkySimulator]]] = {
        "RAYLEIGH": Rayleigh,
        "DEPOLARIZED_RAYLEIGH": DepolarizedRayleigh,
        "ASYMMETRIC": AsymmetricQuartic,
        "ASQ": AsymmetricQuartic,
        "BERRY": Berry,
        "PAN": Pan,
        "QUEEN": QuEEN,
        "QEN": QuEEN,
    }

    __GEOMETRY_PARAMETERS: tuple[str, ...] = (
        "self",
        "times",
        "observation_location",
        "azimuths",
        "altitudes",
    )

    @classmethod
    def __get_sky_simulator(
        cls,
        times: Time,
        sky_model: str,
        azimuths: RealArray,
        altitudes: RealArray,
        observation_location: EarthLocation,
        model_options: dict[str, Any] | None = None,
    ) -> SkySimulator:
        """Return the sky simulator implementation for a named sky model.

        Args:
            times: Observation times for each simulation step.
            sky_model: Sky model name; one of RAYLEIGH, DEPOLARIZED_RAYLEIGH,
                ASYMMETRIC, BERRY, PAN or QUEEN. ASQ is an alias for ASYMMETRIC.
            azimuths: World azimuth grid in degrees.
            altitudes: World altitude grid in degrees.
            observation_location: Location of the observer on Earth.
            model_options: Extra keyword arguments for the chosen model.

        Returns:
            Concrete SkySimulator implementation for the requested model.

        Raises:
            ValueError: If the sky model name is not recognised.
            TypeError: If an option is not accepted by the chosen model.
        """
        if not isinstance(sky_model, str):
            raise InputTypeError(f"sky_model must be a string; got {type(sky_model).__name__}.")
        name: str = sky_model.upper()

        if name not in cls.__SKY_MODELS:
            raise InputValidationError(
                f"Sky model {name} not found. Available models: RAYLEIGH, "
                f"DEPOLARIZED_RAYLEIGH, ASYMMETRIC, BERRY, PAN, QUEEN. "
                f"Note that PAN now selects Pan et al. (2023) as published; the model shipped "
                f"as PAN up to version 0.0.6 is now QUEEN."
            )

        simulator_class: type[SkySimulator] = cls.__SKY_MODELS[name]
        options: dict[str, Any] = dict(model_options or {})

        if options:
            accepted = {
                parameter
                for parameter in signature(simulator_class.__init__).parameters
                if parameter not in cls.__GEOMETRY_PARAMETERS
            }
            unexpected = sorted(set(options) - accepted)
            if unexpected:
                raise InputTypeError(
                    f"Sky model {name} does not accept model_options {unexpected}. "
                    f"Accepted options: {sorted(accepted) if accepted else 'none'}."
                )

        return simulator_class(
            times=times,
            altitudes=altitudes,
            azimuths=azimuths,
            observation_location=observation_location,
            **options,
        )

    def simulate_sky_polarization(
        self,
        sky_model: str,
        observation_location: EarthLocation,
        times: Time,
        cie_sky_type: int,
        altitudes: RealArray,
        azimuths: RealArray,
        altitude_min_clip: float | None = None,
        azimuth_rotation_angle: float | None = None,
        accuracy: bool = False,
        sun_position: SkyCoord | None = None,
        model_options: dict[str, Any] | None = None,
        *,
        sensor_azimuthal_tilt_radians: float | None = None,
        sensor_tilt_angle_radians: float | None = None,
    ) -> tuple[Sequence[FloatArray], ParameterNames]:
        """Simulate sky polarization parameters for a chosen sky model.

        Args:
            sky_model: Sky model name; one of RAYLEIGH, DEPOLARIZED_RAYLEIGH,
                ASYMMETRIC, BERRY, PAN or QUEEN. ASQ is an alias for ASYMMETRIC.
            observation_location: Location of the observer on Earth.
            times: Observation times for each simulation step.
            cie_sky_type: CIE sky type index for radiance model.
            altitudes: Two-dimensional altitude grid in degrees. Interpreted as
                world AltAz when tilt is omitted and as sensor-local directions
                when both tilt arguments are supplied.
            azimuths: Two-dimensional azimuth grid in degrees, with the same
                conditional frame convention as ``altitudes``.
            altitude_min_clip: Minimum world altitude to keep, in degrees. With
                tilt enabled, masking is evaluated after rotating the rays.
            azimuth_rotation_angle: Uniform in-plane analyzer offset in degrees.
                It is subtracted from AOP after Pan conversion and any 3D tilt
                transport; it does not rotate the sampling grid.
            accuracy: Whether to use the optional JPL DE430 ephemeris for the sun
                position. Requires installation of ``pyskylumos[jpl]`` and a
                downloaded or pre-cached DE430 kernel.
            sun_position: Optional explicit sun position to use.
            model_options: Extra keyword arguments for the chosen sky model, such
                as ``depolarization_ratio`` for DEPOLARIZED_RAYLEIGH;
                ``arago_offset``, ``fourth_offset`` and ``normalisation`` for
                ASYMMETRIC; or ``dop_max`` and ``out_of_range`` for QUEEN.
            sensor_azimuthal_tilt_radians: Horizontal rotation-axis parameter in
                radians. The axis is ``(cos(a), -sin(a), 0)`` in North-East-Up
                coordinates. Must be supplied with ``sensor_tilt_angle_radians``.
            sensor_tilt_angle_radians: Right-handed sensor-to-world tilt in
                radians. Must be supplied with
                ``sensor_azimuthal_tilt_radians``.

        Returns:
            Tuple containing simulated sky parameters and their labels. AOP is
            expressed in the active sensor/analyzer frame for every model. Direct
            :class:`~pyskylumos.sky_models.Pan.Pan` simulations retain Pan's
            published local-meridian reference, but this Engine method converts
            Pan AOP before returning it. When sensor tilt is supplied, input
            directions are sensor-local and AOP includes the geometric basis
            transport into that tilted sensor frame.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If array geometry, sky options, or the paired finite
                tilt contract is invalid.
        """
        if not isinstance(times, Time):
            raise InputTypeError("times must be an astropy.time.Time.")
        altitudes = require_real_array("altitudes", altitudes, rank=2)
        azimuths = require_real_array("azimuths", azimuths, rank=2)
        require_same_shape(altitudes=altitudes, azimuths=azimuths)
        if np.isinf(altitudes).any() or np.isinf(azimuths).any():
            raise InputValidationError("azimuths and altitudes must not contain infinity.")
        finite_altitudes = altitudes[np.isfinite(altitudes)]
        if ((finite_altitudes < -90.0) | (finite_altitudes > 90.0)).any():
            raise InputValidationError("finite altitudes must lie in [-90, 90] degrees.")
        if isinstance(cie_sky_type, bool) or not isinstance(cie_sky_type, (int, np.integer)):
            raise InputTypeError("cie_sky_type must be an integer from 1 to 15.")
        if int(cie_sky_type) not in range(1, 16):
            raise InputValidationError(
                f"cie_sky_type must be an integer from 1 to 15; got {cie_sky_type!r}."
            )
        if altitude_min_clip is not None and not isinstance(altitude_min_clip, (int, float)):
            raise InputTypeError("altitude_min_clip must be a float or None.")
        if altitude_min_clip is not None and not np.isfinite(altitude_min_clip):
            raise InputValidationError("altitude_min_clip must be finite or None.")
        if azimuth_rotation_angle is not None and not isinstance(
            azimuth_rotation_angle, (int, float)
        ):
            raise InputTypeError("azimuth_rotation_angle must be a float or None.")
        if azimuth_rotation_angle is not None and not np.isfinite(azimuth_rotation_angle):
            raise InputValidationError("azimuth_rotation_angle must be finite or None.")
        if sun_position is not None and not isinstance(sun_position, SkyCoord):
            raise InputTypeError("sun_position must be an astropy.coordinates.SkyCoord or None.")
        if not isinstance(accuracy, bool):
            raise InputTypeError("accuracy must be a bool.")
        if model_options is not None and not isinstance(model_options, dict):
            raise InputTypeError("model_options must be a dict or None.")

        tilt_values = (sensor_azimuthal_tilt_radians, sensor_tilt_angle_radians)
        if (tilt_values[0] is None) != (tilt_values[1] is None):
            raise InputValidationError(
                "sensor_azimuthal_tilt_radians and sensor_tilt_angle_radians "
                "must be supplied together."
            )
        for tilt_name, tilt_value in zip(
            ("sensor_azimuthal_tilt_radians", "sensor_tilt_angle_radians"),
            tilt_values,
            strict=True,
        ):
            value: Any = tilt_value
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
                raise InputTypeError(f"{tilt_name} must be a finite real number or None.")
            if value is not None and not np.isfinite(float(value)):
                raise InputValidationError(f"{tilt_name} must be finite or None.")

        sensor_directions = directions_from_degrees(
            np.asarray(azimuths, dtype=np.float64),
            np.asarray(altitudes, dtype=np.float64),
        )
        sensor_to_world_rotation: FloatArray | None = None
        simulation_azimuths = azimuths
        simulation_altitudes = altitudes
        if (
            sensor_azimuthal_tilt_radians is not None
            and sensor_tilt_angle_radians is not None
            and sensor_tilt_angle_radians != 0.0
        ):
            sensor_to_world_rotation = sensor_tilt_rotation(
                sensor_azimuthal_tilt_radians,
                sensor_tilt_angle_radians,
            )
            simulation_azimuths, simulation_altitudes = rotate_directions(
                np.asarray(azimuths, dtype=np.float64),
                np.asarray(altitudes, dtype=np.float64),
                sensor_to_world_rotation,
            )

        sky_simulator: SkySimulator = self.__get_sky_simulator(
            times=times,
            azimuths=simulation_azimuths,
            sky_model=sky_model,
            altitudes=simulation_altitudes,
            observation_location=observation_location,
            model_options=model_options,
        )

        with catch_warnings():
            # Pan's fidelity notice is advisory and is documented in the README;
            # it fires once per simulator and would otherwise reach every caller.
            simplefilter("ignore", PanFidelityWarning)
            sky_polarization_parameters: list[FloatArray] = sky_simulator.simulate_sky(
                cie_sky_type=cie_sky_type,
                altitude_min_clip=altitude_min_clip,
                accuracy=accuracy,
                sun_position=sun_position,
            )

        if sky_simulator.AOP_REFERENCE == "local_meridian":
            sky_polarization_parameters[1] = self.convert_local_meridian_aop_to_sensor(
                angle_of_polarization=sky_polarization_parameters[1],
                observed_azimuths=simulation_azimuths,
            )

        if sensor_to_world_rotation is not None:
            world_directions = directions_from_degrees(
                np.asarray(simulation_azimuths, dtype=np.float64),
                np.asarray(simulation_altitudes, dtype=np.float64),
            )
            sky_polarization_parameters[1] = transport_stereographic_aop_to_sensor(
                angle_of_polarization=sky_polarization_parameters[1],
                world_directions=world_directions,
                sensor_directions=sensor_directions,
                sensor_to_world_rotation=sensor_to_world_rotation,
            )

        if azimuth_rotation_angle is not None:
            sky_polarization_parameters[1] = self.__wrap_aop(
                aop=sky_polarization_parameters[1],
                x=-1 * azimuth_rotation_angle,
            )

        return (sky_polarization_parameters, sky_simulator.parameters_simulated)

    def simulate_measurement(
        self,
        degree_of_polarization: RealArray,
        angle_of_polarization: RealArray,
        radiance: RealArray,
    ) -> dict[str, SensorArray]:
        """Simulate sensor measurements from polarization state and radiance.

        Args:
            degree_of_polarization: Input degree of polarization values in [0, 1].
            angle_of_polarization: Finite AOP values in radians, expressed in
                the same active sensor/analyzer frame as the micro-polarizer
                orientation map. Output from :meth:`simulate_sky_polarization`
                satisfies this contract. For direct ``Pan`` output on an
                untilted sensor, first use
                :meth:`convert_local_meridian_aop_to_sensor`; for a tilted
                sensor, use the integrated Engine simulation path instead.
            radiance: Non-negative input radiance values. Positive infinity denotes
                an over-range signal; NaN values in all inputs are treated as masks.

        Returns:
            Dictionary containing simulated angle of polarization and bounded
            degree of polarization in [0, 1], or NaN where undefined or masked.

        Raises:
            ValueError: If the polarization state or radiance is nonphysical.
        """
        degree_of_polarization = require_real_array(
            "degree_of_polarization", degree_of_polarization
        )
        angle_of_polarization = require_real_array("angle_of_polarization", angle_of_polarization)
        radiance = require_real_array("radiance", radiance)
        shape = require_same_shape(
            degree_of_polarization=degree_of_polarization,
            angle_of_polarization=angle_of_polarization,
            radiance=radiance,
        )
        expected_spatial = (self.__number_pixels_vertical, self.__number_pixels_horizontal)
        if shape[1:] != expected_spatial:
            raise InputValidationError(
                "Measurement spatial shape must match the configured sensor dimensions; "
                f"expected {expected_spatial}, got {shape[1:]}."
            )

        iop: RealArray = self.__micro_polarizer.get_intensity_on_pixel(
            degree_of_polarization=degree_of_polarization,
            angle_of_polarization=angle_of_polarization,
            radiance=radiance,
        )

        bits_intensity: SensorArray = self.__sensor_chip.get_bits_intensity(intensity_on_pixel=iop)

        dop, aop = self.__stokes_calculator.simulate_measurements(bits_intensity=bits_intensity)

        return {"dop": dop, "aop": aop}
