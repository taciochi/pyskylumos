from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence, Literal
import numpy as np
from numpy.typing import NDArray

from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

from pyskylumos.engine.Engine import Engine
from pyskylumos.sensor.SlicingPattern import SlicingPattern

from .frame_utils import (
    altaz_grid,
    altaz_to_unit_xyz_local,
    enu_to_ecef_unit,
    enu_to_gcrs_unit,
    rotate_aop_altaz_to_basis,
)

FrameName = Literal["altaz", "enu", "ecef", "gcrs", "camera"]

@dataclass
class EngineConfig:
    # Sensor / optics
    sensor_pixel_size_square_micrometers: float = 4.0
    lens_conjugation_type: str = "object"  # "object" | "image"
    number_pixels_vertical: int = 1080
    number_pixels_horizontal: int = 1920
    lens_focal_length_micrometers: float = 4000.0
    # Micro-polarizer
    tolerance: float = 0.5
    extinction_ratio: float = 100.0
    # Sensor chip
    pixel_saturation_ratio: float = 0.95
    adc_resolution: float = 12
    signal_to_noise_ratio: float = 40.0
    # Polarizer sampling (2x2 default)
    wire_grid_orientations_slicing: Dict[int, SlicingPattern] = None

    def __post_init__(self):
        if self.wire_grid_orientations_slicing is None:
            self.wire_grid_orientations_slicing = {
                0:   SlicingPattern(0, 0, 2),
                45:  SlicingPattern(0, 1, 2),
                90:  SlicingPattern(1, 0, 2),
                135: SlicingPattern(1, 1, 2),
            }

class PolarizationFrameGenerator:
    """
    Produce sky polarization and direction vectors in alternate frames
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine = Engine(
            sensor_pixel_size_square_micrometers=config.sensor_pixel_size_square_micrometers,
            lens_conjugation_type=config.lens_conjugation_type,
            number_pixels_vertical=config.number_pixels_vertical,
            number_pixels_horizontal=config.number_pixels_horizontal,
            lens_focal_length_micrometers=config.lens_focal_length_micrometers,
            tolerance=config.tolerance,
            extinction_ratio=config.extinction_ratio,
            pixel_saturation_ratio=config.pixel_saturation_ratio,
            adc_resolution=config.adc_resolution,
            signal_to_noise_ratio=config.signal_to_noise_ratio,
            wire_grid_orientations_slicing=config.wire_grid_orientations_slicing
        )

    def simulate_from_vectors(
        self,
        sky_model: str,
        location: EarthLocation,
        time_utc: Time,
        vectors: NDArray,
        frame: str = "enu",   # "enu" | "ecef" | "gcrs" | "camera"
        cie_sky_type: int = 12,
        altitude_min_clip_deg: float | None = None,
        camera_yaw_pitch_roll_deg: tuple[float,float,float] = (0.0,0.0,0.0),
        accuracy: bool = False,
    ):
        """
        Convert direction vectors in the given frame to Alt/Az and call the sky model.
        vectors: unit vectors with shape (...,3). Output (dop,aop,radiance) follow the same leading shape.
        """
        import numpy as _np
        from astropy.time import Time as _Time
        from .frame_utils import (
            vectors_enu_to_altaz, vectors_ecef_to_altaz, vectors_gcrs_to_altaz, vectors_camera_to_altaz
        )

        # 1) frame -> AltAz (radians), same shape as vectors[...,3] -> alt/az (...,)
        if frame == "enu":
            alt_rad, az_rad = vectors_enu_to_altaz(vectors)
        elif frame == "ecef":
            alt_rad, az_rad = vectors_ecef_to_altaz(vectors, location, time_utc)
        elif frame == "gcrs":
            alt_rad, az_rad = vectors_gcrs_to_altaz(vectors, location, time_utc)
        elif frame == "camera":
            alt_rad, az_rad = vectors_camera_to_altaz(vectors, camera_yaw_pitch_roll_deg)
        else:
            raise ValueError(f"Unknown frame: {frame}")

        # 2) Engine wants DEGREES, broadcast-friendly with a leading time axis.
        alt_deg = _np.rad2deg(alt_rad).astype(_np.float32)
        az_deg  = _np.rad2deg(az_rad).astype(_np.float32)

        # Preserve original spatial shape
        lead = alt_deg.shape
        alt_b = alt_deg[None, ...]   # (1, *lead)
        az_b  = az_deg[None, ...]    # (1, *lead)

        # 3) Ensure time is array-like (T,)
        if isinstance(time_utc, _Time) and getattr(time_utc, "isscalar", False):
            times = _Time([time_utc])
        else:
            times = _Time(time_utc)
            if getattr(times, "isscalar", False):
                times = _Time([times])

        if altitude_min_clip_deg is None:
            altitude_min_clip_deg = float(_np.nanmin(alt_deg))

        # 4) Call the sky model (returns (T, *lead) arrays)
        results, names = self.engine.simulate_sky_polarization(
            sky_model=sky_model,
            observation_location=location,
            times=times,
            cie_sky_type=cie_sky_type,
            altitudes=alt_b,
            azimuths=az_b,
            altitude_min_clip=altitude_min_clip_deg,
            azimuth_rotation_angle=0.0,
            accuracy=accuracy
        )

        # 5) Drop T if T==1 and cast back to the original shape
        outs = []
        for v in results:
            a = _np.asarray(v)
            if a.ndim >= 1 and a.shape[0] == 1:
                a = a[0]
            outs.append(a.astype(_np.float32))

        dop, aop, radiance, *rest = outs
        return dict(
            dop=dop.reshape(lead),
            aop=aop.reshape(lead),
            radiance=radiance.reshape(lead),
            frame=frame,
            alt=_np.deg2rad(alt_deg).reshape(lead),
            azi=_np.deg2rad(az_deg).reshape(lead),
            meta=dict(
                sky_model=sky_model,
                cie_sky_type=cie_sky_type,
                camera_ypr_deg=camera_yaw_pitch_roll_deg if frame=="camera" else None
            )
        )






    def simulate_to_frame(
        self,
        sky_model: str,
        location: EarthLocation,
        time_utc: Time,
        cie_sky_type: int = 12,
        alt_deg: NDArray[np.float32] = np.linspace(5,85,161).astype(np.float32),
        azi_deg: NDArray[np.float32] = np.linspace(-180,180,361).astype(np.float32),
        frame: FrameName = "ecef",
        camera_yaw_pitch_roll_deg: Tuple[float,float,float] = (0.0,0.0,0.0),
        rotate_aop: bool = False
    ) -> dict:
        """
        Returns dict with:
          dop, aop, radiance   -> (A,Z)
          vectors              -> (A,Z,3) unit vectors in requested frame
          frame, alt, azi, meta
        """
        (dop, aop, radiance, *rest), names, alt, azi = self.simulate_sky(
            sky_model=sky_model,
            location=location,
            time_utc=time_utc,
            cie_sky_type=cie_sky_type,
            alt_deg=alt_deg,
            azi_deg=azi_deg
        )
        ALT, AZI = altaz_grid(alt, azi)
        enu = altaz_to_unit_xyz_local(ALT, AZI)

        if frame in ("altaz","enu"):
            vectors = enu
        elif frame == "ecef":
            vectors = enu_to_ecef_unit(enu, location, time_utc)
        elif frame == "gcrs":
            vectors = enu_to_gcrs_unit(enu, location, time_utc)
        elif frame == "camera":
            y, p, r = camera_yaw_pitch_roll_deg
            y = np.deg2rad(y); p = np.deg2rad(p); r = np.deg2rad(r)
            Rz = np.array([[ np.cos(y), -np.sin(y), 0],
                           [ np.sin(y),  np.cos(y), 0],
                           [         0,          0, 1]], dtype=np.float32)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(p), -np.sin(p)],
                           [0, np.sin(p),  np.cos(p)]], dtype=np.float32)
            Ry = np.array([[ np.cos(r), 0, np.sin(r)],
                           [         0, 1,         0],
                           [-np.sin(r), 0, np.cos(r)]], dtype=np.float32)
            R = Rz @ Rx @ Ry
            vectors = enu @ R.T
        else:
            raise ValueError(f"Unknown frame: {frame}")

        if rotate_aop:
            basis_angle = np.arctan2(vectors[...,1], vectors[...,0])
            aop = rotate_aop_altaz_to_basis(aop, basis_angle)

        return dict(
            dop=dop.astype(np.float32),
            aop=aop.astype(np.float32),
            radiance=radiance.astype(np.float32),
            vectors=vectors.astype(np.float32),
            frame=frame,
            alt=ALT.astype(np.float32),
            azi=AZI.astype(np.float32),
            meta=dict(
                sky_model=sky_model,
                lat=float(location.lat.to_value(u.deg)),
                lon=float(location.lon.to_value(u.deg)),
                height=float(location.height.to_value(u.m)),
                time=str(time_utc.isot),
                cie_sky_type=cie_sky_type,
                camera_ypr_deg=camera_yaw_pitch_roll_deg,
                rotate_aop=rotate_aop
            )
        )
