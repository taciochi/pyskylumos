"""
Demo: ECEF position + directions (using frame_utils via the wrapper)

- Camera: 2048×2048, IMX250MZR 3.45 µm pixels, 1.8 mm equisolid fisheye
- Site:    lat=53.4808°, lon=-2.2426°, h=50 m
- Time:    2025-09-16T12:00:00Z

Key idea:
- Use UNCLIPPED lens alt/az to form rays and masks (so true below-horizon
  pixels are excluded), but still pass altitude_min_clip_deg=1.0 to the engine.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from pyskylumos.frames import PolarizationFrameGenerator, EngineConfig
from pyskylumos.frames.frame_utils import enu_to_ecef_unit, vectors_ecef_to_altaz
from pyskylumos.sensor.OpticalConjugator import OpticalConjugator


# --------------------------
# Helpers
# --------------------------
def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _imshow_masked(arr: np.ndarray, title: str, png_path: str):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)  # NaNs transparent
    plt.figure()
    plt.imshow(np.ma.masked_invalid(arr), origin="lower", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


# --------------------------
# Demo
# --------------------------
def run_demo():
    here = os.path.dirname(__file__)
    out_dir = _ensure_outdir(os.path.join(here, "demo_outputs_ecef_using_frame_utils"))

    # --- Camera / sensor ---
    H = W = 2048
    pixel_size_um = 3.45
    focal_length_um = 1800.0
    lens_projection = "equi_solid_angle"

    # --- Site & time ---
    loc = EarthLocation(lat=53.4808 * u.deg, lon=-2.2426 * u.deg, height=50 * u.m)
    t = Time("2025-09-16T12:00:00", scale="utc")

    # --- Lens: get UNCLIPPED per-pixel AZ/ALT (deg) ---
    oc = OpticalConjugator(
        lens_conjugation_type=lens_projection,
        number_pixels_vertical=H,
        number_pixels_horizontal=W,
        lens_focal_length_micrometers=focal_length_um,
        sensor_pixel_size_square_micrometers=pixel_size_um,
    )
    # <--- IMPORTANT: no clipping here
    az_deg_raw, alt_deg_raw = oc.get_azimuth_altitude(altitude_min_clip=None)  # (H,W)

    # --- Fisheye circle mask (strict) ---
    f_px = focal_length_um / pixel_size_um           # ≈ 521.7 px
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    r = np.hypot(xx - cx, yy - cy)
    safety = 1.0                                     # px margin to avoid boundary artifacts
    mask_circle = r <= (2.0 * f_px - safety)

    # --- Horizon mask from UNCLIPPED altitude ---
    # use a tiny positive epsilon to avoid salt-and-pepper right on the boundary
    eps_deg = 0.2
    mask_horizon = alt_deg_raw >= (0.0 + eps_deg)

    # Combined validity
    mask_valid = mask_circle & mask_horizon

    # --- Build ENU rays from UNCLIPPED Alt/Az (valid pixels only) ---
    az = np.deg2rad(az_deg_raw).astype(np.float64)
    alt = np.deg2rad(alt_deg_raw).astype(np.float64)
    ca, sa = np.cos(alt), np.sin(alt)
    saz, caz = np.sin(az), np.cos(az)
    enu = np.stack([saz * ca,  # East
                    caz * ca,  # North
                    sa], axis=-1).astype(np.float64)  # Up

    # Replace invalid rays with zenith BEFORE conversion/simulation
    enu_dummy = np.zeros((H, W, 3), dtype=np.float64); enu_dummy[..., 2] = 1.0
    enu[~mask_valid] = enu_dummy[~mask_valid]

    # --- Convert ENU -> ECEF via helper ---
    ecef_dirs = enu_to_ecef_unit(enu, loc=loc)  # (H,W,3), float32

    # Optional sanity check: round-trip Alt/Az from these rays
    alt_rt, az_rt = vectors_ecef_to_altaz(ecef_dirs, loc, t)
    err_alt = np.nanmax(np.abs(alt_rt - np.deg2rad(alt)))
    err_az  = np.nanmax(np.abs(((az_rt - np.deg2rad(az)) + np.pi) % (2*np.pi) - np.pi))
    print(f"[check] round-trip: max alt err={err_alt:.3e} rad, max az err={err_az:.3e} rad")

    # --- Engine wrapper (uses frame_utils internally) ---
    cfg = EngineConfig(
        sensor_pixel_size_square_micrometers=pixel_size_um,
        lens_conjugation_type="image",
        number_pixels_vertical=H,
        number_pixels_horizontal=W,
        lens_focal_length_micrometers=focal_length_um,
        tolerance=0.5,
        extinction_ratio=100.0,
        pixel_saturation_ratio=0.95,
        adc_resolution=12,
        signal_to_noise_ratio=40.0,
    )
    gen = PolarizationFrameGenerator(cfg)

    out = gen.simulate_from_vectors(
        sky_model="rayleigh",
        location=loc,
        time_utc=t,
        vectors=ecef_dirs,
        frame="ecef",
        cie_sky_type=12,
        altitude_min_clip_deg=1.0,   # clip only inside engine, not when forming rays
        accuracy=False,
    )

    dop = out["dop"].astype(np.float32)
    aop = out["aop"].astype(np.float32)
    rad = out["radiance"].astype(np.float32)

    # --- Hard-mask invalid pixels AFTER simulation (transparent on plots) ---
    dop[~mask_valid] = np.nan
    aop[~mask_valid] = np.nan
    rad[~mask_valid] = np.nan

    # --- Save ---
    base = os.path.join(out_dir, "ecef_using_frame_utils")
    _imshow_masked(dop, "DoP (ECEF using frame_utils)", base + "_dop.png")
    _imshow_masked(aop, "AoP (ECEF using frame_utils) [rad]", base + "_aop.png")
    _imshow_masked(rad, "Radiance (ECEF using frame_utils)", base + "_radiance.png")

    print(f"\n[demo] Outputs written to: {out_dir}\n")


if __name__ == "__main__":
    run_demo()
