"""
Demo
------------
- Uses OpticalConjugator to compute per-pixel azimuth/altitude (degrees)
  for a square sensor with a 1.8 mm fisheye.
- Calls Engine.simulate_sky_polarization to get DoP/AoP/Radiance at each pixel.
- Saves square PNGs and a compressed NPZ.

"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from pyskylumos.engine.Engine import Engine
from pyskylumos.sensor.SlicingPattern import SlicingPattern
from pyskylumos.sensor.OpticalConjugator import OpticalConjugator


# --------------------------
# Helpers
# --------------------------
def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _save_png(basename: str, arr: np.ndarray, title: str):
    plt.figure()
    plt.imshow(arr, origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(basename + ".png", dpi=160)
    plt.close()


# --------------------------
# Demo
# --------------------------
def run_demo():
    here = os.path.dirname(__file__)
    out_dir = _ensure_outdir(os.path.join(here, "demo_outputs_fisheye_from_package"))

    # --- Camera / sensor settings (square render) ---
    # If you want native 2048×2048, set H=W=2048 (this example uses 1024 for speed).
    H = W = 2048
    pixel_size_um = 3.45          # IMX250MZR pixel pitch (CS505MUP1-like)
    focal_length_um = 1800.0      # 1.8 mm fisheye
    lens_projection = "equi_solid_angle"  # equisolid fisheye per your OpticalConjugator
    conjugation_type = "image"     # camera imaging the distant sky

    # --- Site & time ---
    loc = EarthLocation(lat=53.4808 * u.deg, lon=-2.2426 * u.deg, height=50 * u.m)
    t = Time(["2025-09-16T12:00:00"], scale="utc")  # use a 1-element array to satisfy models

    # --- Build lens & get per-pixel az/alt in DEGREES (H×W) ---
    oc = OpticalConjugator(
        lens_conjugation_type=lens_projection,
        number_pixels_vertical=H,
        number_pixels_horizontal=W,
        lens_focal_length_micrometers=focal_length_um,
        sensor_pixel_size_square_micrometers=pixel_size_um,
    )
    # Clip a little above horizon to avoid singularities; tweak as needed
    az_deg, alt_deg = oc.get_azimuth_altitude(altitude_min_clip=1.0)

    # --- Build engine (original pyskylumos) ---
    pattern = {
        0:   SlicingPattern(0, 0, 2),
        45:  SlicingPattern(0, 1, 2),
        90:  SlicingPattern(1, 0, 2),
        135: SlicingPattern(1, 1, 2),
    }
    engine = Engine(
        sensor_pixel_size_square_micrometers=pixel_size_um,
        lens_conjugation_type=conjugation_type,
        number_pixels_vertical=H,
        number_pixels_horizontal=W,
        lens_focal_length_micrometers=focal_length_um,
        tolerance=0.5,
        extinction_ratio=100.0,
        pixel_saturation_ratio=0.95,
        adc_resolution=12,
        signal_to_noise_ratio=40.0,
        wire_grid_orientations_slicing=pattern,
    )

    # --- Sky simulation (all in ORIGINAL pyskylumos) ---
    # Pass degree arrays with a leading time axis (T=1) for broadcasting.
    results, names = engine.simulate_sky_polarization(
        sky_model="rayleigh",                # or "pan", "berry"
        observation_location=loc,
        times=t,                             # Time array (len 1)
        cie_sky_type=12,
        altitudes=alt_deg[None, ...],        # (1, H, W) degrees
        azimuths=az_deg[None, ...],          # (1, H, W) degrees
        altitude_min_clip=float(np.nanmin(alt_deg)),  # degrees
        azimuth_rotation_angle=0.0,
        accuracy=False,
    )

    # results is typically a list like [dop, aop, radiance, ...] with shape (T, H, W)
    # Drop the time axis (T=1) to get (H, W)
    outs = [np.asarray(a)[0].astype(np.float32) if np.asarray(a).ndim >= 3 and np.asarray(a).shape[0] == 1
            else np.asarray(a).astype(np.float32)
            for a in results]
    # Map common names (fallbacks if names missing)
    name_map = {n.lower(): i for i, n in enumerate(names)} if isinstance(names, (list, tuple)) else {}
    dop = outs[name_map.get("degree_of_polarization", 0)]
    aop = outs[name_map.get("angle_of_polarization", 1)]
    rad = outs[name_map.get("radiance", 2)]

    # --- Save outputs ---
    base = os.path.join(out_dir, "pyskylumos_fisheye_square")
    _save_png(base + "_dop", dop, "DoP (equisolid fisheye, square sensor)")
    _save_png(base + "_aop", aop, "AoP (equisolid fisheye, square sensor) [rad]")
    _save_png(base + "_radiance", rad, "Radiance (equisolid fisheye, square sensor)")

    np.savez_compressed(
        base + ".npz",
        dop=dop, aop=aop, radiance=rad,
        az_deg=az_deg.astype(np.float32),
        alt_deg=alt_deg.astype(np.float32),
        meta=dict(
            sensor_size=(H, W),
            pixel_size_um=pixel_size_um,
            focal_length_um=focal_length_um,
            lens_projection=lens_projection,
            cie_sky_type=12,
            time=str(t.isot[0]),
            lat=float(loc.lat.to_value(u.deg)),
            lon=float(loc.lon.to_value(u.deg)),
            height_m=float(loc.height.to_value(u.m)),
        )
    )

    print("\n[demo] Done.")
    print(f"[demo] Outputs written to: {out_dir}\n")


if __name__ == "__main__":
    run_demo()
