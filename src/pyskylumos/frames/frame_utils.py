# src/pyskylumos/frames/frame_utils.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    EarthLocation, ITRS, GCRS, CartesianRepresentation
)

# -----------------------------
# Public helpers used by generator.py
# -----------------------------
def altaz_grid(
    alt_deg: NDArray | None = None,
    az_deg: NDArray | None = None,
    *,
    n_alt: int = 91,
    n_az: int = 361,
    alt_range_deg: tuple[float, float] = (0.0, 90.0),
    az_range_deg: tuple[float, float] = (-180.0, 180.0),
) -> tuple[NDArray, NDArray]:
    """Build a broadcast-friendly Alt/Az grid (degrees). Returns (ALT, AZI) shaped (n_alt, n_az)."""
    if alt_deg is None:
        alt_deg = np.linspace(alt_range_deg[0], alt_range_deg[1], n_alt, dtype=np.float32)
    else:
        alt_deg = np.asarray(alt_deg, dtype=np.float32).ravel()
        n_alt = alt_deg.size
    if az_deg is None:
        az_deg = np.linspace(az_range_deg[0], az_range_deg[1], n_az, dtype=np.float32)
    else:
        az_deg = np.asarray(az_deg, dtype=np.float32).ravel()
        n_az = az_deg.size
    ALT, AZI = np.meshgrid(alt_deg, az_deg, indexing="ij")
    return ALT.astype(np.float32), AZI.astype(np.float32)

# -----------------------------
# Internal utilities
# -----------------------------
def _normalize(v: NDArray, eps: float = 1e-12) -> NDArray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return (v / n).astype(np.float64)

def _enu_basis_in_ecef(lat_rad: float, lon_rad: float) -> NDArray:
    """
    Return a 3x3 matrix whose ROWS are the ENU basis vectors expressed in ECEF:
      R[0,:] = East_in_ECEF, R[1,:] = North_in_ECEF, R[2,:] = Up_in_ECEF
    """
    sφ, cφ = np.sin(lat_rad), np.cos(lat_rad)
    sλ, cλ = np.sin(lon_rad), np.cos(lon_rad)
    return np.array([
        [-sλ,            cλ,           0.0],   # E
        [-sφ * cλ,  -sφ * sλ,      cφ],       # N
        [ cφ * cλ,   cφ * sλ,      sφ],       # U
    ], dtype=np.float64)

# -----------------------------
# Alt/Az <-> local ENU unit vectors
# -----------------------------
def altaz_to_unit_xyz_local(alt_rad: NDArray, az_rad: NDArray) -> NDArray:
    """
    Alt/Az (radians) -> local ENU unit vector (...,3).
    Azimuth: 0=N, +90°=E. Altitude: 0=horizon, +pi/2=zenith.
    """
    alt = np.asarray(alt_rad, dtype=np.float64)
    az  = np.asarray(az_rad,  dtype=np.float64)
    ca, sa = np.cos(alt), np.sin(alt)
    saz, caz = np.sin(az), np.cos(az)
    E = saz * ca
    N = caz * ca
    U = sa
    v = np.stack([E, N, U], axis=-1)
    return _normalize(v).astype(np.float32)

def unit_xyz_local_to_altaz(enu_vecs: NDArray) -> tuple[NDArray, NDArray]:
    """Local ENU unit vectors (...,3) -> (alt_rad, az_rad) with same leading shape."""
    v = _normalize(enu_vecs)
    E, N, U = v[..., 0], v[..., 1], v[..., 2]
    alt = np.arcsin(np.clip(U, -1.0, 1.0))
    az  = np.arctan2(E, N)  # [-pi, pi]
    return alt.astype(np.float32), az.astype(np.float32)

# Alias used elsewhere
def vectors_enu_to_altaz(enu: NDArray) -> tuple[NDArray, NDArray]:
    return unit_xyz_local_to_altaz(enu)

# -----------------------------
# ENU <-> ECEF unit vectors
# -----------------------------
def enu_to_ecef_unit(enu: NDArray, loc: EarthLocation, obstime: Time | None = None) -> NDArray:
    """
    ENU unit vectors (...,3) -> ECEF unit vectors (...,3) at geodetic location 'loc'.
    'obstime' accepted for signature compatibility; not used by this pure rotation.
    """
    v = _normalize(enu)
    lat = float(loc.lat.to_value(u.rad))
    lon = float(loc.lon.to_value(u.rad))
    R = _enu_basis_in_ecef(lat, lon)  # rows: E,N,U in ECEF
    ecef = np.tensordot(v, R, axes=1)  # v @ R
    return _normalize(ecef).astype(np.float32)

def ecef_to_enu_unit(ecef: NDArray, loc: EarthLocation) -> NDArray:
    """
    ECEF unit vectors (...,3) -> ENU unit vectors (...,3) at geodetic location 'loc'.
    """
    v = _normalize(ecef)
    lat = float(loc.lat.to_value(u.rad))
    lon = float(loc.lon.to_value(u.rad))
    R = _enu_basis_in_ecef(lat, lon)  # rows: E,N,U in ECEF
    enu = np.tensordot(v, R.T, axes=1)  # v @ R^T
    return _normalize(enu).astype(np.float32)

# -----------------------------
# ECEF/GCRS <-> ENU unit vectors
# -----------------------------
def ecef_to_gcrs_unit(ecef: NDArray, obstime: Time) -> NDArray:
    """ECEF unit vectors (...,3) -> GCRS unit vectors (...,3) at 'obstime'."""
    v = _normalize(ecef)
    lead = v.shape[:-1]
    rep = CartesianRepresentation(
        x=v[..., 0].reshape(-1) * u.one,
        y=v[..., 1].reshape(-1) * u.one,
        z=v[..., 2].reshape(-1) * u.one,
    )
    sc_i = ITRS(rep, obstime=obstime)
    sc_g = sc_i.transform_to(GCRS(obstime=obstime))
    xyz = np.stack([sc_g.cartesian.x.value, sc_g.cartesian.y.value, sc_g.cartesian.z.value], axis=-1)
    return _normalize(xyz).reshape(*lead, 3).astype(np.float32)

def gcrs_to_ecef_unit(gcrs: NDArray, obstime: Time) -> NDArray:
    """GCRS unit vectors (...,3) -> ECEF unit vectors (...,3) at 'obstime'."""
    v = _normalize(gcrs)
    lead = v.shape[:-1]
    rep = CartesianRepresentation(
        x=v[..., 0].reshape(-1) * u.one,
        y=v[..., 1].reshape(-1) * u.one,
        z=v[..., 2].reshape(-1) * u.one,
    )
    sc_g = GCRS(rep, obstime=obstime)
    sc_i = sc_g.transform_to(ITRS(obstime=obstime))
    xyz = np.stack([sc_i.cartesian.x.value, sc_i.cartesian.y.value, sc_i.cartesian.z.value], axis=-1)
    return _normalize(xyz).reshape(*lead, 3).astype(np.float32)

def enu_to_gcrs_unit(enu: NDArray, loc: EarthLocation, obstime: Time) -> NDArray:
    """ENU unit vectors (...,3) -> GCRS unit vectors (...,3) at (loc, obstime)."""
    ecef = enu_to_ecef_unit(enu, loc=loc)  # obstime not needed for pure rotation
    return ecef_to_gcrs_unit(ecef, obstime=obstime)

def gcrs_to_enu_unit(gcrs: NDArray, loc: EarthLocation, obstime: Time) -> NDArray:
    """GCRS unit vectors (...,3) -> ENU unit vectors (...,3) at (loc, obstime)."""
    ecef = gcrs_to_ecef_unit(gcrs, obstime=obstime)
    return ecef_to_enu_unit(ecef, loc=loc)

# -----------------------------
# Vectors in ECEF/GCRS -> Alt/Az
# -----------------------------
def vectors_ecef_to_altaz(ecef: NDArray, loc: EarthLocation, obstime: Time) -> tuple[NDArray, NDArray]:
    """Convert *topocentric* ECEF (ITRS) direction vectors at 'loc' into Alt/Az (radians)."""
    enu = ecef_to_enu_unit(ecef, loc=loc)
    return unit_xyz_local_to_altaz(enu)

def vectors_gcrs_to_altaz(gcrs: NDArray, loc: EarthLocation, obstime: Time) -> tuple[NDArray, NDArray]:
    """Convert GCRS direction vectors into Alt/Az at (loc, obstime)."""
    ecef = gcrs_to_ecef_unit(gcrs, obstime=obstime)
    return vectors_ecef_to_altaz(ecef, loc=loc, obstime=obstime)

# -----------------------------
# AoP basis rotation
# -----------------------------
def _wrap_orientation_pi(angle_rad: NDArray) -> NDArray:
    """
    Wrap an orientation (period π) to [-π/2, π/2).
    """
    return ((angle_rad + np.pi/2) % np.pi) - np.pi/2

def rotate_aop_altaz_to_basis(aop_altaz_rad: NDArray, basis_angle_rad: NDArray) -> NDArray:
    """
    Rotate an AoP given in the AltAz basis into a new tangent-plane basis
    whose x-axis is rotated by 'basis_angle_rad'.

    Parameters
    ----------
    aop_altaz_rad : array-like (radians)
        Angle of polarization measured in the AltAz tangent-plane basis.
    basis_angle_rad : array-like (radians)
        Rotation of the *basis* in the same tangent plane (positive = CCW).
        Example: for a basis aligned with the projection of a vector, use atan2(y, x).

    Returns
    -------
    aop_rotated : array-like (radians), wrapped to [-π/2, π/2)
    """
    aop_altaz_rad = np.asarray(aop_altaz_rad, dtype=np.float64)
    basis_angle_rad = np.asarray(basis_angle_rad, dtype=np.float64)
    return _wrap_orientation_pi(aop_altaz_rad - basis_angle_rad).astype(np.float32)

# -----------------------------
# Camera vectors -> Alt/Az
# -----------------------------
def ypr_to_R_enu_from_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> NDArray:
    """
    Return rotation matrix R that maps CAMERA vectors to ENU: v_enu = R @ v_cam.
    - yaw:   about +U (Up)   (deg)
    - pitch: about +E (East) (deg)
    - roll:  about +N (North)(deg)
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[ np.cos(y), -np.sin(y), 0.0],
                   [ np.sin(y),  np.cos(y), 0.0],
                   [        0.,         0., 1.0]], dtype=np.float64)
    Rx = np.array([[1.0,        0.,         0.       ],
                   [0.,  np.cos(p), -np.sin(p)],
                   [0.,  np.sin(p),  np.cos(p)]], dtype=np.float64)
    Ry = np.array([[ np.cos(r), 0.,  np.sin(r)],
                   [        0., 1.,         0.],
                   [-np.sin(r), 0.,  np.cos(r)]], dtype=np.float64)
    return (Rz @ Rx @ Ry).astype(np.float64)

def vectors_camera_to_altaz(v_cam: NDArray, yaw_pitch_roll_deg: tuple[float, float, float]) -> tuple[NDArray, NDArray]:
    """Camera-frame unit vectors (...,3) -> Alt/Az at the local ENU defined by yaw/pitch/roll."""
    v = _normalize(v_cam)
    R = ypr_to_R_enu_from_camera(*yaw_pitch_roll_deg)  # camera -> ENU
    enu = np.tensordot(v, R.T, axes=1)                 # v @ R^T
    return unit_xyz_local_to_altaz(enu)
