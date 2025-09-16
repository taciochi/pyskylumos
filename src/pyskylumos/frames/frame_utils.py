from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from astropy.time import Time
from astropy.coordinates import AltAz, ITRS, GCRS, CartesianRepresentation, EarthLocation
import astropy.units as u

def altaz_grid(alt: NDArray[np.float32], azi: NDArray[np.float32]) -> tuple[NDArray, NDArray]:
    ALT, AZI = np.meshgrid(alt, azi, indexing="ij")
    return ALT, AZI

def altaz_to_unit_xyz_local(alt: NDArray, azi: NDArray) -> NDArray:
    """
    Alt/Az (radians) -> local ENU unit vectors (E,N,U).
    Azimuth per astropy: 0=North, 90=East.
    """
    ce = np.cos(alt)
    E = ce * np.sin(azi)
    N = ce * np.cos(azi)
    U = np.sin(alt)
    return np.stack([E, N, U], axis=-1).astype(np.float32)

def enu_to_ecef_unit(enu: NDArray, loc: EarthLocation, obstime: Time) -> NDArray:
    E, N, U = enu[...,0], enu[...,1], enu[...,2]
    alt = np.arcsin(U)
    azi = np.arctan2(E, N)
    altaz = AltAz(obstime=obstime, location=loc, alt=alt*u.rad, az=azi*u.rad, distance=1*u.one)
    itrs = altaz.transform_to(ITRS(obstime=obstime))
    xyz = itrs.cartesian.get_xyz().value
    xyz = np.moveaxis(xyz, 0, -1).astype(np.float32)
    nrm = np.linalg.norm(xyz, axis=-1, keepdims=True) + 1e-12
    return xyz / nrm

def enu_to_gcrs_unit(enu: NDArray, loc: EarthLocation, obstime: Time) -> NDArray:
    E, N, U = enu[...,0], enu[...,1], enu[...,2]
    alt = np.arcsin(U)
    azi = np.arctan2(E, N)
    altaz = AltAz(obstime=obstime, location=loc, alt=alt*u.rad, az=azi*u.rad, distance=1*u.one)
    gcrs = altaz.transform_to(GCRS(obstime=obstime))
    xyz = gcrs.cartesian.get_xyz().value
    xyz = np.moveaxis(xyz, 0, -1).astype(np.float32)
    nrm = np.linalg.norm(xyz, axis=-1, keepdims=True) + 1e-12
    return xyz / nrm

def rotate_aop_altaz_to_basis(aop_altaz: NDArray, basis_angle: NDArray) -> NDArray:
    """
    Local/approximate AoP rotation: subtract target basis angle (radians).
    For precision over wide FOVs, prefer spherical parallel transport.
    """
    return (aop_altaz - basis_angle).astype(np.float32)

def _flatten_vectors(v: NDArray) -> tuple[NDArray, tuple]:
    v = np.asarray(v, dtype=np.float32)
    if v.shape[-1] != 3:
        raise ValueError("vectors must have shape (..., 3)")
    lead = v.shape[:-1]
    return v.reshape(-1, 3), lead

def vectors_enu_to_altaz(enu: NDArray) -> tuple[NDArray, NDArray]:
    """
    ENU unit vectors (...,3) -> (alt_rad, az_rad) with same leading shape.
    Azimuth: 0=N, +90°=E (astropy convention). Altitude: 0=horizon, +90°=zenith.
    """
    enu = np.asarray(enu, dtype=np.float32)
    E, N, U = enu[..., 0], enu[..., 1], enu[..., 2]
    alt = np.arcsin(np.clip(U, -1.0, 1.0))
    az  = np.arctan2(E, N)  # [-pi, pi]
    return alt.astype(np.float32), az.astype(np.float32)

def vectors_ecef_to_altaz(ecef: NDArray, loc: EarthLocation, obstime: Time) -> tuple[NDArray, NDArray]:
    v2, lead = _flatten_vectors(ecef)
    rep = CartesianRepresentation(x=v2[:,0]*u.m, y=v2[:,1]*u.m, z=v2[:,2]*u.m)
    itrs = ITRS(rep, obstime=obstime)
    altaz = itrs.transform_to(AltAz(location=loc, obstime=obstime))
    alt = altaz.alt.to_value(u.rad).reshape(lead)
    az  = altaz.az.to_value(u.rad).reshape(lead)
    return alt.astype(np.float32), az.astype(np.float32)

def vectors_gcrs_to_altaz(gcrs: NDArray, loc: EarthLocation, obstime: Time) -> tuple[NDArray, NDArray]:
    v2, lead = _flatten_vectors(gcrs)
    rep = CartesianRepresentation(x=v2[:,0]*u.m, y=v2[:,1]*u.m, z=v2[:,2]*u.m)
    g = GCRS(rep, obstime=obstime)
    altaz = g.transform_to(AltAz(location=loc, obstime=obstime))
    alt = altaz.alt.to_value(u.rad).reshape(lead)
    az  = altaz.az.to_value(u.rad).reshape(lead)
    return alt.astype(np.float32), az.astype(np.float32)

def ypr_to_R_enu_from_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> NDArray:
    """R: camera→ENU; yaw about +U, pitch about +E, roll about +N."""
    y = np.deg2rad(yaw_deg); p = np.deg2rad(pitch_deg); r = np.deg2rad(roll_deg)
    Rz = np.array([[ np.cos(y), -np.sin(y), 0],
                   [ np.sin(y),  np.cos(y), 0],
                   [         0,          0, 1]], dtype=np.float32)  # yaw
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p),  np.cos(p)]], dtype=np.float32)   # pitch
    Ry = np.array([[ np.cos(r), 0, np.sin(r)],
                   [         0, 1,         0],
                   [-np.sin(r), 0, np.cos(r)]], dtype=np.float32)   # roll
    return (Rz @ Rx @ Ry).astype(np.float32)

def vectors_camera_to_altaz(v_cam: NDArray,
                            yaw_pitch_roll_deg: tuple[float,float,float]) -> tuple[NDArray, NDArray]:
    """Camera unit vectors (...,3) + extrinsics → AltAz (radians) via ENU."""
    R = ypr_to_R_enu_from_camera(*yaw_pitch_roll_deg)  # camera→ENU
    v_enu = v_cam @ R.T
    return vectors_enu_to_altaz(v_enu)