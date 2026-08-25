"""Internal rigid-pose and stereographic polarization-basis geometry.

Coordinates use a North-East-Up Cartesian basis. Camera rotations map
sensor-local vectors into world vectors; polarization transport applies the
inverse rotation to a physical tangent vector so its final angle is measured in
the sensor's analyzer chart.
"""

from __future__ import annotations

import numpy as np

from pyskylumos._types import FloatArray


def sensor_tilt_rotation(azimuthal_tilt: float, tilt_angle: float) -> FloatArray:
    """Return the right-handed sensor-to-world tilt rotation.

    Args:
        azimuthal_tilt: Horizontal rotation-axis parameter in radians. The axis
            is ``(cos(a), -sin(a), 0)`` in North-East-Up coordinates.
        tilt_angle: Right-handed rotation angle in radians.

    Returns:
        A ``(3, 3)`` float64 rotation matrix mapping sensor vectors to world
        vectors.
    """
    cosine_axis = np.cos(azimuthal_tilt)
    sine_axis = np.sin(azimuthal_tilt)
    cosine_tilt = np.cos(tilt_angle)
    sine_tilt = np.sin(tilt_angle)

    rotation_canonical = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine_tilt, -sine_tilt],
            [0.0, sine_tilt, cosine_tilt],
        ],
        dtype=np.float64,
    )
    transition = np.array(
        [
            [cosine_axis, -sine_axis, 0.0],
            [sine_axis, cosine_axis, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(transition.T @ rotation_canonical @ transition, dtype=np.float64)


def directions_from_degrees(azimuths: FloatArray, altitudes: FloatArray) -> FloatArray:
    """Convert degree-valued azimuth/altitude arrays to Cartesian unit vectors.

    Args:
        azimuths: Azimuths measured from North toward East, in degrees.
        altitudes: Altitudes above the local horizontal plane, in degrees.

    Returns:
        Float64 vectors with ``(x, y, z) = (North, East, Up)`` on the final axis.
    """
    azimuths_radians = np.deg2rad(azimuths)
    altitudes_radians = np.deg2rad(altitudes)
    return np.asarray(
        np.stack(
            (
                np.cos(altitudes_radians) * np.cos(azimuths_radians),
                np.cos(altitudes_radians) * np.sin(azimuths_radians),
                np.sin(altitudes_radians),
            ),
            axis=-1,
        ),
        dtype=np.float64,
    )


def directions_to_degrees(directions: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert Cartesian direction vectors to degree-valued spherical angles.

    Args:
        directions: Cartesian vectors whose final axis contains North, East,
            and Up components.

    Returns:
        Float64 azimuth and altitude arrays in degrees.
    """
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]
    azimuths = np.rad2deg(np.arctan2(y, x))
    altitudes = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return (
        np.asarray(azimuths, dtype=np.float64),
        np.asarray(altitudes, dtype=np.float64),
    )


def rotate_directions(
    azimuths: FloatArray,
    altitudes: FloatArray,
    rotation: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply a sensor-to-world rotation to degree-valued sampling directions.

    Args:
        azimuths: Sensor-local azimuth grid in degrees.
        altitudes: Sensor-local altitude grid in degrees.
        rotation: ``(3, 3)`` sensor-to-world rotation matrix.

    Returns:
        World azimuth and altitude grids in degrees.
    """
    directions = directions_from_degrees(azimuths, altitudes)
    rotated = np.asarray(np.einsum("ij,...j->...i", rotation, directions), dtype=np.float64)
    return directions_to_degrees(rotated)


def transport_stereographic_aop_to_sensor(
    angle_of_polarization: FloatArray,
    world_directions: FloatArray,
    sensor_directions: FloatArray,
    sensor_to_world_rotation: FloatArray,
) -> FloatArray:
    """Transport world-chart AOP into the tilted sensor's stereographic chart.

    The AOP direction in the world stereographic plane is lifted through the
    differential of inverse stereographic projection. The resulting tangent
    vector is rotated into sensor coordinates and projected through the
    differential of forward stereographic projection. Both charts use the
    zenith-centred chart obtained by projection from nadir,
    ``(u, v) = (x, y) / (1 + z)``.

    Args:
        angle_of_polarization: World-chart axial AOP in radians. Leading axes,
            such as time, may precede the spatial grid.
        world_directions: World unit vectors for the spatial grid.
        sensor_directions: Corresponding sensor-local unit vectors.
        sensor_to_world_rotation: Rotation mapping sensor vectors into world
            coordinates.

    Returns:
        Float64 AOP in the sensor chart, wrapped onto ``[-pi/2, pi/2)``. NaNs
        propagate, and the exact stereographic nadir singularity returns NaN.
    """
    spatial_rank = world_directions.ndim - 1
    leading_rank = angle_of_polarization.ndim - spatial_rank
    leading_axes = (1,) * leading_rank

    world_x = world_directions[..., 0]
    world_y = world_directions[..., 1]
    world_z = world_directions[..., 2]
    world_chart_scale = 1.0 + world_z

    chart_epsilon = 8.0 * np.finfo(np.float64).eps
    world_chart_valid = np.abs(world_chart_scale) > chart_epsilon

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = world_x / world_chart_scale
        v = world_y / world_chart_scale
        denominator = 1.0 + u**2 + v**2
        denominator_squared = denominator**2

        derivative_u = np.stack(
            (
                2.0 * (1.0 - u**2 + v**2) / denominator_squared,
                -4.0 * u * v / denominator_squared,
                -4.0 * u / denominator_squared,
            ),
            axis=-1,
        )
        derivative_v = np.stack(
            (
                -4.0 * u * v / denominator_squared,
                2.0 * (1.0 + u**2 - v**2) / denominator_squared,
                -4.0 * v / denominator_squared,
            ),
            axis=-1,
        )

        basis_shape = leading_axes + derivative_u.shape
        derivative_u = derivative_u.reshape(basis_shape)
        derivative_v = derivative_v.reshape(basis_shape)
        polarization_tangent_world = (
            derivative_u * np.cos(angle_of_polarization)[..., np.newaxis]
            + derivative_v * np.sin(angle_of_polarization)[..., np.newaxis]
        )
        polarization_tangent_sensor = np.einsum(
            "ij,...j->...i",
            sensor_to_world_rotation.T,
            polarization_tangent_world,
        )

        sensor_x = sensor_directions[..., 0].reshape(leading_axes + world_x.shape)
        sensor_y = sensor_directions[..., 1].reshape(leading_axes + world_y.shape)
        sensor_z = sensor_directions[..., 2].reshape(leading_axes + world_z.shape)
        sensor_chart_scale = 1.0 + sensor_z
        sensor_chart_denominator = sensor_chart_scale**2

        tangent_x = polarization_tangent_sensor[..., 0]
        tangent_y = polarization_tangent_sensor[..., 1]
        tangent_z = polarization_tangent_sensor[..., 2]
        plane_u = (sensor_chart_scale * tangent_x - sensor_x * tangent_z) / sensor_chart_denominator
        plane_v = (sensor_chart_scale * tangent_y - sensor_y * tangent_z) / sensor_chart_denominator
        sensor_aop = np.arctan2(plane_v, plane_u)

    sensor_chart_valid = np.abs(sensor_chart_scale) > chart_epsilon
    valid = (
        world_chart_valid.reshape(leading_axes + world_chart_valid.shape)
        & sensor_chart_valid
        & np.isfinite(angle_of_polarization)
        & np.isfinite(sensor_aop)
    )
    wrapped = (sensor_aop + np.pi / 2.0) % np.pi - np.pi / 2.0
    return np.asarray(np.where(valid, wrapped, np.nan), dtype=np.float64)


__all__ = (
    "directions_from_degrees",
    "rotate_directions",
    "sensor_tilt_rotation",
    "transport_stereographic_aop_to_sensor",
)
