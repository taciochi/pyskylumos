"""Shared stereographic geometry and Berry's quartic polarization field.

This module holds the mathematics common to every quartic-singularity sky model:
:class:`~pyskylumos.sky_models.AsymmetricQuartic.AsymmetricQuartic`,
:class:`~pyskylumos.sky_models.Berry.Berry`,
:class:`~pyskylumos.sky_models.Pan.Pan` and
:class:`~pyskylumos.sky_models.QuEEN.QuEEN`. A single, individually tested
implementation therefore backs their shared geometry.

References:
    Berry M V, Dennis M R and Lee R L Jr 2004 *Polarization singularities in the
    clear sky* New J. Phys. **6** 162 -- Eqs. (2.6), (2.7) and (4.2).

    Berry M V 2015 *Nature's optics and our understanding of light* Contemp. Phys.
    **56** 2-16 -- Eq. (37).

See the project README mathematical reference, sections 2-4, for the derivations,
in particular the half-angle placement identity and the OpenSky-derived
``e^{-2i*alpha_s}`` global AOP convention.

All directions in this module belong to the fixed world chart. Camera-pose and
analyzer-basis transport are separate Engine operations.
"""

from numpy import (
    absolute,
    angle,
    arctan,
    clip,
    complex128,
    conjugate,
    cos,
    exp,
    ones_like,
    sin,
    sqrt,
    tan,
)

from pyskylumos._types import ComplexArray, FloatArray


def project(zenith_angle: FloatArray, azimuth: FloatArray) -> ComplexArray:
    """Project a world direction onto the zenith-centred stereographic plane.

    Implements ``xi = tan(phi / 2) * exp(i * alpha)`` (Berry 2015, Eq. 37), under
    which the visible hemisphere maps to the unit disc, the zenith to the origin
    and the horizon to ``|xi| = 1``.

    Args:
        zenith_angle: Zenith angle of the direction in radians.
        azimuth: World azimuth of the direction in radians.

    Returns:
        Complex stereographic coordinate of the direction.
    """
    return tan(zenith_angle / 2) * exp(1j * azimuth)


def unproject(projection: ComplexArray) -> tuple[FloatArray, FloatArray]:
    """Recover a world direction from its stereographic coordinate.

    Exact inverse of :func:`project`.

    Args:
        projection: Complex stereographic coordinate.

    Returns:
        Tuple of zenith angle and azimuth, both in radians. The azimuth is
        returned on ``(-pi, pi]``.
    """
    return 2 * arctan(absolute(projection)), angle(projection)


def antipode(projection: ComplexArray) -> ComplexArray:
    """Return the antipodal point of a stereographic coordinate.

    The antipodal map on the stereographic plane is the reciprocal conjugate
    ``xi -> -1 / conj(xi)`` (Berry 2004, section 2), which sends a direction to
    the diametrically opposite direction on the sky sphere.

    Args:
        projection: Complex stereographic coordinate.

    Returns:
        Complex stereographic coordinate of the antipodal direction.
    """
    return (-1 / conjugate(projection)).astype(complex128)


def offset_along_solar_vertical(
    sun_zenith_angle: FloatArray,
    sun_azimuth: FloatArray,
    angular_offset: FloatArray,
    above_sun: bool,
) -> ComplexArray:
    """Place a point at a given angular offset from the sun on the solar vertical.

    A direction offset by ``delta`` from the sun along the solar vertical great
    circle has zenith angle ``phi_s -/+ delta`` and azimuth ``alpha_s``, so its
    stereographic coordinate is ``tan((phi_s -/+ delta) / 2) * exp(i * alpha_s)``.
    Expanding with the tangent addition identity, and writing
    ``t = tan(phi_s / 2)`` and ``a = tan(delta / 2)``, gives the Moebius forms
    used here (Berry 2004, Eq. 2.6). The mapping is exact, not an approximation.

    Args:
        sun_zenith_angle: Zenith angle of the sun in radians.
        sun_azimuth: Azimuth of the sun in radians.
        angular_offset: Angular distance from the sun in radians.
        above_sun: Whether the point sits above the sun (towards the zenith).
            ``False`` places it below the sun (towards the horizon).

    Returns:
        Complex stereographic coordinate of the offset direction.
    """
    sun_radius: FloatArray = tan(sun_zenith_angle / 2)
    offset_radius: FloatArray = tan(angular_offset / 2)

    if above_sun:
        radius = (sun_radius - offset_radius) / (1 + sun_radius * offset_radius)
    else:
        radius = (sun_radius + offset_radius) / (1 - sun_radius * offset_radius)

    return (radius * exp(1j * sun_azimuth)).astype(complex128)


def place_on_solar_meridian(
    sun_zenith_angle: FloatArray,
    sun_azimuth: FloatArray,
    signed_offset: FloatArray,
) -> ComplexArray:
    """Place a point on the solar meridian by signed angular offset from the sun.

    The meridian is parametrised by ``psi = phi_s + signed_offset``, measured from
    the zenith through the sun. A negative ``psi`` makes ``tan(psi / 2)`` negative,
    which is the same complex number as a positive radius at azimuth
    ``alpha_s + pi``, so the anti-solar half needs no special case.

    Berry's antipodal map is exactly ``psi -> psi +/- pi``, so this reproduces
    :func:`antipode` when the offset is shifted by half a turn. Unlike a
    component-wise Moebius offset, this construction is exactly covariant under
    rotation of the whole scene.

    Args:
        sun_zenith_angle: Zenith angle of the sun in radians.
        sun_azimuth: Azimuth of the sun in radians.
        signed_offset: Offset along the meridian in radians, positive towards the
            horizon below the sun and negative towards the anti-solar half.

    Returns:
        Complex stereographic coordinate of the offset direction.
    """
    return (tan((sun_zenith_angle + signed_offset) / 2) * exp(1j * sun_azimuth)).astype(complex128)


def omega_from_roots(
    observed_point_projection: ComplexArray,
    roots: tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray],
) -> ComplexArray:
    """Compute the quartic polarization field from four explicit roots.

    Generalizes :func:`omega` by taking all four zeros directly instead of deriving
    the anti-solar pair from the solar pair. Roots are ordered below-sun (Brewster),
    above-sun (Babinet), above-anti-sun (Arago) and below-anti-sun (fourth).

    Args:
        observed_point_projection: Stereographic coordinate of the observed point.
        roots: The four zeros of the field, in the order described above.

    Returns:
        Complex polarization field at the observed point.
    """
    below_sun, above_sun, above_anti_sun, below_anti_sun = roots

    numerator: ComplexArray = (
        -4
        * (observed_point_projection - below_sun)
        * (observed_point_projection - above_sun)
        * (observed_point_projection - above_anti_sun)
        * (observed_point_projection - below_anti_sun)
    )

    denominator: FloatArray = (
        ((1 + absolute(observed_point_projection) ** 2) ** 2)
        * absolute(below_sun - above_anti_sun)
        * absolute(above_sun - below_anti_sun)
    )

    return numerator / denominator


def half_angle_modulus(
    observed_point_zenith_angle: FloatArray,
    observed_point_azimuth: FloatArray,
    signed_offsets: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    sun_zenith_angle: FloatArray,
    sun_azimuth: FloatArray,
) -> FloatArray:
    """Return the unnormalized modulus ``prod_k sin(Gamma_k / 2)``.

    Berry's Eq. (4.2) equals ``4 * prod_k sin(Gamma_k / 2)``, where ``Gamma_k`` is the
    angular distance from the observed direction to root ``k``. The identity follows
    from ``abs(xi - xi_k)^2 / [(1 + abs(xi)^2)(1 + abs(xi_k)^2)] = sin^2(Gamma_k / 2)``
    and, unlike Eq. (4.2), it assumes nothing about antipodal pairing.

    Do not substitute the full-angle product ``prod_k abs(sin(Gamma_k))``: because
    ``sin(pi - Gamma) = sin(Gamma)`` it is unconditionally antipodally invariant and
    introduces four spurious zeros at the antipodes of the roots whenever the roots
    are asymmetric.

    Args:
        observed_point_zenith_angle: Zenith angle of the observed point in radians.
        observed_point_azimuth: Azimuth of the observed point in radians.
        signed_offsets: Signed meridian offsets of the four roots, in radians.
        sun_zenith_angle: Zenith angle of the sun in radians.
        sun_azimuth: Azimuth of the sun in radians.

    Returns:
        The product of ``sin(Gamma_k / 2)`` over the four roots.
    """
    azimuth_difference: FloatArray = cos(observed_point_azimuth - sun_azimuth)
    modulus: FloatArray = ones_like(observed_point_zenith_angle)

    for signed_offset in signed_offsets:
        meridian_angle: FloatArray = sun_zenith_angle + signed_offset
        cosine_separation: FloatArray = (
            cos(observed_point_zenith_angle) * cos(meridian_angle)
            + sin(observed_point_zenith_angle) * sin(meridian_angle) * azimuth_difference
        )
        modulus = modulus * sqrt(clip((1 - cosine_separation) / 2, 0.0, None))

    return modulus


def omega(
    observed_point_projection: ComplexArray,
    below_sun_projection: ComplexArray,
    above_sun_projection: ComplexArray,
) -> ComplexArray:
    """Compute Berry's normalized complex polarization field.

    Implements Berry 2004, Eq. (4.2)::

        omega(xi) = -4 (xi - xi_+)(xi - xi_-)(xi + 1/conj(xi_+))(xi + 1/conj(xi_-))
                    / [ (1 + |xi|^2)^2 |xi_+ + 1/conj(xi_+)| |xi_- + 1/conj(xi_-)| ]

    where ``xi_+`` lies below and ``xi_-`` above the sun. The two anti-solar roots
    are the reciprocal-conjugate antipodes of the solar-side roots and are built
    here rather than supplied, so the pair can never be typo'd apart. The
    denominator makes ``|omega|`` antipodally invariant and normalizes its maximum
    to unity.

    The leading ``-4`` is part of the definition and is not cosmetic: ``-1`` is
    ``exp(i * pi)``, so dropping it rotates every angle of polarization derived
    from ``omega`` by exactly 90 degrees.

    Args:
        observed_point_projection: Stereographic coordinate of the observed point.
        below_sun_projection: Stereographic coordinate of the below-sun root
            (the Brewster point).
        above_sun_projection: Stereographic coordinate of the above-sun root
            (the Babinet point).

    Returns:
        Complex polarization field at the observed point.
    """
    return omega_from_roots(
        observed_point_projection,
        (
            below_sun_projection,
            above_sun_projection,
            antipode(below_sun_projection),
            antipode(above_sun_projection),
        ),
    )
