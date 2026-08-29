"""Shared helpers for public input validation and deprecated aliases."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Final
from warnings import warn

import numpy as np

from pyskylumos.exceptions import ConfigurationError, InputTypeError, InputValidationError

UNSET: Final = object()

#: Absolute tolerance for the orthonormality and determinant checks applied to a
#: supplied rotation matrix. It is a validation bound only; it is never used to
#: collapse a genuine near-identity rotation onto the no-rotation path.
_ROTATION_TOLERANCE: Final = 1e-6


def resolve_deprecated_alias(
    *,
    preferred_name: str,
    preferred_value: Any,
    legacy_name: str,
    legacy_value: Any,
    default: Any = UNSET,
) -> Any:
    """Resolve one preferred/legacy argument pair.

    Args:
        preferred_name: Current name of the argument.
        preferred_value: Value passed under the current name, or ``UNSET``.
        legacy_name: Deprecated name of the same argument.
        legacy_value: Value passed under the deprecated name, or ``UNSET``.
        default: Value to return when neither name was supplied. Leaving it
            ``UNSET`` makes the argument required, which is the behaviour the
            renamed constructor arguments rely on.

    Returns:
        The resolved value under the current name.

    Raises:
        ConfigurationError: If both names are supplied, or if neither is
            supplied and the argument is required.
    """
    if preferred_value is not UNSET and legacy_value is not UNSET:
        raise ConfigurationError(
            f"Pass only {preferred_name!r}; it replaces deprecated {legacy_name!r}."
        )
    if legacy_value is not UNSET:
        warn(
            f"{legacy_name} is deprecated and will be removed in 0.2.0; "
            f"use {preferred_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy_value
    if preferred_value is not UNSET:
        return preferred_value
    if default is not UNSET:
        return default
    raise ConfigurationError(f"Missing required argument: {preferred_name}.")


def require_argument(name: str, value: Any) -> Any:
    """Return a required non-aliased argument or raise a targeted error."""
    if value is UNSET:
        raise ConfigurationError(f"Missing required argument: {name}.")
    return value


def require_real(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Validate and return a finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite real number; got {value!r}.")
    result = float(value)
    if not np.isfinite(result):
        raise ConfigurationError(f"{name} must be finite; got {value!r}.")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ConfigurationError(f"{name} must be {operator} {minimum}; got {value!r}.")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ConfigurationError(f"{name} must be {operator} {maximum}; got {value!r}.")
    return result


def require_integer(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate and return a non-Boolean integer scalar."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigurationError(f"{name} must be an integer; got {value!r}.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}; got {value!r}.")
    if maximum is not None and result > maximum:
        raise ConfigurationError(f"{name} must be <= {maximum}; got {value!r}.")
    return result


def require_real_array(name: str, value: Any, *, rank: int | None = 3) -> np.ndarray:
    """Validate a real NumPy array with the requested rank."""
    if not isinstance(value, np.ndarray):
        raise InputTypeError(f"{name} must be a numpy.ndarray; got {type(value).__name__}.")
    if rank is not None and value.ndim != rank:
        raise InputValidationError(f"{name} must have rank {rank}; got shape {value.shape}.")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.complexfloating):
        raise InputTypeError(f"{name} must contain real numeric values; got dtype {value.dtype}.")
    return value


def require_rotation_matrix(name: str, value: Any) -> np.ndarray:
    """Validate a proper ``(3, 3)`` floating-point rotation matrix.

    The matrix is never projected, normalised, or otherwise repaired; an input
    that fails any check is rejected rather than corrected.

    Args:
        name: Argument name used in error messages.
        value: Candidate rotation matrix.

    Returns:
        A fresh float64 copy of the validated matrix, so later mutation of the
        result cannot reach the caller's array.

    Raises:
        InputTypeError: If the value is not a NumPy array or does not have a
            real floating-point dtype.
        InputValidationError: If the shape is not ``(3, 3)``, the values are not
            all finite, or the matrix is not a proper rotation.
    """
    if not isinstance(value, np.ndarray):
        raise InputTypeError(f"{name} must be a numpy.ndarray; got {type(value).__name__}.")
    if not np.issubdtype(value.dtype, np.floating):
        raise InputTypeError(
            f"{name} must have a real floating-point dtype; got dtype {value.dtype}."
        )
    if value.shape != (3, 3):
        raise InputValidationError(f"{name} must have shape (3, 3); got shape {value.shape}.")
    if not np.isfinite(value).all():
        raise InputValidationError(f"{name} must contain only finite values.")

    matrix = np.array(value, dtype=np.float64)
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=_ROTATION_TOLERANCE):
        raise InputValidationError(
            f"{name} must be orthonormal; transpose(R) @ R is not the identity "
            f"within absolute tolerance {_ROTATION_TOLERANCE}."
        )
    if not np.allclose(np.linalg.det(matrix), 1.0, rtol=0.0, atol=_ROTATION_TOLERANCE):
        raise InputValidationError(
            f"{name} must be a proper rotation with determinant +1; got "
            f"{float(np.linalg.det(matrix))!r}."
        )
    return matrix


def require_same_shape(**arrays: np.ndarray) -> tuple[int, ...]:
    """Require all named arrays to have one identical shape."""
    shapes = {name: value.shape for name, value in arrays.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        details = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        raise InputValidationError(f"Input arrays must have identical shapes; got {details}.")
    return next(iter(unique))
