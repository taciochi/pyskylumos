"""Shared NumPy and callable type aliases."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type SensorArray = NDArray[np.float32]
type RealArray = NDArray[np.floating[Any]]
type ComplexArray = NDArray[np.complex128]
type BoolArray = NDArray[np.bool_]
type ProjectionHook = Callable[..., FloatArray]
type ParameterNames = tuple[str, ...]

__all__ = (
    "BoolArray",
    "ComplexArray",
    "FloatArray",
    "ParameterNames",
    "ProjectionHook",
    "RealArray",
    "SensorArray",
)
