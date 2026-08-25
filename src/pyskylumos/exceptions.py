"""Public exception hierarchy for stable validation contracts."""


class PySkyLumosError(Exception):
    """Base class for package-defined errors."""


class ConfigurationError(PySkyLumosError, ValueError):
    """Raised when an object is configured with an invalid value."""


class InputValidationError(PySkyLumosError, ValueError):
    """Raised when runtime input has an invalid value, rank, or shape."""


class InputTypeError(PySkyLumosError, TypeError):
    """Raised when public input has the wrong Python type."""


__all__ = (
    "ConfigurationError",
    "InputTypeError",
    "InputValidationError",
    "PySkyLumosError",
)
