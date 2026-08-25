"""Data structure describing slicing patterns for sensor orientations."""

from dataclasses import dataclass

from pyskylumos._validation import require_integer


@dataclass(frozen=True)
class SlicingPattern:
    """Define a slicing pattern for selecting pixels by orientation."""

    start_row: int
    start_column: int
    step: int

    def __post_init__(self) -> None:
        """Validate integer, positive slicing coordinates at construction."""
        require_integer("start_row", self.start_row, minimum=0)
        require_integer("start_column", self.start_column, minimum=0)
        require_integer("step", self.step, minimum=1)
