"""Data structure describing slicing patterns for sensor orientations."""

from dataclasses import dataclass


@dataclass
class SlicingPattern:
    """Define a slicing pattern for selecting pixels by orientation."""

    start_row: int
    start_column: int
    step: int
