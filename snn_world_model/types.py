"""Small shared data types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Action:
    left: float
    right: float

    def smoothness_from(self, previous: "Action") -> float:
        return abs(self.left - previous.left) + abs(self.right - previous.right)

    def effort(self) -> float:
        return abs(self.left) + abs(self.right)
