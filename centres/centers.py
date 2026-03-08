from dataclasses import dataclass


@dataclass
class Center:
    id: int
    x: float
    y: float
    scale: float
    strength: float
    parent: int = None
