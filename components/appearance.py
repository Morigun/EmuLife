from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Appearance(Component):
    r: int = 128
    g: int = 128
    b: int = 128
    size: float = 5.0
    shape: str = "circle"
