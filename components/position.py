from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Position(Component):
    x: float = 0.0
    y: float = 0.0
