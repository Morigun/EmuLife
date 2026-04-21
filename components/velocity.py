from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Velocity(Component):
    dx: float = 0.0
    dy: float = 0.0
