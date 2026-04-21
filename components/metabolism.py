from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Metabolism(Component):
    rate: float = 1.0
