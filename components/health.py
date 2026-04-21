from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Health(Component):
    current: float = 100.0
    max_value: float = 100.0
