from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Age(Component):
    current: int = 0
    max_age: int = 1000
