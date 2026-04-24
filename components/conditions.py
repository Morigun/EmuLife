from dataclasses import dataclass, field

from core.ecs import Component


@dataclass
class Condition:
    name: str
    duration: int
    speed_mult: float = 1.0
    metabolism_mult: float = 1.0
    efficiency_mult: float = 1.0


@dataclass
class Conditions(Component):
    effects: list[Condition] = field(default_factory=list)
