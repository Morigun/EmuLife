from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Reproduction(Component):
    threshold: float = 0.7
    cooldown: int = 0
    max_cooldown: int = 60
    repro_type: str = "sexual"
