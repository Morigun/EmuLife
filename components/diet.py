from dataclasses import dataclass
from enum import Enum
from core.ecs import Component


class DietType(Enum):
    HERBIVORE = "herbivore"
    OMNIVORE = "omnivore"
    PREDATOR = "predator"


@dataclass
class Diet(Component):
    diet_type: DietType = DietType.HERBIVORE
    efficiency: float = 1.0
