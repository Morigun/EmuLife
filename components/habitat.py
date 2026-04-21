from dataclasses import dataclass
from core.ecs import Component


@dataclass
class Habitat(Component):
    habitat_type: str = "terrestrial"
