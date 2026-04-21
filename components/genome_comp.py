from dataclasses import dataclass
from core.ecs import Component
from core.genome import Genome


@dataclass
class GenomeComp(Component):
    genome: Genome = None
