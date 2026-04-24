from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import Config


@dataclass
class Genome:
    genes: list[float] = field(default_factory=list)

    @staticmethod
    def random_instance(config: Config | None = None) -> Genome:
        length = config.genome.genome_length if config else 16
        genes = [random.random() for _ in range(length)]
        return Genome(genes)

    @staticmethod
    def crossover(parent1: Genome, parent2: Genome, config: Config | None = None) -> Genome:
        length = min(len(parent1.genes), len(parent2.genes))
        breakpoint = random.randint(0, length)
        child_genes = parent1.genes[:breakpoint] + parent2.genes[breakpoint:length]
        return Genome(child_genes)

    def mutate(self, config: Config | None = None) -> Genome:
        rate = config.genome.mutation_rate if config else 0.1
        strength = config.genome.mutation_strength if config else 0.1
        new_genes = []
        for g in self.genes:
            if random.random() < rate:
                g = g + random.uniform(-strength, strength)
                g = max(0.0, min(1.0, g))
            new_genes.append(g)
        return Genome(new_genes)

    def get_gene(self, index: int, default: float = 0.0) -> float:
        if 0 <= index < len(self.genes):
            return self.genes[index]
        return default

    @property
    def size(self) -> float:
        return self.get_gene(0)

    @property
    def speed(self) -> float:
        return self.get_gene(1)

    @property
    def vision(self) -> float:
        return self.get_gene(2)

    @property
    def metabolism_gene(self) -> float:
        return self.get_gene(3)

    @property
    def diet_type_value(self) -> float:
        return self.get_gene(4)

    @property
    def repro_threshold_gene(self) -> float:
        return self.get_gene(5)

    @property
    def aggression(self) -> float:
        return self.get_gene(6)

    @property
    def r_color(self) -> float:
        return self.get_gene(7)

    @property
    def g_color(self) -> float:
        return self.get_gene(8)

    @property
    def b_color(self) -> float:
        return self.get_gene(9)

    @property
    def reproduction_type(self) -> float:
        return self.get_gene(10)

    @property
    def habitat(self) -> float:
        return self.get_gene(11)

    @property
    def photosynth(self) -> float:
        return self.get_gene(13)
