from __future__ import annotations

import random

import numpy as np

from core.ecs import System, EntityManager
from core.world import World
from core.genome import Genome
from core.entity_data import EntityData
from utils.spatial_hash import SpatialHash
from config import Config


class AbiogenesisSystem(System):
    def __init__(
        self,
        entity_manager: EntityManager,
        spatial_hash: SpatialHash,
        config: Config,
        entity_data: EntityData = None,
    ) -> None:
        self.em = entity_manager
        self.spatial_hash = spatial_hash
        self.config = config
        self.entity_data = entity_data
        self.lightning_events: list[tuple[int, int]] = []

    def update(self, world: object, dt: float) -> None:
        w: World = world
        ac = self.config.abiogenesis
        self.lightning_events.clear()

        if self.em.entity_count >= self.config.simulation.max_population:
            return

        biomass = w.biomass
        candidates = biomass > ac.biomass_threshold
        if np.any(candidates):
            ys, xs = np.where(candidates)
            probs = biomass[ys, xs] * ac.biomass_spawn_chance
            rolls = np.random.random(len(ys))
            spawned = rolls < probs
            for y, x in zip(ys[spawned], xs[spawned]):
                if self.em.entity_count >= self.config.simulation.max_population:
                    break
                self._spawn_organism(w, float(x), float(y), w.walkable_mask[y, x])

        if random.random() < ac.lightning_chance:
            for _ in range(100):
                lx = random.randint(0, w.width - 1)
                ly = random.randint(0, w.height - 1)
                if w.walkable_mask[ly, lx]:
                    r = ac.lightning_radius
                    y0 = max(0, ly - r)
                    y1 = min(w.height, ly + r + 1)
                    x0 = max(0, lx - r)
                    x1 = min(w.width, lx + r + 1)
                    w.biomass[y0:y1, x0:x1] += ac.lightning_biomass_boost
                    self._spawn_organism(w, float(lx), float(ly), True)
                    self.lightning_events.append((lx, ly))
                    break

    def _spawn_organism(self, w: World, x: float, y: float, walkable: bool) -> None:
        from systems.reproduction import create_organism

        genome = Genome.random_instance(self.config)
        if not walkable:
            genome.genes[11] = random.random() * 0.33

        eid = create_organism(
            self.em, genome, x, y, self.config,
            energy_fraction=0.3,
            parent_energy_sum=100.0,
            entity_data=self.entity_data,
        )
        self.spatial_hash.insert(eid, x, y)
