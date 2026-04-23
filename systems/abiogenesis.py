from __future__ import annotations

import random

import numpy as np

from core.ecs import System, EntityManager
from core.world import World
from core.genome import Genome
from core.entity_data import EntityData
from utils.spatial_hash import SpatialHash
from utils.numba_kernels import compute_stats_kernel, HAS_NUMBA
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

        ed = self.entity_data
        pred_count = 0
        if ed is not None:
            n = ed.count
            if HAS_NUMBA:
                _, _, pred_count = compute_stats_kernel(ed.diet_type, ed.alive, n)
            else:
                pred_count = int(np.sum(ed.alive[:n] & (ed.diet_type[:n] == 2)))

            if pred_count <= 2 and random.random() < 0.005:
                for attempt in range(50):
                    rand_idx = random.randint(0, n - 1)
                    if ed.alive[rand_idx] and ed.diet_type[rand_idx] != 2:
                        lx = float(ed.x[rand_idx]) + random.uniform(-30, 30)
                        ly = float(ed.y[rand_idx]) + random.uniform(-30, 30)
                        lx = max(0.0, min(float(w.width - 1), lx))
                        ly = max(0.0, min(float(w.height - 1), ly))
                        if w.walkable_mask[int(ly), int(lx)]:
                            self._spawn_organism(w, lx, ly, True, force_predator=True)
                            self.lightning_events.append((int(lx), int(ly)))
                            break

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

    def _spawn_organism(self, w: World, x: float, y: float, walkable: bool, force_predator: bool = False) -> None:
        from systems.reproduction import create_organism

        genome = Genome.random_instance(self.config)
        if not walkable:
            genome.genes[11] = random.random() * 0.33
        if force_predator:
            genome.genes[4] = 0.66 + random.random() * 0.34
            genome.genes[6] = max(0.4, genome.aggression)

        energy_fraction = 0.6 if force_predator else 0.3

        eid = create_organism(
            self.em, genome, x, y, self.config,
            energy_fraction=energy_fraction,
            parent_energy_sum=100.0,
            entity_data=self.entity_data,
        )
        self.spatial_hash.insert(eid, x, y)
