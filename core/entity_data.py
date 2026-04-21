from __future__ import annotations

import numpy as np

from components.diet import DietType


_DIET_TYPE_TO_INT = {
    DietType.HERBIVORE: 0,
    DietType.OMNIVORE: 1,
    DietType.PREDATOR: 2,
}

INT_TO_DIET_TYPE = {v: k for k, v in _DIET_TYPE_TO_INT.items()}


class EntityData:
    MAX_ENTITIES = 15000

    def __init__(self) -> None:
        self.x = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.y = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.dx = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.dy = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.energy = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.max_energy = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.health = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.max_health = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.age = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        self.max_age = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        self.metabolism = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.size_gene = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.speed_gene = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.vision = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.diet_type = np.zeros(self.MAX_ENTITIES, dtype=np.int8)
        self.efficiency = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.repro_threshold = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.repro_cooldown = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        self.repro_max_cooldown = np.zeros(self.MAX_ENTITIES, dtype=np.int32)
        self.aggression = np.zeros(self.MAX_ENTITIES, dtype=np.float32)
        self.alive = np.zeros(self.MAX_ENTITIES, dtype=np.bool_)
        self.r = np.zeros(self.MAX_ENTITIES, dtype=np.uint8)
        self.g = np.zeros(self.MAX_ENTITIES, dtype=np.uint8)
        self.b = np.zeros(self.MAX_ENTITIES, dtype=np.uint8)
        self.repro_type = np.zeros(self.MAX_ENTITIES, dtype=np.int8)
        self.habitat = np.zeros(self.MAX_ENTITIES, dtype=np.int8)
        self.eids = np.full(self.MAX_ENTITIES, -1, dtype=np.int32)

        self.count: int = 0
        self.eid_to_idx: dict[int, int] = {}
        self.idx_to_eid: dict[int, int] = {}
        self._free: list[int] = []

    def add(self, eid: int, genome, x: float, y: float, config, energy_fraction: float = 0.2, parent_energy_sum: float = 200.0) -> int:
        if self._free:
            idx = self._free.pop()
        else:
            idx = self.count
            self.count += 1

        size_val = 3.0 + genome.size * 7.0
        max_speed = 1.0 + genome.speed * 4.0
        vision = 20.0 + genome.vision * 80.0
        metab_rate = 0.5 + genome.metabolism_gene * 2.0
        diet = _diet_type_to_int(genome.diet_type_value)
        repro_thresh = 0.5 + genome.repro_threshold_gene * 0.4
        aggression = genome.aggression
        repro_type_val = 0 if genome.reproduction_type < 0.4 else 1
        habitat_val = _habitat_gene_to_int(genome.habitat)

        max_energy = 50.0 + size_val * 10.0
        max_health = 30.0 + size_val * 10.0
        max_age_val = int(800 + size_val * 200)
        child_energy = max_energy * energy_fraction

        self.x[idx] = x
        self.y[idx] = y
        self.dx[idx] = 0.0
        self.dy[idx] = 0.0
        self.energy[idx] = child_energy
        self.max_energy[idx] = max_energy
        self.health[idx] = max_health
        self.max_health[idx] = max_health
        self.age[idx] = 0
        self.max_age[idx] = max_age_val
        self.metabolism[idx] = metab_rate
        self.size_gene[idx] = size_val
        self.speed_gene[idx] = max_speed
        self.vision[idx] = vision
        self.diet_type[idx] = diet
        self.efficiency[idx] = 0.8 + aggression * 0.2
        self.repro_threshold[idx] = repro_thresh
        if repro_type_val == 0:
            cd = config.reproduction.asexual_cooldown
        else:
            cd = config.reproduction.sexual_cooldown
        self.repro_cooldown[idx] = cd
        self.repro_max_cooldown[idx] = cd
        self.aggression[idx] = aggression
        self.alive[idx] = True
        self.r[idx] = int(genome.r_color * 255)
        self.g[idx] = int(genome.g_color * 255)
        self.b[idx] = int(genome.b_color * 255)
        self.repro_type[idx] = repro_type_val
        self.habitat[idx] = habitat_val
        self.eids[idx] = eid

        self.eid_to_idx[eid] = idx
        self.idx_to_eid[idx] = eid
        return idx

    def remove(self, eid: int) -> None:
        idx = self.eid_to_idx.pop(eid, None)
        if idx is not None:
            self.alive[idx] = False
            self.eids[idx] = -1
            self._free.append(idx)
            del self.idx_to_eid[idx]

    def sync_from_ecs(self, em) -> None:
        from components.position import Position
        from components.velocity import Velocity
        from components.energy import Energy
        from components.health import Health
        from components.age import Age
        from components.genome_comp import GenomeComp
        from components.metabolism import Metabolism
        from components.diet import Diet
        from components.reproduction import Reproduction
        from components.sensor import Sensor

        for eid in em.get_entities_with(Position, Energy):
            pos = em.get_component(eid, Position)
            energy = em.get_component(eid, Energy)
            if pos is None or energy is None:
                continue

            vel = em.get_component(eid, Velocity)
            health = em.get_component(eid, Health)
            age_comp = em.get_component(eid, Age)
            genome_comp = em.get_component(eid, GenomeComp)
            metab = em.get_component(eid, Metabolism)
            diet = em.get_component(eid, Diet)
            repro = em.get_component(eid, Reproduction)

            if self._free:
                idx = self._free.pop()
            else:
                idx = self.count
                self.count += 1

            self.x[idx] = pos.x
            self.y[idx] = pos.y
            self.dx[idx] = vel.dx if vel else 0.0
            self.dy[idx] = vel.dy if vel else 0.0
            self.energy[idx] = energy.current
            self.max_energy[idx] = energy.max_value
            self.health[idx] = health.current if health else 100.0
            self.max_health[idx] = health.max_value if health else 100.0
            self.age[idx] = age_comp.current if age_comp else 0
            self.max_age[idx] = age_comp.max_age if age_comp else 1000

            if genome_comp and genome_comp.genome:
                g = genome_comp.genome
                self.metabolism[idx] = metab.rate if metab else 1.0
                self.size_gene[idx] = 3.0 + g.size * 7.0
                self.speed_gene[idx] = 1.0 + g.speed * 4.0
                sensor = em.get_component(eid, Sensor)
                self.vision[idx] = sensor.radius if sensor else 50.0
                self.r[idx] = int(g.r_color * 255)
                self.g[idx] = int(g.g_color * 255)
                self.b[idx] = int(g.b_color * 255)
                self.aggression[idx] = g.aggression
            else:
                self.metabolism[idx] = 1.0
                self.size_gene[idx] = 5.0
                self.speed_gene[idx] = 3.0
                self.vision[idx] = 50.0
                self.aggression[idx] = 0.5

            if diet:
                self.diet_type[idx] = _DIET_TYPE_TO_INT[diet.diet_type]
                self.efficiency[idx] = diet.efficiency
            else:
                self.diet_type[idx] = 0
                self.efficiency[idx] = 1.0

            if repro:
                self.repro_threshold[idx] = repro.threshold
                self.repro_cooldown[idx] = repro.cooldown
                self.repro_max_cooldown[idx] = repro.max_cooldown
                self.repro_type[idx] = 0 if repro.repro_type == "asexual" else 1
            else:
                self.repro_threshold[idx] = 0.7
                self.repro_cooldown[idx] = 0
                self.repro_max_cooldown[idx] = 150
                self.repro_type[idx] = 1

            self.habitat[idx] = 1

            self.alive[idx] = True
            self.eids[idx] = eid
            self.eid_to_idx[eid] = idx
            self.idx_to_eid[idx] = eid


def _diet_type_to_int(value: float) -> int:
    if value < 0.33:
        return 0
    elif value < 0.66:
        return 1
    else:
        return 2


def _habitat_gene_to_int(value: float) -> int:
    if value < 0.33:
        return 0
    elif value < 0.66:
        return 1
    else:
        return 2
