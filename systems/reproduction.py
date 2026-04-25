from __future__ import annotations

import random

from core.ecs import System, EntityManager
from components.position import Position
from components.velocity import Velocity
from components.energy import Energy
from components.health import Health
from components.age import Age
from components.genome_comp import GenomeComp
from components.appearance import Appearance
from components.sensor import Sensor
from components.metabolism import Metabolism
from components.diet import Diet, DietType
from components.reproduction import Reproduction
from core.genome import Genome
from core.world import World
from utils.spatial_hash import SpatialHash
from config import Config


def diet_type_from_gene(value: float) -> DietType:
    if value < 0.33:
        return DietType.HERBIVORE
    elif value < 0.66:
        return DietType.OMNIVORE
    elif value < 0.95:
        return DietType.PREDATOR
    else:
        return DietType.CARNIVOROUS_PLANT


def habitat_type_from_gene(value: float) -> str:
    if value < 0.33:
        return "aquatic"
    elif value < 0.66:
        return "terrestrial"
    else:
        return "amphibious"


def repro_type_from_gene(value: float, is_predator: bool = False) -> str:
    if is_predator:
        if value < 0.50:
            return "asexual"
        elif value < 0.90:
            return "hermaphrodite"
        else:
            return "sexual"
    else:
        if value < 0.30:
            return "asexual"
        elif value < 0.70:
            return "hermaphrodite"
        else:
            return "sexual"


def create_organism(
    em: EntityManager,
    genome: Genome,
    x: float,
    y: float,
    config: Config,
    energy_fraction: float = 0.2,
    parent_energy_sum: float = 200.0,
    entity_data=None,
    origin: int = 0,
    parent_eid: int = -1,
) -> int:
    from components.habitat import Habitat

    size_val = 3.0 + genome.size * 7.0
    max_speed = 1.0 + genome.speed * 4.0
    vision = 20.0 + genome.vision * 80.0
    metab_rate = 0.5 + genome.metabolism_gene * 2.0
    diet = diet_type_from_gene(genome.diet_type_value)
    repro_thresh = 0.5 + genome.repro_threshold_gene * 0.4
    aggression = genome.aggression
    repro_type = repro_type_from_gene(genome.reproduction_type, diet == DietType.PREDATOR)
    if diet == DietType.CARNIVOROUS_PLANT:
        repro_type = "asexual"
    habitat_type = habitat_type_from_gene(genome.habitat)

    max_energy = 50.0 + size_val * 10.0
    max_health = 30.0 + size_val * 10.0
    max_age_val = int(800 + size_val * 200)

    if diet == DietType.PREDATOR:
        child_energy = max_energy * config.energy.predator_child_energy_fraction
    else:
        child_energy = max_energy * energy_fraction

    if repro_type == "asexual":
        shape = "diamond"
    elif repro_type == "hermaphrodite":
        shape = "pentagon"
    elif diet == DietType.PREDATOR:
        shape = "triangle"
    elif diet == DietType.CARNIVOROUS_PLANT:
        shape = "hexagon"
    else:
        shape = "circle"

    if repro_type in ("sexual", "hermaphrodite"):
        cd = config.reproduction.sexual_cooldown
    elif diet == DietType.PREDATOR:
        cd = 120
    else:
        cd = config.reproduction.asexual_cooldown

    eid = em.create_entity(
        Position(x=x, y=y),
        Velocity(dx=0.0, dy=0.0),
        Energy(current=child_energy, max_value=max_energy),
        Health(current=max_health, max_value=max_health),
        Age(current=0, max_age=max_age_val),
        GenomeComp(genome=genome),
        Appearance(
            r=int(genome.r_color * 255),
            g=int(genome.g_color * 255),
            b=int(genome.b_color * 255),
            size=size_val,
            shape=shape,
        ),
        Sensor(radius=vision),
        Metabolism(rate=metab_rate),
        Diet(diet_type=diet, efficiency=0.8 + aggression * 0.2),
        Reproduction(threshold=repro_thresh, cooldown=cd, max_cooldown=cd, repro_type=repro_type),
        Habitat(habitat_type=habitat_type),
    )

    from components.conditions import Conditions
    em.add_component(eid, Conditions())

    if entity_data is not None:
        entity_data.add(eid, genome, x, y, config, energy_fraction, parent_energy_sum=parent_energy_sum, origin=origin, parent_eid=parent_eid)

    return eid


class ReproductionSystem(System):
    def __init__(self, entity_manager: EntityManager, spatial_hash: SpatialHash, config: Config, entity_data=None) -> None:
        self.em = entity_manager
        self.spatial_hash = spatial_hash
        self.config = config
        self.entity_data = entity_data
        self.newborn_entities: list[tuple[int, float, float]] = []
        self.births_asexual_tick: int = 0
        self.births_sexual_tick: int = 0
        self._nearby_set: set[int] = set()
        self._density_set: set[int] = set()

    def update(self, world: object, dt: float) -> None:
        w: World = world
        self.newborn_entities.clear()
        self.births_asexual_tick = 0
        self.births_sexual_tick = 0
        newborn_count = 0

        if self.entity_data is not None:
            self._update_soa(w, newborn_count)
        else:
            self._update_ecs(w, newborn_count)

    def _update_soa(self, w: World, newborn_count: int) -> None:
        import numpy as np
        ed = self.entity_data
        n = ed.count
        s = slice(0, n)

        alive = ed.alive[s]
        has_cooldown = ed.repro_cooldown[s] > 0
        decrement_mask = alive & has_cooldown
        ed.repro_cooldown[s] -= np.where(decrement_mask, 1, 0).astype(np.int32)

        ready_mask = alive & ~has_cooldown
        ready_indices = np.where(ready_mask)[0]

        max_births = self.config.reproduction.max_births_per_tick
        density_radius = self.config.reproduction.density_radius
        density_max = self.config.reproduction.density_max_neighbors
        _density = self._density_set

        for idx in ready_indices:
            idx_int = int(idx)
            eid = ed.idx_to_eid.get(idx_int)
            if eid is None:
                continue

            if newborn_count >= max_births:
                break

            if self.em.entity_count + newborn_count >= self.config.simulation.max_population:
                break

            energy_val = float(ed.energy[idx_int])
            max_energy_val = float(ed.max_energy[idx_int])
            repro_thresh = float(ed.repro_threshold[idx_int])

            if energy_val < repro_thresh * max_energy_val:
                continue

            px = float(ed.x[idx_int])
            py = float(ed.y[idx_int])
            _density.clear()
            self.spatial_hash.query_nearby_into(px, py, density_radius, _density)
            if len(_density) > density_max:
                continue

            pos = self.em.get_component(eid, Position)
            genome_comp = self.em.get_component(eid, GenomeComp)
            diet = self.em.get_component(eid, Diet)
            if pos is None or genome_comp is None or diet is None:
                continue

            repro = self.em.get_component(eid, Reproduction)
            repro_type_int = int(ed.repro_type[idx_int])

            if repro_type_int == 0:
                child_genome = genome_comp.genome.mutate(self.config)
                parent_sum = energy_val
                spread = self.config.carnivorous_plant.spawn_spread if int(ed.diet_type[idx_int]) == 3 else 5.0
                cx = pos.x + random.uniform(-spread, spread)
                cy = pos.y + random.uniform(-spread, spread)
                cx = max(0.0, min(float(w.width - 1), cx))
                cy = max(0.0, min(float(w.height - 1), cy))

                cost = self.config.energy.asexual_reproduction_energy_cost
                new_energy = energy_val * (1.0 - cost)
                ed.energy[idx_int] = new_energy
                energy_comp = self.em.get_component(eid, Energy)
                if energy_comp:
                    energy_comp.current = new_energy

                if int(ed.diet_type[idx_int]) == 2:
                    cd = 120
                else:
                    cd = self.config.reproduction.asexual_cooldown
                ed.repro_cooldown[idx_int] = cd
                ed.repro_max_cooldown[idx_int] = cd
                if repro:
                    repro.cooldown = cd
                    repro.max_cooldown = cd

                child_frac = self.config.energy.asexual_child_energy_fraction
                new_eid = create_organism(self.em, child_genome, cx, cy, self.config, child_frac, parent_sum, self.entity_data, origin=1, parent_eid=eid)
                self.newborn_entities.append((new_eid, cx, cy))
                self.births_asexual_tick += 1
                newborn_count += 1

            elif repro_type_int == 2:
                did_sexual = False
                partner_id = self._find_partner(eid, pos, diet.diet_type)
                if partner_id is not None:
                    partner_genome_comp = self.em.get_component(partner_id, GenomeComp)
                    if partner_genome_comp and partner_genome_comp.genome:
                        did_sexual = True
                        child_genome = Genome.crossover(genome_comp.genome, partner_genome_comp.genome, self.config)
                        child_genome = child_genome.mutate(self.config)
                        partner_energy = self.em.get_component(partner_id, Energy)
                        parent_sum = energy_val + (partner_energy.current if partner_energy else 0)
                        spread = self.config.carnivorous_plant.spawn_spread if int(ed.diet_type[idx_int]) == 3 else 5.0
                        cx = pos.x + random.uniform(-spread, spread)
                        cy = pos.y + random.uniform(-spread, spread)
                        cx = max(0.0, min(float(w.width - 1), cx))
                        cy = max(0.0, min(float(w.height - 1), cy))

                        new_energy = energy_val * (1.0 - self.config.energy.reproduction_energy_cost)
                        ed.energy[idx_int] = new_energy
                        energy_comp = self.em.get_component(eid, Energy)
                        if energy_comp:
                            energy_comp.current = new_energy

                        if partner_energy:
                            partner_new = partner_energy.current * (1.0 - self.config.energy.reproduction_energy_cost)
                            partner_energy.current = partner_new
                            p_idx = ed.eid_to_idx.get(partner_id)
                            if p_idx is not None:
                                ed.energy[p_idx] = partner_new

                        cd = self.config.reproduction.sexual_cooldown
                        ed.repro_cooldown[idx_int] = cd
                        ed.repro_max_cooldown[idx_int] = cd
                        if repro:
                            repro.cooldown = cd
                            repro.max_cooldown = cd
                        partner_repro = self.em.get_component(partner_id, Reproduction)
                        if partner_repro:
                            partner_repro.cooldown = partner_repro.max_cooldown
                            p_idx = ed.eid_to_idx.get(partner_id)
                            if p_idx is not None:
                                ed.repro_cooldown[p_idx] = partner_repro.max_cooldown

                        new_eid = create_organism(self.em, child_genome, cx, cy, self.config, self.config.energy.child_energy_fraction, parent_sum, self.entity_data, origin=2, parent_eid=eid)
                        self.newborn_entities.append((new_eid, cx, cy))
                        self.births_sexual_tick += 1
                        newborn_count += 1

                if not did_sexual:
                    child_genome = genome_comp.genome.mutate(self.config)
                    parent_sum = energy_val
                    spread = self.config.carnivorous_plant.spawn_spread if int(ed.diet_type[idx_int]) == 3 else 5.0
                    cx = pos.x + random.uniform(-spread, spread)
                    cy = pos.y + random.uniform(-spread, spread)
                    cx = max(0.0, min(float(w.width - 1), cx))
                    cy = max(0.0, min(float(w.height - 1), cy))

                    cost = self.config.energy.asexual_reproduction_energy_cost
                    new_energy = energy_val * (1.0 - cost)
                    ed.energy[idx_int] = new_energy
                    energy_comp = self.em.get_component(eid, Energy)
                    if energy_comp:
                        energy_comp.current = new_energy

                    cd = self.config.reproduction.asexual_cooldown
                    ed.repro_cooldown[idx_int] = cd
                    ed.repro_max_cooldown[idx_int] = cd
                    if repro:
                        repro.cooldown = cd
                        repro.max_cooldown = cd

                    child_frac = self.config.energy.asexual_child_energy_fraction
                    new_eid = create_organism(self.em, child_genome, cx, cy, self.config, child_frac, parent_sum, self.entity_data, origin=1, parent_eid=eid)
                    self.newborn_entities.append((new_eid, cx, cy))
                    self.births_asexual_tick += 1
                    newborn_count += 1

            else:
                partner_id = self._find_partner(eid, pos, diet.diet_type)
                if partner_id is None:
                    if random.random() < 0.01:
                        child_genome = genome_comp.genome.mutate(self.config)
                        parent_sum = energy_val
                        cx = pos.x + random.uniform(-5.0, 5.0)
                        cy = pos.y + random.uniform(-5.0, 5.0)
                        cx = max(0.0, min(float(w.width - 1), cx))
                        cy = max(0.0, min(float(w.height - 1), cy))
                        cost = self.config.energy.asexual_reproduction_energy_cost
                        new_energy = energy_val * (1.0 - cost)
                        ed.energy[idx_int] = new_energy
                        energy_comp = self.em.get_component(eid, Energy)
                        if energy_comp:
                            energy_comp.current = new_energy
                        cd = self.config.reproduction.sexual_cooldown
                        ed.repro_cooldown[idx_int] = cd
                        ed.repro_max_cooldown[idx_int] = cd
                        if repro:
                            repro.cooldown = cd
                            repro.max_cooldown = cd
                        child_frac = self.config.energy.asexual_child_energy_fraction
                        new_eid = create_organism(self.em, child_genome, cx, cy, self.config, child_frac, parent_sum, self.entity_data, origin=1, parent_eid=eid)
                        self.newborn_entities.append((new_eid, cx, cy))
                        self.births_asexual_tick += 1
                        newborn_count += 1
                    continue
                partner_genome_comp = self.em.get_component(partner_id, GenomeComp)
                if partner_genome_comp and partner_genome_comp.genome:
                    child_genome = Genome.crossover(genome_comp.genome, partner_genome_comp.genome, self.config)
                    child_genome = child_genome.mutate(self.config)
                    partner_energy = self.em.get_component(partner_id, Energy)
                    parent_sum = energy_val + (partner_energy.current if partner_energy else 0)
                    spread = self.config.carnivorous_plant.spawn_spread if int(ed.diet_type[idx_int]) == 3 else 5.0
                    cx = pos.x + random.uniform(-spread, spread)
                    cy = pos.y + random.uniform(-spread, spread)
                    cx = max(0.0, min(float(w.width - 1), cx))
                    cy = max(0.0, min(float(w.height - 1), cy))

                    new_energy = energy_val * (1.0 - self.config.energy.reproduction_energy_cost)
                    ed.energy[idx_int] = new_energy
                    energy_comp = self.em.get_component(eid, Energy)
                    if energy_comp:
                        energy_comp.current = new_energy

                    if partner_energy:
                        partner_new = partner_energy.current * (1.0 - self.config.energy.reproduction_energy_cost)
                        partner_energy.current = partner_new
                        p_idx = ed.eid_to_idx.get(partner_id)
                        if p_idx is not None:
                            ed.energy[p_idx] = partner_new

                    cd = self.config.reproduction.sexual_cooldown
                    ed.repro_cooldown[idx_int] = cd
                    ed.repro_max_cooldown[idx_int] = cd
                    if repro:
                        repro.cooldown = cd
                        repro.max_cooldown = cd
                    partner_repro = self.em.get_component(partner_id, Reproduction)
                    if partner_repro:
                        partner_repro.cooldown = partner_repro.max_cooldown
                        p_idx = ed.eid_to_idx.get(partner_id)
                        if p_idx is not None:
                            ed.repro_cooldown[p_idx] = partner_repro.max_cooldown

                    new_eid = create_organism(self.em, child_genome, cx, cy, self.config, self.config.energy.child_energy_fraction, parent_sum, self.entity_data, origin=2, parent_eid=eid)
                    self.newborn_entities.append((new_eid, cx, cy))
                    self.births_sexual_tick += 1
                    newborn_count += 1

    def _update_ecs(self, w: World, newborn_count: int) -> None:
        max_births = self.config.reproduction.max_births_per_tick
        density_radius = self.config.reproduction.density_radius
        density_max = self.config.reproduction.density_max_neighbors
        _density = self._density_set

        for eid in list(self.em.get_entities_with(Position, Energy, Reproduction, GenomeComp, Diet)):
            if newborn_count >= max_births:
                break

            if self.em.entity_count + newborn_count >= self.config.simulation.max_population:
                break

            pos = self.em.get_component(eid, Position)
            energy = self.em.get_component(eid, Energy)
            repro = self.em.get_component(eid, Reproduction)
            genome_comp = self.em.get_component(eid, GenomeComp)
            diet = self.em.get_component(eid, Diet)

            if pos is None or energy is None or repro is None or genome_comp is None or diet is None:
                continue

            if repro.cooldown > 0:
                repro.cooldown -= 1
                continue

            if energy.current < repro.threshold * energy.max_value:
                continue

            _density.clear()
            self.spatial_hash.query_nearby_into(pos.x, pos.y, density_radius, _density)
            if len(_density) > density_max:
                continue

            repro_type = repro.repro_type if repro else "asexual"

            if repro_type == "asexual":
                child_genome = genome_comp.genome.mutate(self.config)
                parent_sum = energy.current

                spread = self.config.carnivorous_plant.spawn_spread if diet.diet_type == DietType.CARNIVOROUS_PLANT else 5.0
                cx = pos.x + random.uniform(-spread, spread)
                cy = pos.y + random.uniform(-spread, spread)
                cx = max(0.0, min(float(w.width - 1), cx))
                cy = max(0.0, min(float(w.height - 1), cy))

                cost = self.config.energy.asexual_reproduction_energy_cost
                energy.current *= (1.0 - cost)
                if diet.diet_type == DietType.PREDATOR:
                    cd = 120
                else:
                    cd = self.config.reproduction.asexual_cooldown
                repro.cooldown = cd
                repro.max_cooldown = cd

                child_frac = self.config.energy.asexual_child_energy_fraction
                new_eid = create_organism(self.em, child_genome, cx, cy, self.config, child_frac, parent_sum, self.entity_data, origin=1, parent_eid=eid)
                self.newborn_entities.append((new_eid, cx, cy))
                self.births_asexual_tick += 1
                newborn_count += 1

            elif repro_type == "hermaphrodite":
                partner_id = self._find_partner(eid, pos, diet.diet_type)
                if partner_id is not None:
                    partner_genome_comp = self.em.get_component(partner_id, GenomeComp)
                    if partner_genome_comp and partner_genome_comp.genome:
                        child_genome = Genome.crossover(genome_comp.genome, partner_genome_comp.genome, self.config)
                        child_genome = child_genome.mutate(self.config)

                        partner_energy = self.em.get_component(partner_id, Energy)
                        parent_sum = energy.current + (partner_energy.current if partner_energy else 0)

                        spread = self.config.carnivorous_plant.spawn_spread if diet.diet_type == DietType.CARNIVOROUS_PLANT else 5.0
                        cx = pos.x + random.uniform(-spread, spread)
                        cy = pos.y + random.uniform(-spread, spread)
                        cx = max(0.0, min(float(w.width - 1), cx))
                        cy = max(0.0, min(float(w.height - 1), cy))

                        energy.current *= (1.0 - self.config.energy.reproduction_energy_cost)
                        if partner_energy:
                            partner_energy.current *= (1.0 - self.config.energy.reproduction_energy_cost)

                        cd = self.config.reproduction.sexual_cooldown
                        repro.cooldown = cd
                        repro.max_cooldown = cd
                        partner_repro = self.em.get_component(partner_id, Reproduction)
                        if partner_repro:
                            partner_repro.cooldown = partner_repro.max_cooldown

                        new_eid = create_organism(self.em, child_genome, cx, cy, self.config, self.config.energy.child_energy_fraction, parent_sum, self.entity_data, origin=2, parent_eid=eid)
                        self.newborn_entities.append((new_eid, cx, cy))
                        self.births_sexual_tick += 1
                        newborn_count += 1
                        continue

                child_genome = genome_comp.genome.mutate(self.config)
                parent_sum = energy.current

                spread = self.config.carnivorous_plant.spawn_spread if diet.diet_type == DietType.CARNIVOROUS_PLANT else 5.0
                cx = pos.x + random.uniform(-spread, spread)
                cy = pos.y + random.uniform(-spread, spread)
                cx = max(0.0, min(float(w.width - 1), cx))
                cy = max(0.0, min(float(w.height - 1), cy))

                cost = self.config.energy.asexual_reproduction_energy_cost
                energy.current *= (1.0 - cost)
                cd = self.config.reproduction.asexual_cooldown
                repro.cooldown = cd
                repro.max_cooldown = cd

                child_frac = self.config.energy.asexual_child_energy_fraction
                new_eid = create_organism(self.em, child_genome, cx, cy, self.config, child_frac, parent_sum, self.entity_data, origin=1, parent_eid=eid)
                self.newborn_entities.append((new_eid, cx, cy))
                self.births_asexual_tick += 1
                newborn_count += 1

            else:
                partner_id = self._find_partner(eid, pos, diet.diet_type)
                if partner_id is None:
                    if random.random() < 0.01:
                        child_genome = genome_comp.genome.mutate(self.config)
                        parent_sum = energy.current
                        spread = self.config.carnivorous_plant.spawn_spread if diet.diet_type == DietType.CARNIVOROUS_PLANT else 5.0
                        cx = pos.x + random.uniform(-spread, spread)
                        cy = pos.y + random.uniform(-spread, spread)
                        cx = max(0.0, min(float(w.width - 1), cx))
                        cy = max(0.0, min(float(w.height - 1), cy))
                        cost = self.config.energy.asexual_reproduction_energy_cost
                        energy.current *= (1.0 - cost)
                        cd = self.config.reproduction.sexual_cooldown
                        repro.cooldown = cd
                        repro.max_cooldown = cd
                        child_frac = self.config.energy.asexual_child_energy_fraction
                        new_eid = create_organism(self.em, child_genome, cx, cy, self.config, child_frac, parent_sum, self.entity_data, origin=1, parent_eid=eid)
                        self.newborn_entities.append((new_eid, cx, cy))
                        self.births_asexual_tick += 1
                        newborn_count += 1
                    continue
                partner_genome_comp = self.em.get_component(partner_id, GenomeComp)
                if partner_genome_comp and partner_genome_comp.genome:
                    child_genome = Genome.crossover(genome_comp.genome, partner_genome_comp.genome, self.config)
                    child_genome = child_genome.mutate(self.config)

                    partner_energy = self.em.get_component(partner_id, Energy)
                    parent_sum = energy.current + (partner_energy.current if partner_energy else 0)

                    spread = self.config.carnivorous_plant.spawn_spread if diet.diet_type == DietType.CARNIVOROUS_PLANT else 5.0
                    cx = pos.x + random.uniform(-spread, spread)
                    cy = pos.y + random.uniform(-spread, spread)
                    cx = max(0.0, min(float(w.width - 1), cx))
                    cy = max(0.0, min(float(w.height - 1), cy))

                    energy.current *= (1.0 - self.config.energy.reproduction_energy_cost)
                    if partner_energy:
                        partner_energy.current *= (1.0 - self.config.energy.reproduction_energy_cost)

                    cd = self.config.reproduction.sexual_cooldown
                    repro.cooldown = cd
                    repro.max_cooldown = cd
                    partner_repro = self.em.get_component(partner_id, Reproduction)
                    if partner_repro:
                        partner_repro.cooldown = partner_repro.max_cooldown

                    new_eid = create_organism(self.em, child_genome, cx, cy, self.config, self.config.energy.child_energy_fraction, parent_sum, self.entity_data, origin=2, parent_eid=eid)
                    self.newborn_entities.append((new_eid, cx, cy))
                    self.births_sexual_tick += 1
                    newborn_count += 1

    def _find_partner(self, eid: int, pos: Position, diet_type: DietType) -> int | None:
        if diet_type == DietType.CARNIVOROUS_PLANT:
            return None
        entity_count = self.em.entity_count
        base_radius = min(250.0, 120.0 + max(0.0, 150.0 - entity_count) * 1.5)
        if diet_type == DietType.PREDATOR:
            base_radius = min(400.0, base_radius * 3.0)
        search_radius = base_radius
        _nearby = self._nearby_set
        self.spatial_hash.query_nearby_excluding_into(pos.x, pos.y, search_radius, eid, _nearby)
        ed = self.entity_data
        for nid in _nearby:
            n_diet = self.em.get_component(nid, Diet)
            if n_diet is None or n_diet.diet_type != diet_type:
                continue
            n_energy = self.em.get_component(nid, Energy)
            if n_energy is None:
                continue
            if ed is not None:
                n_idx = ed.eid_to_idx.get(nid)
                if n_idx is not None and ed.repro_cooldown[n_idx] > 0:
                    continue
            else:
                n_repro = self.em.get_component(nid, Reproduction)
                if n_repro and n_repro.cooldown > 0:
                    continue
            return nid
        return None
