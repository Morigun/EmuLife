from core.ecs import System, EntityManager
from core.entity_data import EntityData
from core.world import World
from utils.spatial_hash import SpatialHash
from config import Config

DIET_HERBIVORE = 0
DIET_OMNIVORE = 1
DIET_PREDATOR = 2
DIET_CARNIVOROUS_PLANT = 3

class InteractionSystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, spatial_hash: SpatialHash, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.config = config
        self.spatial_hash = spatial_hash
        self.entity_data = entity_data
        self._nearby_set: set[int] = set()
        self.tick = 0

    def update(self, world: object, dt: float) -> None:
        self.tick += 1
        if self.entity_data is not None:
            self._update_soa(world, dt)
        else:
            self._update_ecs(world, dt)

    def _update_soa(self, world: object, dt: float) -> None:
        from components.health import Health
        from components.energy import Energy

        ed = self.entity_data
        n = ed.count
        _nearby = self._nearby_set
        processed: set[tuple[int, int]] = set()

        for idx in range(n):
            if not ed.alive[idx]:
                continue

            diet_int = int(ed.diet_type[idx])
            if diet_int != 2:
                continue

            eid = ed.idx_to_eid.get(idx)
            if eid is None:
                continue

            px, py = ed.x[idx], ed.y[idx]
            raw_size = (float(ed.size_gene[idx]) - 3.0) / 7.0
            aggression = float(ed.aggression[idx])

            self.spatial_hash.query_nearby_into(px, py, 5.0, _nearby)
            for nid in _nearby:
                if nid == eid:
                    continue
                pair = (min(eid, nid), max(eid, nid))
                if pair in processed:
                    continue

                n_idx = ed.eid_to_idx.get(nid)
                if n_idx is None or not ed.alive[n_idx]:
                    continue

                dx = px - ed.x[n_idx]
                dy = py - ed.y[n_idx]
                dist_sq = dx * dx + dy * dy

                if dist_sq > 25.0:
                    continue

                processed.add(pair)
                n_diet_int = int(ed.diet_type[n_idx])

                damage_mult = 40.0
                valid_target = n_diet_int != 2 and n_diet_int != 3
                if not valid_target:
                    continue

                damage = raw_size * aggression * damage_mult * dt
                ed.health[n_idx] -= damage

                from components.conditions import Condition, Conditions
                if diet_int == 2 and damage > 3.0:
                    target_conds = self.em.get_component(nid, Conditions)
                    if target_conds is None:
                        target_conds = Conditions()
                        self.em.add_component(nid, target_conds)
                    if len(target_conds.effects) < 3:
                        wound_duration = max(30, int(damage * 8))
                        target_conds.effects.append(
                            Condition("wound", wound_duration, speed_mult=0.7, metabolism_mult=1.3)
                        )

                if diet_int == 2:
                    blood_meal = damage * self.config.energy.blood_meal_fraction
                    ed.energy[idx] = min(float(ed.max_energy[idx]) * 1.5, float(ed.energy[idx]) + blood_meal)
                    a_energy = self.em.get_component(eid, Energy)
                    if a_energy:
                        a_energy.current = float(ed.energy[idx])

                n_health = self.em.get_component(nid, Health)
                if n_health:
                    n_health.current = float(ed.health[n_idx])

                if ed.health[n_idx] <= 0:
                    gain = float(ed.energy[n_idx]) * self.config.energy.predation_efficiency
                    ed.energy[idx] = min(float(ed.max_energy[idx]) * 1.5, float(ed.energy[idx]) + gain + 30.0)
                    n_energy = self.em.get_component(nid, Energy)
                    if n_energy:
                        n_energy.current = float(ed.energy[n_idx])
                    a_energy = self.em.get_component(eid, Energy)
                    if a_energy:
                        a_energy.current = float(ed.energy[idx])

        cpc = self.config.carnivorous_plant
        trap_cd = cpc.trap_cooldown_ticks
        for idx in range(n):
            if not ed.alive[idx]:
                continue
            diet_int = int(ed.diet_type[idx])
            if diet_int != DIET_CARNIVOROUS_PLANT:
                continue
            if self.tick % trap_cd != idx % trap_cd:
                continue

            eid = ed.idx_to_eid.get(idx)
            if eid is None:
                continue

            px, py = ed.x[idx], ed.y[idx]
            trap_power = float(ed.aggression[idx])
            trap_radius = cpc.trap_base_radius + trap_power * (cpc.trap_max_radius - cpc.trap_base_radius)
            targets = 0

            self.spatial_hash.query_nearby_into(px, py, trap_radius, _nearby)
            for nid in _nearby:
                if nid == eid:
                    continue
                if targets >= cpc.max_trap_targets:
                    break

                n_idx = ed.eid_to_idx.get(nid)
                if n_idx is None or not ed.alive[n_idx]:
                    continue
                n_diet_int = int(ed.diet_type[n_idx])
                if n_diet_int == DIET_CARNIVOROUS_PLANT:
                    continue

                dx = px - ed.x[n_idx]
                dy = py - ed.y[n_idx]
                dist_sq = dx * dx + dy * dy
                if dist_sq > trap_radius * trap_radius:
                    continue

                damage = trap_power * cpc.trap_damage_mult * dt
                ed.health[n_idx] -= damage
                targets += 1

                from components.conditions import Condition, Conditions
                if damage > 1.0:
                    target_conds = self.em.get_component(nid, Conditions)
                    if target_conds is None:
                        target_conds = Conditions()
                        self.em.add_component(nid, target_conds)
                    has_trapped = any(e.name == "trapped" for e in target_conds.effects)
                    if not has_trapped and len(target_conds.effects) < 3:
                        trap_duration = max(60, cpc.trap_cooldown_ticks * 3)
                        target_conds.effects.append(
                            Condition("trapped", trap_duration, speed_mult=0.1, metabolism_mult=1.0)
                        )

                n_health = self.em.get_component(nid, Health)
                if n_health:
                    n_health.current = float(ed.health[n_idx])

                if ed.health[n_idx] <= 0:
                    gain = float(ed.energy[n_idx]) * self.config.energy.predation_efficiency
                    ed.energy[idx] = min(float(ed.max_energy[idx]) * 1.5, float(ed.energy[idx]) + gain + 30.0)
                    a_energy = self.em.get_component(eid, Energy)
                    if a_energy:
                        a_energy.current = float(ed.energy[idx])

    def _update_ecs(self, world: object, dt: float) -> None:
        import math
        from components.position import Position
        from components.energy import Energy
        from components.health import Health
        from components.diet import Diet, DietType
        from components.genome_comp import GenomeComp

        processed: set[tuple[int, int]] = set()
        _nearby = self._nearby_set

        for eid in list(self.em.get_entities_with(Position, Energy, Diet)):
            pos = self.em.get_component(eid, Position)
            energy = self.em.get_component(eid, Energy)
            diet = self.em.get_component(eid, Diet)
            if pos is None or energy is None or diet is None:
                continue

            if diet.diet_type != DietType.PREDATOR:
                if diet.diet_type == DietType.CARNIVOROUS_PLANT:
                    cpc = self.config.carnivorous_plant
                    genome_comp = self.em.get_component(eid, GenomeComp)
                    trap_power = genome_comp.genome.aggression if genome_comp and genome_comp.genome else 0.3
                    trap_radius = cpc.trap_base_radius + trap_power * (cpc.trap_max_radius - cpc.trap_base_radius)
                    targets = 0
                    self.spatial_hash.query_nearby_into(pos.x, pos.y, trap_radius, _nearby)
                    for nid in _nearby:
                        if nid == eid or targets >= cpc.max_trap_targets:
                            break
                        n_diet = self.em.get_component(nid, Diet)
                        if n_diet and n_diet.diet_type == DietType.CARNIVOROUS_PLANT:
                            continue
                        npos = self.em.get_component(nid, Position)
                        if npos is None:
                            continue
                        ddx = pos.x - npos.x
                        ddy = pos.y - npos.y
                        if math.sqrt(ddx * ddx + ddy * ddy) > trap_radius:
                            continue
                        damage = trap_power * cpc.trap_damage_mult * dt
                        n_health = self.em.get_component(nid, Health)
                        if n_health:
                            n_health.current -= damage
                            targets += 1
                            if self.entity_data is not None:
                                n_idx = self.entity_data.eid_to_idx.get(nid)
                                if n_idx is not None:
                                    self.entity_data.health[n_idx] = n_health.current

                            if damage > 1.0:
                                from components.conditions import Condition, Conditions
                                target_conds = self.em.get_component(nid, Conditions)
                                if target_conds is None:
                                    target_conds = Conditions()
                                    self.em.add_component(nid, target_conds)
                                has_trapped = any(e.name == "trapped" for e in target_conds.effects)
                                if not has_trapped and len(target_conds.effects) < 3:
                                    trap_duration = max(60, cpc.trap_cooldown_ticks * 3)
                                    target_conds.effects.append(
                                        Condition("trapped", trap_duration, speed_mult=0.1, metabolism_mult=1.0)
                                    )

                            if n_health and n_health.current <= 0:
                                n_energy = self.em.get_component(nid, Energy)
                                gain = (n_energy.current * self.config.energy.predation_efficiency) if n_energy else 0
                                energy.current = min(energy.max_value * 1.5, energy.current + gain + 30.0)
                                if self.entity_data is not None:
                                    a_idx = self.entity_data.eid_to_idx.get(eid)
                                    if a_idx is not None:
                                        self.entity_data.energy[a_idx] = energy.current
                continue

            genome_comp = self.em.get_component(eid, GenomeComp)
            aggression = genome_comp.genome.aggression if genome_comp and genome_comp.genome else 0.5
            size = genome_comp.genome.size if genome_comp and genome_comp.genome else 0.5

            self.spatial_hash.query_nearby_into(pos.x, pos.y, 5.0, _nearby)
            for nid in _nearby:
                if nid == eid:
                    continue
                pair = (min(eid, nid), max(eid, nid))
                if pair in processed:
                    continue

                npos = self.em.get_component(nid, Position)
                if npos is None:
                    continue

                dx = pos.x - npos.x
                dy = pos.y - npos.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > 5.0:
                    continue

                processed.add(pair)
                n_diet = self.em.get_component(nid, Diet)

                if diet.diet_type == DietType.PREDATOR:
                    if n_diet and n_diet.diet_type != DietType.PREDATOR and n_diet.diet_type != DietType.CARNIVOROUS_PLANT:
                        n_health = self.em.get_component(nid, Health)
                        if n_health:
                            damage = size * aggression * 40.0 * dt
                            n_health.current -= damage
                            if self.entity_data is not None:
                                n_idx = self.entity_data.eid_to_idx.get(nid)
                                if n_idx is not None:
                                    self.entity_data.health[n_idx] = n_health.current
                            blood_meal = damage * self.config.energy.blood_meal_fraction
                            energy.current = min(energy.max_value * 1.5, energy.current + blood_meal)
                            if self.entity_data is not None:
                                a_idx = self.entity_data.eid_to_idx.get(eid)
                                if a_idx is not None:
                                    self.entity_data.energy[a_idx] = energy.current
                            if n_health.current <= 0:
                                n_energy = self.em.get_component(nid, Energy)
                                gain = (n_energy.current * self.config.energy.predation_efficiency) if n_energy else 0
                                energy.current = min(energy.max_value * 1.5, energy.current + gain + 30.0)
                                if self.entity_data is not None:
                                    a_idx = self.entity_data.eid_to_idx.get(eid)
                                    if a_idx is not None:
                                        self.entity_data.energy[a_idx] = energy.current
