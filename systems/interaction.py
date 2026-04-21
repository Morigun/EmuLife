from core.ecs import System, EntityManager
from core.entity_data import EntityData
from core.world import World
from utils.spatial_hash import SpatialHash
from config import Config

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
            if diet_int == 0:
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

                if dist_sq > 9.0:
                    continue

                processed.add(pair)
                n_diet_int = int(ed.diet_type[n_idx])

                damage_mult = 15.0 if diet_int == 2 else 5.0
                valid_target = (diet_int == 2 and n_diet_int != 2) or (diet_int == 1 and n_diet_int == 0)
                if not valid_target:
                    continue

                damage = raw_size * aggression * damage_mult * dt
                ed.health[n_idx] -= damage

                n_health = self.em.get_component(nid, Health)
                if n_health:
                    n_health.current = float(ed.health[n_idx])

                if ed.health[n_idx] <= 0:
                    eff = self.config.energy.predation_efficiency
                    if diet_int == 1:
                        eff *= 0.5
                    gain = float(ed.energy[n_idx]) * eff
                    ed.energy[idx] = min(float(ed.max_energy[idx]), float(ed.energy[idx]) + gain)
                    n_energy = self.em.get_component(nid, Energy)
                    if n_energy:
                        n_energy.current = float(ed.energy[n_idx])
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

            if diet.diet_type == DietType.HERBIVORE:
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

                if dist > 3.0:
                    continue

                processed.add(pair)
                n_diet = self.em.get_component(nid, Diet)

                if diet.diet_type == DietType.PREDATOR:
                    if n_diet and n_diet.diet_type != DietType.PREDATOR:
                        n_health = self.em.get_component(nid, Health)
                        if n_health:
                            damage = size * aggression * 15.0 * dt
                            n_health.current -= damage
                            if self.entity_data is not None:
                                n_idx = self.entity_data.eid_to_idx.get(nid)
                                if n_idx is not None:
                                    self.entity_data.health[n_idx] = n_health.current
                            if n_health.current <= 0:
                                n_energy = self.em.get_component(nid, Energy)
                                gain = (n_energy.current * self.config.energy.predation_efficiency) if n_energy else 0
                                energy.current = min(energy.max_value, energy.current + gain)
                                if self.entity_data is not None:
                                    a_idx = self.entity_data.eid_to_idx.get(eid)
                                    if a_idx is not None:
                                        self.entity_data.energy[a_idx] = energy.current

                elif diet.diet_type == DietType.OMNIVORE:
                    if n_diet and n_diet.diet_type == DietType.HERBIVORE:
                        n_health = self.em.get_component(nid, Health)
                        if n_health:
                            damage = size * aggression * 5.0 * dt
                            n_health.current -= damage
                            if self.entity_data is not None:
                                n_idx = self.entity_data.eid_to_idx.get(nid)
                                if n_idx is not None:
                                    self.entity_data.health[n_idx] = n_health.current
                            if n_health.current <= 0:
                                n_energy = self.em.get_component(nid, Energy)
                                gain = (n_energy.current * self.config.energy.predation_efficiency * 0.5) if n_energy else 0
                                energy.current = min(energy.max_value, energy.current + gain)
                                if self.entity_data is not None:
                                    a_idx = self.entity_data.eid_to_idx.get(eid)
                                    if a_idx is not None:
                                        self.entity_data.energy[a_idx] = energy.current
