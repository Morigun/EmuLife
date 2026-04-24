import numpy as np

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from core.world import World
from utils.spatial_hash import SpatialHash
from config import Config


class DeathSystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, spatial_hash: SpatialHash = None, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.config = config
        self.spatial_hash = spatial_hash
        self.entity_data = entity_data
        self.deaths_tick: int = 0

    def update(self, world: object, dt: float) -> None:
        self.deaths_tick = 0
        if self.entity_data is None:
            self._update_ecs()
            return
        self._update_soa(world)

    def _update_soa(self, world: object) -> None:
        w: World = world
        ed = self.entity_data
        n = ed.count
        s = slice(0, n)

        dead = ed.alive[s] & ((ed.health[s] <= 0) | (ed.energy[s] <= 0))

        dead_indices = np.where(dead)[0]
        for idx in dead_indices:
            idx_int = int(idx)
            eid = ed.idx_to_eid.get(idx_int)
            if eid is None:
                continue

            ix = int(ed.x[idx_int])
            iy = int(ed.y[idx_int])
            if 0 <= ix < w.width and 0 <= iy < w.height:
                w.biomass[iy, ix] += float(ed.size_gene[idx_int]) * 5.0

            if self.spatial_hash:
                self.spatial_hash.remove(eid)
            ed.remove(eid)
            self.em.remove_entity(eid)
            self.deaths_tick += 1

    def _update_ecs(self) -> None:
        from components.health import Health
        from components.energy import Energy
        to_remove: list[int] = []

        for eid in list(self.em.get_entities_with(Health, Energy)):
            health = self.em.get_component(eid, Health)
            energy = self.em.get_component(eid, Energy)
            if health is None or energy is None:
                continue

            if health.current <= 0 or energy.current <= 0:
                to_remove.append(eid)

        for eid in to_remove:
            if self.spatial_hash:
                self.spatial_hash.remove(eid)
            self.em.remove_entity(eid)
            self.deaths_tick += 1
