import numpy as np

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from config import Config
from components.conditions import Conditions


class ConditionSystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, entity_data: EntityData = None):
        self.em = entity_manager
        self.config = config
        self.entity_data = entity_data

    def update(self, world: object, dt: float) -> None:
        if self.entity_data is None:
            return
        self._update_soa(world)

    def _update_soa(self, world: object) -> None:
        from core.world import World
        w: World = world
        ed = self.entity_data
        n = ed.count

        ed.speed_mod[:n] = 1.0
        ed.metabolism_mod[:n] = 1.0
        ed.efficiency_mod[:n] = 1.0

        conditions_map = self.em.get_all_of_type(Conditions)
        for eid, conds in conditions_map.items():
            idx = ed.eid_to_idx.get(eid)
            if idx is None or not ed.alive[idx]:
                continue

            expired = []
            for i, eff in enumerate(conds.effects):
                eff.duration -= 1
                if eff.duration <= 0:
                    expired.append(i)
                else:
                    ed.speed_mod[idx] *= eff.speed_mult
                    ed.metabolism_mod[idx] *= eff.metabolism_mult
                    ed.efficiency_mod[idx] *= eff.efficiency_mult

            for i in reversed(expired):
                conds.effects.pop(i)

        if w.metabolism_mult != 1.0:
            ed.metabolism_mod[:n] *= w.metabolism_mult
