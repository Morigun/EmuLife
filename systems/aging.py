import numpy as np

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from config import Config


class AgingSystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.config = config
        self.entity_data = entity_data

    def update(self, world: object, dt: float) -> None:
        if self.entity_data is None:
            self._update_ecs()
            return
        self._update_soa()

    def _update_soa(self) -> None:
        ed = self.entity_data
        n = ed.count
        s = slice(0, n)

        alive = ed.alive[s]
        ed.age[s] += np.where(alive, 1, 0)

        died_of_age = alive & (ed.age[s] >= ed.max_age[s])
        ed.health[s] = np.where(died_of_age, 0, ed.health[s])

    def _update_ecs(self) -> None:
        from components.age import Age
        from components.health import Health
        for eid in list(self.em.get_entities_with(Age)):
            age = self.em.get_component(eid, Age)
            if age is None:
                continue

            age.current += 1

            if age.current >= age.max_age:
                health = self.em.get_component(eid, Health)
                if health:
                    health.current = 0
