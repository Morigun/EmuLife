from core.ecs import System
from core.world import World
from config import Config


class WorldUpdateSystem(System):
    def __init__(self, config: Config) -> None:
        self.config = config
        self._tick_counter = 0
        self._regen_interval = 2

    def update(self, world: object, dt: float) -> None:
        self._tick_counter += 1
        w: World = world
        if self._tick_counter % self._regen_interval == 0:
            w.regenerate_all()

        decay = 1.0 - self.config.abiogenesis.biomass_decay_rate
        w.biomass *= decay
