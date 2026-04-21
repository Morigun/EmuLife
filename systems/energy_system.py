import numpy as np

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from core.world import World
from config import Config


class EnergySystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.config = config
        self.entity_data = entity_data

    def update(self, world: object, dt: float) -> None:
        if self.entity_data is None:
            self._update_ecs(world, dt)
            return
        self._update_soa(world, dt)

    def _update_soa(self, world: object, dt: float) -> None:
        w: World = world
        ed = self.entity_data
        ec = self.config.energy
        n = ed.count
        s = slice(0, n)

        alive = ed.alive[s]
        raw_size = (ed.size_gene[s] - 3.0) / 7.0
        cost = ed.metabolism[s] * (0.5 + raw_size) * dt
        speed = np.sqrt(ed.dx[s] ** 2 + ed.dy[s] ** 2)
        cost += speed * raw_size * 0.5 * dt

        ed.energy[s] -= np.where(alive, cost, 0)

        ix = np.clip(ed.x[s].astype(np.int32), 0, w.width - 1)
        iy = np.clip(ed.y[s].astype(np.int32), 0, w.height - 1)

        from core.world import TileType
        is_water = w.tile_types[iy, ix] == TileType.WATER

        non_pred = ed.diet_type[s] != 2
        food = w.food_values[iy, ix]
        habitat = ed.habitat[s]

        can_eat = alive & non_pred & (food > 0) & (
            ((habitat == 0) & is_water)
            | ((habitat == 1) & ~is_water)
            | (habitat == 2)
        )

        flat_indices = iy * w.width + ix
        counts = np.zeros(w.width * w.height, dtype=np.float32)
        np.add.at(counts, flat_indices, 1.0)
        per_entity_count = counts[flat_indices]

        available = np.maximum(food, 0)
        share = available / np.maximum(per_entity_count, 1.0)
        eat_amount = np.minimum(ec.energy_from_food * dt, share)

        ed.energy[s] += np.where(can_eat, eat_amount * ed.efficiency[s], 0).astype(np.float32)

        depletion = np.where(can_eat, eat_amount, 0)
        np.add.at(w.food_values, (iy, ix), -depletion)
        np.clip(w.food_values, 0, None, out=w.food_values)

        ed.energy[s] = np.minimum(ed.energy[s], ed.max_energy[s])

    def _update_ecs(self, world: object, dt: float) -> None:
        w: World = world
        ec = self.config.energy

        from components.position import Position
        from components.velocity import Velocity
        from components.energy import Energy
        from components.metabolism import Metabolism
        from components.diet import Diet, DietType
        from components.genome_comp import GenomeComp
        import math

        for eid in list(self.em.get_entities_with(Position, Energy, Metabolism)):
            pos = self.em.get_component(eid, Position)
            energy = self.em.get_component(eid, Energy)
            metab = self.em.get_component(eid, Metabolism)
            if pos is None or energy is None or metab is None:
                continue

            genome_comp = self.em.get_component(eid, GenomeComp)
            size = genome_comp.genome.size if genome_comp else 0.5

            base_cost = metab.rate * (0.5 + size) * dt
            vel = self.em.get_component(eid, Velocity)
            if vel is not None:
                speed = math.sqrt(vel.dx ** 2 + vel.dy ** 2)
                base_cost += speed * size * 0.5 * dt

            energy.current -= base_cost

            diet = self.em.get_component(eid, Diet)
            if diet is None or diet.diet_type != DietType.PREDATOR:
                ix, iy = int(pos.x), int(pos.y)
                fv = w.get_food(ix, iy)
                if fv > 0:
                    eat_amount = min(ec.energy_from_food * dt, fv)
                    energy.current += eat_amount * diet.efficiency if diet else eat_amount
                    w.food_values[iy, ix] -= eat_amount

            energy.current = min(energy.current, energy.max_value)
