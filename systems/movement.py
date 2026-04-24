import numpy as np

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from core.world import World
from utils.numba_kernels import movement_update_kernel, HAS_NUMBA
from config import Config


class MovementSystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.config = config
        self.entity_data = entity_data
        self.moved_entities: list[tuple[int, float, float]] = []
        self.moved_mask: np.ndarray | None = None
        self.old_x: np.ndarray | None = None
        self.old_y: np.ndarray | None = None

    def update(self, world: object, dt: float) -> None:
        if self.entity_data is None:
            self._update_ecs(world, dt)
            return
        self._update_soa(world, dt)

    def _update_soa(self, world: object, dt: float) -> None:
        w: World = world
        ed = self.entity_data
        n = ed.count

        if HAS_NUMBA:
            moved, old_x, old_y = movement_update_kernel(
                ed.x, ed.y, ed.dx, ed.dy, ed.habitat, ed.alive, ed.diet_type,
                w.tile_types, n, w.width, w.height, dt,
            )
            self.moved_mask = moved
            self.old_x = old_x
            self.old_y = old_y
            return

        s = slice(0, n)

        plant_mask = ed.diet_type[s] == 3
        ed.dx[s] = np.where(plant_mask & ed.alive[s], 0.0, ed.dx[s])
        ed.dy[s] = np.where(plant_mask & ed.alive[s], 0.0, ed.dy[s])

        old_x = ed.x[s].copy()
        old_y = ed.y[s].copy()

        new_x = np.clip(ed.x[s] + ed.dx[s] * dt, 0, w.width - 1)
        new_y = np.clip(ed.y[s] + ed.dy[s] * dt, 0, w.height - 1)

        ix = np.clip(new_x.astype(np.int32), 0, w.width - 1)
        iy = np.clip(new_y.astype(np.int32), 0, w.height - 1)
        tile_types = w.tile_types[iy, ix]

        from core.world import TileType
        is_water = tile_types == TileType.WATER

        habitat = ed.habitat[s]
        terrestrial = habitat == 1
        aquatic = habitat == 0
        amphibious = habitat == 2

        cur_ix = np.clip(ed.x[s].astype(np.int32), 0, w.width - 1)
        cur_iy = np.clip(ed.y[s].astype(np.int32), 0, w.height - 1)
        cur_is_water = w.tile_types[cur_iy, cur_ix] == TileType.WATER
        on_wrong_tile = (terrestrial & cur_is_water) | (aquatic & ~cur_is_water)

        walkable = (
            on_wrong_tile
            | (terrestrial & ~is_water)
            | (aquatic & is_water)
            | amphibious
        ) & ed.alive[s]

        ed.x[s] = np.where(walkable, new_x, ed.x[s])
        ed.y[s] = np.where(walkable, new_y, ed.y[s])

        bounce = ~walkable & ed.alive[s]
        ed.dx[s] = np.where(bounce, -ed.dx[s] * 0.5, ed.dx[s])
        ed.dy[s] = np.where(bounce, -ed.dy[s] * 0.5, ed.dy[s])

        self.moved_mask = walkable
        self.old_x = old_x
        self.old_y = old_y

    def _update_ecs(self, world: object, dt: float) -> None:
        w: World = world
        self.moved_entities.clear()
        from components.position import Position
        from components.velocity import Velocity
        from components.habitat import Habitat
        from components.diet import Diet, DietType
        for eid in self.em.get_entities_with(Position, Velocity):
            pos = self.em.get_component(eid, Position)
            vel = self.em.get_component(eid, Velocity)
            if pos is None or vel is None:
                continue

            diet_comp = self.em.get_component(eid, Diet)
            if diet_comp is not None and diet_comp.diet_type == DietType.CARNIVOROUS_PLANT:
                vel.dx = 0.0
                vel.dy = 0.0
                continue

            old_x, old_y = pos.x, pos.y
            new_x = pos.x + vel.dx * dt
            new_y = pos.y + vel.dy * dt

            new_x = max(0.0, min(float(w.width - 1), new_x))
            new_y = max(0.0, min(float(w.height - 1), new_y))

            hab = self.em.get_component(eid, Habitat)
            on_wrong_tile = False
            if hab is not None:
                cur_is_water = not w.is_walkable(pos.x, pos.y)
                if hab.habitat_type == "terrestrial" and cur_is_water:
                    on_wrong_tile = True
                elif hab.habitat_type == "aquatic" and not cur_is_water:
                    on_wrong_tile = True

            if on_wrong_tile or w.is_walkable(new_x, new_y):
                pos.x = new_x
                pos.y = new_y
                self.moved_entities.append((eid, old_x, old_y))
            else:
                vel.dx = -vel.dx * 0.5
                vel.dy = -vel.dy * 0.5
