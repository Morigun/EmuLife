import math

import numpy as np

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from components.position import Position
from components.energy import Energy
from components.sensor import Sensor, NearbyEntity
from components.diet import Diet, DietType
from utils.spatial_hash import SpatialHash
from utils.numba_kernels import find_nearest_food_kernel, HAS_NUMBA
from core.world import World
from config import Config

STAGGER_BATCHES = 6

_DIET_INT_TO_TYPE = {0: DietType.HERBIVORE, 1: DietType.OMNIVORE, 2: DietType.PREDATOR}

NEARBY_MAX = 16


class SensorSystem(System):
    def __init__(self, entity_manager: EntityManager, spatial_hash: SpatialHash, config: Config, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.spatial_hash = spatial_hash
        self.config = config
        self.entity_data = entity_data
        self.tick = 0
        self._nearby_set: set[int] = set()

        self.nearby_eids = np.full((EntityData.MAX_ENTITIES, NEARBY_MAX), -1, dtype=np.int32)
        self.nearby_dist = np.full((EntityData.MAX_ENTITIES, NEARBY_MAX), 1e18, dtype=np.float32)
        self.nearby_diet = np.full((EntityData.MAX_ENTITIES, NEARBY_MAX), -1, dtype=np.int8)
        self.nearby_count = np.zeros(EntityData.MAX_ENTITIES, dtype=np.int32)

        self.food_x = np.full(EntityData.MAX_ENTITIES, -1.0, dtype=np.float32)
        self.food_y = np.full(EntityData.MAX_ENTITIES, -1.0, dtype=np.float32)

    def update(self, world: object, dt: float) -> None:
        w: World = world
        self.tick += 1
        _nearby = self._nearby_set

        if self.entity_data is not None:
            self._update_staggered(w, _nearby)
        else:
            self._update_ecs(w, _nearby)

    def _update_staggered(self, w: World, _nearby: set[int]) -> None:
        ed = self.entity_data
        current_batch = self.tick % STAGGER_BATCHES
        n = ed.count

        nearby_eids = self.nearby_eids
        nearby_dist = self.nearby_dist
        nearby_diet_arr = self.nearby_diet
        nearby_count = self.nearby_count
        food_x = self.food_x
        food_y = self.food_y
        sh = self.spatial_hash

        ed_x = ed.x
        ed_y = ed.y
        ed_alive = ed.alive
        ed_diet_type = ed.diet_type
        ed_energy = ed.energy
        ed_max_energy = ed.max_energy
        ed_vision = ed.vision
        ed_habitat = ed.habitat
        ed_eid_to_idx = ed.eid_to_idx
        ed_idx_to_eid = ed.idx_to_eid

        vision_mult = w.vision_mult

        for idx in range(n):
            if not ed_alive[idx]:
                continue
            if idx % STAGGER_BATCHES != current_batch:
                continue

            eid = ed_idx_to_eid.get(idx)
            if eid is None:
                continue

            px = float(ed_x[idx])
            py = float(ed_y[idx])
            radius = float(ed_vision[idx]) * vision_mult
            radius_sq = radius * radius

            old_count = int(nearby_count[idx])
            if old_count > 0:
                nearby_eids[idx, :old_count] = -1
                nearby_dist[idx, :old_count] = 1e18
                nearby_diet_arr[idx, :old_count] = -1
            nearby_count[idx] = 0
            food_x[idx] = -1.0
            food_y[idx] = -1.0

            sh.query_nearby_excluding_into(px, py, radius, eid, _nearby)

            count = 0
            for nid in _nearby:
                if count >= NEARBY_MAX:
                    break
                n_idx = ed_eid_to_idx.get(nid)
                if n_idx is None or not ed_alive[n_idx]:
                    continue
                dx = float(ed_x[n_idx]) - px
                dy = float(ed_y[n_idx]) - py
                dist_sq = dx * dx + dy * dy
                if dist_sq <= radius_sq:
                    nearby_eids[idx, count] = nid
                    nearby_dist[idx, count] = dist_sq ** 0.5
                    nearby_diet_arr[idx, count] = int(ed_diet_type[n_idx])
                    count += 1

            if count > 1:
                order = np.argsort(nearby_dist[idx, :count])
                tmp_eids = nearby_eids[idx, :count].copy()
                tmp_dist = nearby_dist[idx, :count].copy()
                tmp_diet = nearby_diet_arr[idx, :count].copy()
                for j in range(count):
                    nearby_eids[idx, j] = tmp_eids[order[j]]
                    nearby_dist[idx, j] = tmp_dist[order[j]]
                    nearby_diet_arr[idx, j] = tmp_diet[order[j]]

            nearby_count[idx] = count

            if ed_energy[idx] < ed_max_energy[idx] * 0.5:
                habitat_int = int(ed_habitat[idx])
                pos = self._find_nearest_food(w, px, py, radius, habitat_int)
                if pos is not None:
                    food_x[idx] = pos[0]
                    food_y[idx] = pos[1]

    def _update_ecs(self, w: World, _nearby: set[int]) -> None:
        for eid in self.em.get_entities_with(Position, Sensor):
            pos = self.em.get_component(eid, Position)
            sensor_comp = self.em.get_component(eid, Sensor)
            if pos is None or sensor_comp is None:
                continue

            radius = sensor_comp.radius * w.vision_mult
            self.spatial_hash.query_nearby_excluding_into(pos.x, pos.y, radius, eid, _nearby)

            filtered = []
            for nid in _nearby:
                npos = self.em.get_component(nid, Position)
                if npos is None:
                    continue
                dx = npos.x - pos.x
                dy = npos.y - pos.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= radius:
                    n_diet = self.em.get_component(nid, Diet)
                    diet_type = n_diet.diet_type if n_diet else None
                    filtered.append(NearbyEntity(eid=nid, dist=dist, diet_type=diet_type))

            filtered.sort(key=lambda x: x.dist)
            sensor_comp.nearby_entities = filtered

            energy = self.em.get_component(eid, Energy)
            hungry = energy is not None and energy.current < energy.max_value * 0.5
            cache_expired = self.tick - sensor_comp.food_cache_tick >= sensor_comp.food_cache_interval

            if not hungry:
                sensor_comp.nearest_food_pos = None
            elif cache_expired:
                from components.habitat import Habitat
                hab = self.em.get_component(eid, Habitat)
                habitat_int = {"aquatic": 0, "terrestrial": 1, "amphibious": 2}.get(hab.habitat_type, 1) if hab else 1
                sensor_comp.nearest_food_pos = self._find_nearest_food(w, pos.x, pos.y, radius, habitat_int)
                sensor_comp.food_cache_tick = self.tick

    def _find_nearest_food(self, world: World, x: float, y: float, radius: float, habitat_int: int = 1):
        if HAS_NUMBA:
            found, fx, fy = find_nearest_food_kernel(
                world.food_values, world.tile_types,
                x, y, radius, habitat_int, world.width, world.height,
            )
            return (fx, fy) if found else None

        from core.world import TileType

        step = 5
        r = int(radius)
        ix, iy = int(x), int(y)

        x0 = max(0, ix - r)
        x1 = min(world.width, ix + r + 1)
        y0 = max(0, iy - r)
        y1 = min(world.height, iy + r + 1)

        food = world.food_values[y0:y1:step, x0:x1:step]
        if food.size == 0:
            return None

        tile_region = world.tile_types[y0:y1:step, x0:x1:step]
        is_water = tile_region == TileType.WATER

        has_food = food > 1.0
        if habitat_int == 0:
            has_food = has_food & is_water
        elif habitat_int == 1:
            has_food = has_food & ~is_water

        if not np.any(has_food):
            return None

        rows, cols = food.shape
        sample_ys = np.arange(rows) * step + y0
        sample_xs = np.arange(cols) * step + x0

        dy = (sample_ys - iy).astype(np.float64)
        dx = (sample_xs - ix).astype(np.float64)
        dist_sq = dy[:, np.newaxis] ** 2 + dx[np.newaxis, :] ** 2

        radius_sq = radius * radius
        mask = has_food & (dist_sq <= radius_sq)

        if not np.any(mask):
            return None

        dist_sq_masked = np.where(mask, dist_sq, np.inf)
        min_flat = np.argmin(dist_sq_masked)
        min_r, min_c = np.unravel_index(min_flat, dist_sq_masked.shape)

        return (float(sample_xs[min_c]), float(sample_ys[min_r]))
