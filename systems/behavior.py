import math
import random

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from components.position import Position
from components.velocity import Velocity
from components.energy import Energy
from components.sensor import Sensor
from components.diet import Diet, DietType
from components.genome_comp import GenomeComp
from components.reproduction import Reproduction
from core.world import World
from utils.spatial_hash import SpatialHash
from config import Config

STAGGER_BATCHES = 6


class BehaviorSystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, entity_data: EntityData = None, spatial_hash: SpatialHash = None) -> None:
        self.em = entity_manager
        self.config = config
        self.entity_data = entity_data
        self.spatial_hash = spatial_hash
        self.tick = 0
        self._scent_set: set[int] = set()

    def update(self, world: object, dt: float) -> None:
        self.tick += 1
        if self.entity_data is not None:
            self._update_staggered(world, dt)
        else:
            self._update_ecs(world)

    def _update_staggered(self, world: object, dt: float) -> None:
        ed = self.entity_data
        current_batch = self.tick % STAGGER_BATCHES
        n = ed.count

        sensor_sys = getattr(self, '_sensor_system', None)
        nearby_eids = sensor_sys.nearby_eids if sensor_sys else None
        nearby_dist = sensor_sys.nearby_dist if sensor_sys else None
        nearby_diet = sensor_sys.nearby_diet if sensor_sys else None
        nearby_count = sensor_sys.nearby_count if sensor_sys else None
        food_x = sensor_sys.food_x if sensor_sys else None
        food_y = sensor_sys.food_y if sensor_sys else None

        use_soa_nearby = nearby_eids is not None

        _scent_set = self._scent_set
        sh = self.spatial_hash
        em = self.em

        for idx in range(n):
            if not ed.alive[idx]:
                continue
            if idx % STAGGER_BATCHES != current_batch:
                continue

            diet_int = int(ed.diet_type[idx])
            if diet_int == 3:
                ed.dx[idx] = 0.0
                ed.dy[idx] = 0.0
                continue

            px, py = ed.x[idx], ed.y[idx]
            max_speed = float(ed.speed_gene[idx]) * float(ed.speed_mod[idx])

            target_dx, target_dy = 0.0, 0.0
            action = "wander"

            if use_soa_nearby:
                action = self._decide_soa_fast(idx, ed, nearby_diet, nearby_count)
            else:
                eid = ed.idx_to_eid.get(idx)
                if eid is not None:
                    sensor = em.get_component(eid, Sensor)
                    if sensor is not None:
                        action = self._decide_soa(idx, eid, ed, sensor)

            if action == "flee":
                target_dx, target_dy = self._flee_soa_fast(idx, ed, nearby_eids, nearby_diet, nearby_count) if use_soa_nearby else self._flee_soa(idx, None, ed)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            elif action == "hunt":
                target_dx, target_dy = self._hunt_soa_fast(idx, ed, nearby_eids, nearby_diet, nearby_count) if use_soa_nearby else self._hunt_soa(idx, None, ed)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            elif action == "reproduce":
                target_dx, target_dy = self._seek_mate_soa_fast(idx, ed, nearby_eids, nearby_diet, nearby_count) if use_soa_nearby else self._seek_mate_soa(idx, None, None, ed)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            elif action == "eat":
                if use_soa_nearby and food_x is not None:
                    target_dx, target_dy = self._seek_food_soa_fast(idx, ed, food_x, food_y, world)
                else:
                    target_dx, target_dy = self._seek_food_soa(idx, None, ed, world)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            else:
                if diet_int == 2 and sh is not None:
                    eid = ed.idx_to_eid.get(idx)
                    if eid is not None:
                        scent_dx, scent_dy = self._scent_hunt(idx, eid, ed, px, py)
                        if scent_dx != 0.0 or scent_dy != 0.0:
                            target_dx, target_dy = scent_dx, scent_dy
                        else:
                            target_dx = random.uniform(-1, 1)
                            target_dy = random.uniform(-1, 1)
                    else:
                        target_dx = random.uniform(-1, 1)
                        target_dy = random.uniform(-1, 1)
                else:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)

            if action == "hunt":
                max_speed *= 2.5

            if target_dx != 0 or target_dy != 0:
                mag = math.sqrt(target_dx ** 2 + target_dy ** 2)
                if mag > 0:
                    target_dx /= mag
                    target_dy /= mag

            ed.dx[idx] = target_dx * max_speed
            ed.dy[idx] = target_dy * max_speed

    def _decide_soa_fast(self, idx, ed, nearby_diet, nearby_count):
        low_energy = ed.energy[idx] < ed.max_energy[idx] * 0.3
        diet_int = int(ed.diet_type[idx])
        nc = int(nearby_count[idx])

        if diet_int == 2:
            limit = min(nc, 8)
            for j in range(limit):
                nd = int(nearby_diet[idx, j])
                if nd != 2 and nd != 3:
                    return "hunt"

        if diet_int in (0, 1):
            limit = min(nc, 5)
            for j in range(limit):
                if int(nearby_diet[idx, j]) == 2:
                    return "flee"

        if low_energy:
            if diet_int != 2:
                return "eat"

        repro_type_int = int(ed.repro_type[idx])
        if repro_type_int in (1, 2):
            if ed.repro_cooldown[idx] <= 0:
                repro_threshold = float(ed.repro_threshold[idx]) * ed.max_energy[idx]
                if ed.energy[idx] > repro_threshold:
                    return "reproduce"

        return "wander"

    def _flee_soa_fast(self, idx, ed, nearby_eids, nearby_diet, nearby_count):
        px, py = ed.x[idx], ed.y[idx]
        dx, dy = 0.0, 0.0
        nc = int(nearby_count[idx])
        limit = min(nc, 5)
        for j in range(limit):
            if int(nearby_diet[idx, j]) == 2:
                nid = int(nearby_eids[idx, j])
                n_idx = ed.eid_to_idx.get(nid)
                if n_idx is not None:
                    dx += px - ed.x[n_idx]
                    dy += py - ed.y[n_idx]
        return dx, dy

    def _hunt_soa_fast(self, idx, ed, nearby_eids, nearby_diet, nearby_count):
        px, py = ed.x[idx], ed.y[idx]
        nc = int(nearby_count[idx])
        for j in range(nc):
            nd = int(nearby_diet[idx, j])
            if nd != 2 and nd != 3:
                nid = int(nearby_eids[idx, j])
                n_idx = ed.eid_to_idx.get(nid)
                if n_idx is not None:
                    return ed.x[n_idx] - px, ed.y[n_idx] - py
        return 0.0, 0.0

    def _seek_mate_soa_fast(self, idx, ed, nearby_eids, nearby_diet, nearby_count):
        diet_int = int(ed.diet_type[idx])
        nc = int(nearby_count[idx])
        for j in range(nc):
            if int(nearby_diet[idx, j]) == diet_int:
                nid = int(nearby_eids[idx, j])
                n_idx = ed.eid_to_idx.get(nid)
                if n_idx is not None and ed.repro_cooldown[n_idx] <= 0:
                    return ed.x[n_idx] - ed.x[idx], ed.y[n_idx] - ed.y[idx]
        return 0.0, 0.0

    def _seek_food_soa_fast(self, idx, ed, food_x, food_y, world=None):
        if world is not None:
            w = world
            ix = int(ed.x[idx])
            iy = int(ed.y[idx])
            if 0 <= ix < w.width and 0 <= iy < w.height:
                if w.food_values[iy, ix] > 5.0:
                    return 0.0, 0.0
        fx = float(food_x[idx])
        fy = float(food_y[idx])
        if fx >= 0:
            return fx - ed.x[idx], fy - ed.y[idx]
        return 0.0, 0.0

    def _decide_soa(self, idx, eid, ed, sensor):
        low_energy = ed.energy[idx] < ed.max_energy[idx] * 0.3
        diet_int = int(ed.diet_type[idx])

        if diet_int == 2:
            for nb in sensor.nearby_entities[:8]:
                if nb.diet_type is not None and nb.diet_type != DietType.PREDATOR:
                    return "hunt"

        if diet_int in (0, 1):
            for nb in sensor.nearby_entities[:5]:
                if nb.diet_type == DietType.PREDATOR:
                    return "flee"

        if low_energy:
            if diet_int != 2:
                return "eat"

        repro_type_int = int(ed.repro_type[idx])
        if repro_type_int in (1, 2):
            repro_cooldown = ed.repro_cooldown[idx]
            if repro_cooldown <= 0:
                repro_threshold = float(ed.repro_threshold[idx]) * ed.max_energy[idx]
                if ed.energy[idx] > repro_threshold:
                    return "reproduce"

        return "wander"

    def _flee_soa(self, idx, sensor, ed):
        px, py = ed.x[idx], ed.y[idx]
        dx, dy = 0.0, 0.0
        if sensor:
            for nb in sensor.nearby_entities[:5]:
                if nb.diet_type == DietType.PREDATOR:
                    n_idx = ed.eid_to_idx.get(nb.eid)
                    if n_idx is not None:
                        dx += px - ed.x[n_idx]
                        dy += py - ed.y[n_idx]
        return dx, dy

    def _hunt_soa(self, idx, sensor, ed):
        px, py = ed.x[idx], ed.y[idx]
        if sensor:
            for nb in sensor.nearby_entities:
                if nb.diet_type is not None and nb.diet_type != DietType.PREDATOR:
                    n_idx = ed.eid_to_idx.get(nb.eid)
                    if n_idx is not None:
                        return ed.x[n_idx] - px, ed.y[n_idx] - py
        return 0.0, 0.0

    def _seek_food_soa(self, idx, sensor, ed, world=None):
        ix = int(ed.x[idx])
        iy = int(ed.y[idx])
        if world is not None:
            w = world
            if 0 <= ix < w.width and 0 <= iy < w.height:
                if w.food_values[iy, ix] > 5.0:
                    return 0.0, 0.0
        if sensor and sensor.nearest_food_pos:
            fx, fy = sensor.nearest_food_pos
            return fx - ed.x[idx], fy - ed.y[idx]
        return 0.0, 0.0

    def _seek_mate_soa(self, idx, eid, sensor, ed):
        diet_int = int(ed.diet_type[idx])
        if sensor:
            for nb in sensor.nearby_entities:
                if nb.diet_type is not None:
                    n_idx = ed.eid_to_idx.get(nb.eid)
                    if n_idx is not None and int(ed.diet_type[n_idx]) == diet_int and ed.repro_cooldown[n_idx] <= 0:
                        return ed.x[n_idx] - ed.x[idx], ed.y[n_idx] - ed.y[idx]
        return 0.0, 0.0

    def _scent_hunt(self, idx, eid, ed, px, py):
        scent_radius = 150.0
        scent_set = self._scent_set
        self.spatial_hash.query_nearby_excluding_into(px, py, scent_radius, eid, scent_set)
        best_dist_sq = 1e18
        best_dx, best_dy = 0.0, 0.0
        for nid in scent_set:
            n_idx = ed.eid_to_idx.get(nid)
            if n_idx is None or not ed.alive[n_idx]:
                continue
            nd = int(ed.diet_type[n_idx])
            if nd == 2 or nd == 3:
                continue
            ddx = float(ed.x[n_idx]) - px
            ddy = float(ed.y[n_idx]) - py
            dsq = ddx * ddx + ddy * ddy
            if dsq < best_dist_sq:
                best_dist_sq = dsq
                best_dx, best_dy = ddx, ddy
        return best_dx, best_dy

    def _update_ecs(self, world: object) -> None:
        for eid in self.em.get_entities_with(Position, Velocity, Energy, Sensor):
            pos = self.em.get_component(eid, Position)
            vel = self.em.get_component(eid, Velocity)
            energy = self.em.get_component(eid, Energy)
            sensor = self.em.get_component(eid, Sensor)
            if pos is None or vel is None or energy is None or sensor is None:
                continue

            diet = self.em.get_component(eid, Diet)
            if diet and diet.diet_type == DietType.CARNIVOROUS_PLANT:
                vel.dx = 0.0
                vel.dy = 0.0
                continue

            genome_comp = self.em.get_component(eid, GenomeComp)
            repro = self.em.get_component(eid, Reproduction)

            max_speed = self._max_speed(genome_comp)

            target_dx, target_dy = 0.0, 0.0
            action = self._decide(eid, energy, sensor, diet, repro, genome_comp)

            speed_mult = 2.5 if action == "hunt" else 1.0

            if action == "flee":
                target_dx, target_dy = self._flee(pos, sensor)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            elif action == "hunt":
                target_dx, target_dy = self._hunt(pos, sensor, eid)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            elif action == "reproduce":
                target_dx, target_dy = self._seek_mate(pos, sensor, eid, diet)
            elif action == "eat":
                target_dx, target_dy = self._seek_food(pos, sensor, world)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            else:
                if diet and diet.diet_type == DietType.PREDATOR and self.spatial_hash is not None:
                    scent_dx, scent_dy = self._scent_hunt_ecs(pos, eid)
                    if scent_dx != 0.0 or scent_dy != 0.0:
                        target_dx, target_dy = scent_dx, scent_dy
                    else:
                        target_dx = random.uniform(-1, 1)
                        target_dy = random.uniform(-1, 1)
                else:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)

            if target_dx != 0 or target_dy != 0:
                mag = math.sqrt(target_dx ** 2 + target_dy ** 2)
                if mag > 0:
                    target_dx /= mag
                    target_dy /= mag

            vel.dx = target_dx * max_speed * speed_mult
            vel.dy = target_dy * max_speed * speed_mult

    def _scent_hunt_ecs(self, pos, eid):
        scent_radius = 150.0
        scent_set = self._scent_set
        self.spatial_hash.query_nearby_excluding_into(pos.x, pos.y, scent_radius, eid, scent_set)
        best_dist_sq = 1e18
        best_dx, best_dy = 0.0, 0.0
        for nid in scent_set:
            n_diet = self.em.get_component(nid, Diet)
            if n_diet is None or n_diet.diet_type == DietType.PREDATOR:
                continue
            n_pos = self.em.get_component(nid, Position)
            if n_pos is None:
                continue
            ddx = n_pos.x - pos.x
            ddy = n_pos.y - pos.y
            dsq = ddx * ddx + ddy * ddy
            if dsq < best_dist_sq:
                best_dist_sq = dsq
                best_dx, best_dy = ddx, ddy
        return best_dx, best_dy

    def _max_speed(self, genome_comp: GenomeComp | None) -> float:
        if genome_comp and genome_comp.genome:
            return 1.0 + genome_comp.genome.speed * 4.0
        return 2.0

    def _decide(self, eid, energy, sensor, diet, repro, genome_comp):
        low_energy = energy.current < energy.max_value * 0.3

        if diet and diet.diet_type == DietType.PREDATOR:
            for nb in sensor.nearby_entities[:8]:
                if nb.diet_type is not None and nb.diet_type != DietType.PREDATOR:
                    return "hunt"

        if diet and diet.diet_type in (DietType.HERBIVORE, DietType.OMNIVORE):
            for nb in sensor.nearby_entities[:5]:
                if nb.diet_type == DietType.PREDATOR:
                    return "flee"

        if low_energy:
            if diet and diet.diet_type != DietType.PREDATOR:
                return "eat"

        if repro and repro.repro_type in ("sexual", "hermaphrodite") and repro.cooldown <= 0:
            repro_threshold = repro.threshold * energy.max_value
            if energy.current > repro_threshold:
                return "reproduce"

        return "wander"

    def _flee(self, pos, sensor):
        dx, dy = 0.0, 0.0
        for nb in sensor.nearby_entities[:5]:
            if nb.diet_type == DietType.PREDATOR:
                npos = self.em.get_component(nb.eid, Position)
                if npos:
                    dx += pos.x - npos.x
                    dy += pos.y - npos.y
        return dx, dy

    def _hunt(self, pos, sensor, eid):
        for nb in sensor.nearby_entities:
            if nb.diet_type is not None and nb.diet_type != DietType.PREDATOR:
                npos = self.em.get_component(nb.eid, Position)
                if npos:
                    return npos.x - pos.x, npos.y - pos.y
        return 0.0, 0.0

    def _seek_food(self, pos, sensor, world=None):
        if world is not None:
            w = world
            ix, iy = int(pos.x), int(pos.y)
            if 0 <= ix < w.width and 0 <= iy < w.height:
                if w.food_values[iy, ix] > 5.0:
                    return 0.0, 0.0
        if sensor.nearest_food_pos:
            fx, fy = sensor.nearest_food_pos
            return fx - pos.x, fy - pos.y
        return 0.0, 0.0

    def _seek_mate(self, pos, sensor, eid, diet):
        my_diet = diet.diet_type if diet else DietType.HERBIVORE
        for nb in sensor.nearby_entities:
            if nb.diet_type == my_diet:
                n_repro = self.em.get_component(nb.eid, Reproduction)
                if n_repro is None or n_repro.cooldown <= 0:
                    npos = self.em.get_component(nb.eid, Position)
                    if npos:
                        return npos.x - pos.x, npos.y - pos.y
        return 0.0, 0.0
