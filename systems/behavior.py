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
from config import Config

STAGGER_BATCHES = 2


class BehaviorSystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.config = config
        self.entity_data = entity_data
        self.tick = 0

    def update(self, world: object, dt: float) -> None:
        self.tick += 1
        if self.entity_data is not None:
            self._update_staggered(world, dt)
        else:
            self._update_ecs()

    def _update_staggered(self, world: object, dt: float) -> None:
        ed = self.entity_data
        current_batch = self.tick % STAGGER_BATCHES
        n = ed.count

        for idx in range(n):
            if not ed.alive[idx]:
                continue
            if idx % STAGGER_BATCHES != current_batch:
                continue

            eid = ed.idx_to_eid.get(idx)
            if eid is None:
                continue

            sensor = self.em.get_component(eid, Sensor)
            if sensor is None:
                continue

            px, py = ed.x[idx], ed.y[idx]
            max_speed = float(ed.speed_gene[idx])

            target_dx, target_dy = 0.0, 0.0
            action = self._decide_soa(idx, eid, ed, sensor)

            if action == "flee":
                target_dx, target_dy = self._flee_soa(idx, sensor, ed)
            elif action == "hunt":
                target_dx, target_dy = self._hunt_soa(idx, sensor, ed)
            elif action == "reproduce":
                target_dx, target_dy = self._seek_mate_soa(idx, eid, sensor, ed)
                if target_dx == 0.0 and target_dy == 0.0:
                    target_dx = random.uniform(-1, 1)
                    target_dy = random.uniform(-1, 1)
            elif action == "eat":
                target_dx, target_dy = self._seek_food_soa(idx, sensor, ed, world)
                if target_dx == 0.0 and target_dy == 0.0:
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

            ed.dx[idx] = target_dx * max_speed
            ed.dy[idx] = target_dy * max_speed

            vel = self.em.get_component(eid, Velocity)
            if vel:
                vel.dx = target_dx * max_speed
                vel.dy = target_dy * max_speed

    def _decide_soa(self, idx, eid, ed, sensor):
        low_energy = ed.energy[idx] < ed.max_energy[idx] * 0.3
        diet_int = int(ed.diet_type[idx])

        if diet_int == 2:
            for nb in sensor.nearby_entities[:3]:
                if nb.diet_type is not None and nb.diet_type != DietType.PREDATOR:
                    if low_energy:
                        return "hunt"
                    break

        if diet_int in (0, 1):
            for nb in sensor.nearby_entities[:5]:
                if nb.diet_type == DietType.PREDATOR:
                    return "flee"

        if low_energy:
            return "eat"

        repro_type_int = int(ed.repro_type[idx])
        if repro_type_int == 1:
            repro_cooldown = ed.repro_cooldown[idx]
            if repro_cooldown <= 0:
                if ed.energy[idx] > ed.max_energy[idx] * 0.8:
                    return "reproduce"

        return "wander"

    def _flee_soa(self, idx, sensor, ed):
        px, py = ed.x[idx], ed.y[idx]
        dx, dy = 0.0, 0.0
        for nb in sensor.nearby_entities[:5]:
            if nb.diet_type == DietType.PREDATOR:
                n_idx = ed.eid_to_idx.get(nb.eid)
                if n_idx is not None:
                    dx += px - ed.x[n_idx]
                    dy += py - ed.y[n_idx]
        return dx, dy

    def _hunt_soa(self, idx, sensor, ed):
        px, py = ed.x[idx], ed.y[idx]
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
        if sensor.nearest_food_pos:
            fx, fy = sensor.nearest_food_pos
            return fx - ed.x[idx], fy - ed.y[idx]
        return 0.0, 0.0

    def _seek_mate_soa(self, idx, eid, sensor, ed):
        diet_int = int(ed.diet_type[idx])
        my_diet = {0: DietType.HERBIVORE, 1: DietType.OMNIVORE, 2: DietType.PREDATOR}.get(diet_int, DietType.HERBIVORE)
        for nb in sensor.nearby_entities:
            if nb.diet_type == my_diet:
                n_repro = self.em.get_component(nb.eid, Reproduction)
                if n_repro is None or n_repro.cooldown <= 0:
                    n_idx = ed.eid_to_idx.get(nb.eid)
                    if n_idx is not None:
                        return ed.x[n_idx] - ed.x[idx], ed.y[n_idx] - ed.y[idx]
        return 0.0, 0.0

    def _update_ecs(self) -> None:
        for eid in self.em.get_entities_with(Position, Velocity, Energy, Sensor):
            pos = self.em.get_component(eid, Position)
            vel = self.em.get_component(eid, Velocity)
            energy = self.em.get_component(eid, Energy)
            sensor = self.em.get_component(eid, Sensor)
            if pos is None or vel is None or energy is None or sensor is None:
                continue

            genome_comp = self.em.get_component(eid, GenomeComp)
            diet = self.em.get_component(eid, Diet)
            repro = self.em.get_component(eid, Reproduction)

            max_speed = self._max_speed(genome_comp)

            target_dx, target_dy = 0.0, 0.0
            action = self._decide(eid, energy, sensor, diet, repro, genome_comp)

            if action == "flee":
                target_dx, target_dy = self._flee(pos, sensor)
            elif action == "hunt":
                target_dx, target_dy = self._hunt(pos, sensor, eid)
            elif action == "reproduce":
                target_dx, target_dy = self._seek_mate(pos, sensor, eid, diet)
            elif action == "eat":
                target_dx, target_dy = self._seek_food(pos, sensor, world)
                if target_dx == 0.0 and target_dy == 0.0:
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

            vel.dx = target_dx * max_speed
            vel.dy = target_dy * max_speed

    def _max_speed(self, genome_comp: GenomeComp | None) -> float:
        if genome_comp and genome_comp.genome:
            return 1.0 + genome_comp.genome.speed * 4.0
        return 2.0

    def _decide(self, eid, energy, sensor, diet, repro, genome_comp):
        low_energy = energy.current < energy.max_value * 0.3

        if diet and diet.diet_type == DietType.PREDATOR:
            for nb in sensor.nearby_entities[:3]:
                if nb.diet_type is not None and nb.diet_type != DietType.PREDATOR:
                    if low_energy:
                        return "hunt"
                    break

            for nb in sensor.nearby_entities[:3]:
                if nb.diet_type == DietType.PREDATOR and nb.eid != eid:
                    pass

        if diet and diet.diet_type in (DietType.HERBIVORE, DietType.OMNIVORE):
            for nb in sensor.nearby_entities[:5]:
                if nb.diet_type == DietType.PREDATOR:
                    return "flee"

        if low_energy:
            return "eat"

        if repro and repro.repro_type != "asexual" and repro.cooldown <= 0:
            if energy.current > energy.max_value * 0.8:
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
