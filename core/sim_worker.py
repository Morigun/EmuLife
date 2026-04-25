from __future__ import annotations

import random

import numpy as np
from multiprocessing import shared_memory, Value, Pipe

from config import Config
from core.ecs import EntityManager
from core.world import World
from core.genome import Genome
from core.entity_data import EntityData
from utils.spatial_hash import SpatialHash
from utils.logger import get_logger

from systems.sensor_system import SensorSystem
from systems.condition_system import ConditionSystem
from systems.behavior import BehaviorSystem
from systems.movement import MovementSystem
from systems.interaction import InteractionSystem
from systems.energy_system import EnergySystem
from systems.aging import AgingSystem
from systems.reproduction import ReproductionSystem, create_organism
from systems.death import DeathSystem
from systems.world_update import WorldUpdateSystem
from systems.abiogenesis import AbiogenesisSystem
from systems.day_night_system import DayNightSystem

from components.position import Position

from core.shared_buffers import SharedEntityBuffer


class SimWorker:
    def __init__(
        self,
        config: Config,
        shared_buf_name: str,
        frame_counter: Value,
        control_pipe,
    ) -> None:
        self.config = config
        self.logger = get_logger()

        self.em = EntityManager()
        self.world = World(self.config)
        self.spatial_hash = SpatialHash(self.config.spatial_hash.cell_size, self.config.world.width, self.config.world.height)
        self.entity_data = EntityData()

        self.sensor_system = SensorSystem(self.em, self.spatial_hash, self.config, self.entity_data)
        self.movement_system = MovementSystem(self.em, self.config, self.entity_data)
        self.reproduction_system = ReproductionSystem(self.em, self.spatial_hash, self.config, self.entity_data)
        self.abiogenesis_system = AbiogenesisSystem(self.em, self.spatial_hash, self.config, self.entity_data)

        self.behavior_system = BehaviorSystem(self.em, self.config, self.entity_data, self.spatial_hash)
        self.behavior_system._sensor_system = self.sensor_system

        self.systems = [
            DayNightSystem(self.config),
            self.sensor_system,
            ConditionSystem(self.em, self.config, self.entity_data),
            self.behavior_system,
            self.movement_system,
            InteractionSystem(self.em, self.config, self.spatial_hash, self.entity_data),
            EnergySystem(self.em, self.config, self.entity_data),
            AgingSystem(self.em, self.config, self.entity_data),
            self.reproduction_system,
            DeathSystem(self.em, self.config, self.spatial_hash, self.entity_data),
            self.abiogenesis_system,
            WorldUpdateSystem(self.config),
        ]

        self.tick = 0
        self.frame_counter = frame_counter
        self.control_pipe = control_pipe

        self._spawn_initial_population()
        self.entity_data.sync_from_ecs(self.em)

        self.shared_buf = SharedEntityBuffer(name=shared_buf_name, create=False)
        self._push_frame()

        from utils.numba_kernels import HAS_NUMBA
        if HAS_NUMBA:
            from utils.numba_kernels import warmup
            warmup()

        self.logger.info("SimWorker initialized")

    def _spawn_initial_population(self) -> None:
        attempts = 0
        spawned = 0
        target = self.config.simulation.initial_population

        while spawned < target and attempts < target * 10:
            x = random.uniform(10, self.config.world.width - 10)
            y = random.uniform(10, self.config.world.height - 10)

            genome = Genome.random_instance(self.config)
            habitat_val = genome.habitat
            is_water = not self.world.is_walkable(x, y)

            if habitat_val < 0.33:
                if not is_water:
                    attempts += 1
                    continue
            else:
                if is_water:
                    attempts += 1
                    continue

            eid = create_organism(
                self.em, genome, x, y, self.config,
                energy_fraction=0.7, parent_energy_sum=200.0,
            )
            pos = self.em.get_component(eid, Position)
            if pos:
                self.spatial_hash.insert(eid, pos.x, pos.y)
            spawned += 1
            attempts += 1

        self.logger.info(f"SimWorker spawned {spawned} organisms")

    def _update_spatial_hash_incremental(self) -> None:
        ed = self.entity_data
        moved_mask = self.movement_system.moved_mask
        sh = self.spatial_hash
        if moved_mask is not None and len(moved_mask) > 0:
            indices = np.where(moved_mask)[0]
            old_x_arr = self.movement_system.old_x
            old_y_arr = self.movement_system.old_y
            ed_x = ed.x
            ed_y = ed.y
            idx_to_eid = ed.idx_to_eid
            entity_cells = sh._entity_cells

            for idx in indices:
                idx_int = int(idx)
                eid = idx_to_eid.get(idx_int)
                if eid is None:
                    continue
                if eid in entity_cells:
                    ox = float(old_x_arr[idx_int])
                    oy = float(old_y_arr[idx_int])
                    nx = float(ed_x[idx_int])
                    ny = float(ed_y[idx_int])
                    old_cx = int(ox // sh.cell_size)
                    old_cy = int(oy // sh.cell_size)
                    new_cx = int(nx // sh.cell_size)
                    new_cy = int(ny // sh.cell_size)
                    if old_cx != new_cx or old_cy != new_cy:
                        sh.remove(eid)
                        sh.insert(eid, nx, ny)
                else:
                    sh.insert(eid, float(ed_x[idx_int]), float(ed_y[idx_int]))

        for eid, x, y in self.reproduction_system.newborn_entities:
            sh.insert(eid, x, y)

    def _push_frame(self) -> None:
        self.shared_buf.write_from_entity_data(self.entity_data)
        self.shared_buf.tick = self.tick
        with self.frame_counter.get_lock():
            self.frame_counter.value += 1

    def run(self) -> None:
        sim_speed = self.config.simulation.simulation_speed
        paused = False
        self.logger.info("SimWorker running")

        while True:
            while self.control_pipe.poll():
                msg = self.control_pipe.recv()
                if msg is None:
                    self.shared_buf.close()
                    self.logger.info("SimWorker stopped")
                    return
                cmd = msg.get("cmd")
                if cmd == "pause":
                    paused = msg.get("value", True)
                elif cmd == "speed":
                    sim_speed = msg.get("value", sim_speed)

            if not paused:
                sim_dt = sim_speed
                self._update_spatial_hash_incremental()

                for system in self.systems:
                    system.update(self.world, sim_dt)

                self.tick += 1
                self._push_frame()
