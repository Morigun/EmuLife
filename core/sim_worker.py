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
        self.spatial_hash = SpatialHash(self.config.spatial_hash.cell_size)
        self.entity_data = EntityData()

        self.movement_system = MovementSystem(self.em, self.config, self.entity_data)
        self.reproduction_system = ReproductionSystem(self.em, self.spatial_hash, self.config, self.entity_data)
        self.abiogenesis_system = AbiogenesisSystem(self.em, self.spatial_hash, self.config, self.entity_data)

        self.systems = [
            DayNightSystem(self.config),
            SensorSystem(self.em, self.spatial_hash, self.config, self.entity_data),
            ConditionSystem(self.em, self.config, self.entity_data),
            BehaviorSystem(self.em, self.config, self.entity_data, self.spatial_hash),
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
        if moved_mask is not None and len(moved_mask) > 0:
            indices = np.where(moved_mask)[0]
            for idx in indices:
                eid = ed.idx_to_eid.get(int(idx))
                if eid is None:
                    continue
                old_x = float(self.movement_system.old_x[idx])
                old_y = float(self.movement_system.old_y[idx])
                new_x = float(ed.x[idx])
                new_y = float(ed.y[idx])
                if eid in self.spatial_hash._entity_cells:
                    self.spatial_hash.update(eid, old_x, old_y, new_x, new_y)
                else:
                    self.spatial_hash.insert(eid, new_x, new_y)

        for eid, x, y in self.reproduction_system.newborn_entities:
            self.spatial_hash.insert(eid, x, y)

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
