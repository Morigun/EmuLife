from __future__ import annotations

import random

import numpy as np
import pygame

from config import Config
from core.ecs import EntityManager
from core.world import World
from core.camera import Camera
from core.genome import Genome
from core.entity_data import EntityData
from utils.spatial_hash import SpatialHash
from utils.logger import get_logger

from systems.sensor_system import SensorSystem
from systems.behavior import BehaviorSystem
from systems.movement import MovementSystem
from systems.interaction import InteractionSystem
from systems.energy_system import EnergySystem
from systems.aging import AgingSystem
from systems.reproduction import ReproductionSystem, create_organism
from systems.death import DeathSystem
from systems.world_update import WorldUpdateSystem
from systems.abiogenesis import AbiogenesisSystem

from rendering.renderer import Renderer
from rendering.ui import UI
from rendering.minimap import Minimap

from components.position import Position


class Simulation:
    def __init__(self) -> None:
        self.config = Config()
        self.logger = get_logger()

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.screen.width, self.config.screen.height)
        )
        pygame.display.set_caption("EmuLife — Цифровая Генетическая Экосистема")
        self.clock = pygame.time.Clock()

        self.em = EntityManager()
        self.world = World(self.config)
        self.spatial_hash = SpatialHash(self.config.spatial_hash.cell_size)
        self.entity_data = EntityData()
        self.camera = Camera(
            x=self.config.world.width / 2,
            y=self.config.world.height / 2,
            zoom=1.0,
            screen_width=self.config.screen.width,
            screen_height=self.config.screen.height,
        )

        self.movement_system = MovementSystem(self.em, self.config, self.entity_data)
        self.reproduction_system = ReproductionSystem(self.em, self.spatial_hash, self.config, self.entity_data)
        self.abiogenesis_system = AbiogenesisSystem(self.em, self.spatial_hash, self.config, self.entity_data)

        self.systems = [
            SensorSystem(self.em, self.spatial_hash, self.config, self.entity_data),
            BehaviorSystem(self.em, self.config, self.entity_data),
            self.movement_system,
            InteractionSystem(self.em, self.config, self.spatial_hash, self.entity_data),
            EnergySystem(self.em, self.config, self.entity_data),
            AgingSystem(self.em, self.config, self.entity_data),
            self.reproduction_system,
            DeathSystem(self.em, self.config, self.spatial_hash, self.entity_data),
            self.abiogenesis_system,
            WorldUpdateSystem(self.config),
        ]

        self.renderer = Renderer(self.screen, self.config)
        self.ui = UI(self.screen, self.config)
        self.minimap = Minimap(self.screen, self.config)

        self.tick = 0
        self.running = True
        self.paused = False
        self.sim_speed = self.config.simulation.simulation_speed

        self._spawn_initial_population()

        self.entity_data.sync_from_ecs(self.em)

        self.logger.info("EmuLife initialized")

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

        self.logger.info(f"Spawned {spawned} organisms")

    def _rebuild_spatial_hash(self) -> None:
        self.spatial_hash.clear()
        for eid in self.em.get_entities_with(Position):
            pos = self.em.get_component(eid, Position)
            if pos:
                self.spatial_hash.insert(eid, pos.x, pos.y)

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

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_h:
                    self.ui.show_stats = not self.ui.show_stats
                elif event.key == pygame.K_m:
                    self.minimap.visible = not self.minimap.visible
                elif event.key == pygame.K_f:
                    if self.ui.selected_entity is not None:
                        if self.camera.follow_entity == self.ui.selected_entity:
                            self.camera.follow_entity = None
                        else:
                            self.camera.follow_entity = self.ui.selected_entity
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                    self.sim_speed = min(10.0, self.sim_speed + 0.5)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self.sim_speed = max(0.5, self.sim_speed - 0.5)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    wx, wy = self.camera.screen_to_world(event.pos[0], event.pos[1])
                    self._select_entity_at(wx, wy)
                elif event.button == 4:
                    self.camera.zoom_at(1.15, event.pos[0], event.pos[1])
                elif event.button == 5:
                    self.camera.zoom_at(1.0 / 1.15, event.pos[0], event.pos[1])

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if event.y > 0:
                    self.camera.zoom_at(1.15, mx, my)
                elif event.y < 0:
                    self.camera.zoom_at(1.0 / 1.15, mx, my)

    def _select_entity_at(self, wx: float, wy: float) -> None:
        best_id = None
        best_dist_sq = 225.0

        nearby = self.spatial_hash.query_nearby(wx, wy, 15.0)
        for eid in nearby:
            pos = self.em.get_component(eid, Position)
            if pos is None:
                continue
            dx = pos.x - wx
            dy = pos.y - wy
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_id = eid

        self.ui.selected_entity = best_id
        if best_id is None:
            self.camera.follow_entity = None

    def _handle_input(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        dx, dy = 0.0, 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += 1
        if dx != 0 or dy != 0:
            self.camera.follow_entity = None
            self.camera.pan(dx, dy, dt)

    def run(self) -> None:
        while self.running:
            real_dt = self.clock.tick(self.config.screen.fps_cap) / 1000.0
            fps = self.clock.get_fps()

            self._handle_events()
            self._handle_input(real_dt)

            self.camera.update(self.em, real_dt, self.entity_data)

            if not self.paused:
                sim_dt = self.sim_speed
                self._update_spatial_hash_incremental()

                for system in self.systems:
                    system.update(self.world, sim_dt)

                self.tick += 1

            self.screen.fill((20, 20, 30))
            self.renderer.render_world(self.world, self.camera)

            for lx, ly in self.abiogenesis_system.lightning_events:
                sx, sy = self.camera.world_to_screen(float(lx), float(ly))
                radius = int(10 * self.camera.zoom)
                if radius > 0:
                    lightning_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(lightning_surf, (255, 255, 200, 120), (radius, radius), radius)
                    self.screen.blit(lightning_surf, (int(sx) - radius, int(sy) - radius))

            self.renderer.render_entities_from_soa(self.entity_data, self.camera)
            self.renderer.render_selected(self.em, self.camera, self.ui.selected_entity, self.entity_data)

            self.ui.render_stats(self.em, fps, self.tick, self.sim_speed)
            self.ui.render_selected_info(self.em)
            self.ui.render_help()
            self.minimap.render(self.world, self.camera, self.em, self.tick, self.entity_data)

            pygame.display.flip()

        pygame.quit()
        self.logger.info("EmuLife stopped")


def main() -> None:
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
