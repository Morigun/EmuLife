from __future__ import annotations

import json
import os
import random
import multiprocessing
import struct
from types import SimpleNamespace

from utils.cpu_limit import limit_numba_threads, set_affinity, set_affinity_for_process

limit_numba_threads()

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

from rendering.renderer import Renderer
from rendering.ui import UI
from rendering.minimap import Minimap

from components.position import Position
from components.diet import DietType

from utils.numba_kernels import compute_stats_kernel, HAS_NUMBA


def _run_sim_worker(config, shared_buf_name, frame_counter, control_pipe):
    from utils.cpu_limit import set_affinity
    set_affinity()
    from core.sim_worker import SimWorker
    worker = SimWorker(config, shared_buf_name, frame_counter, control_pipe)
    worker.run()


class Simulation:
    def __init__(self) -> None:
        n_cores = set_affinity()
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
        self.death_system = DeathSystem(self.em, self.config, self.spatial_hash, self.entity_data)

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
            self.death_system,
            self.abiogenesis_system,
            WorldUpdateSystem(self.config),
        ]

        self.tick = 0
        self.running = True
        self.paused = False
        self.sim_speed = self.config.simulation.simulation_speed
        self._cumulative = {
            "abiogenesis": 0,
            "births_asexual": 0,
            "births_sexual": 0,
            "deaths": 0,
        }

        self.renderer = Renderer(self.screen, self.config)
        self.ui = UI(self.screen, self.config)
        self.ui.em = self.em
        self.ui.entity_data = self.entity_data
        self.ui.cumulative = self._cumulative
        self.minimap = Minimap(self.screen, self.config)

        self._spawn_initial_population()

        self.entity_data.sync_from_ecs(self.em)

        if HAS_NUMBA:
            from utils.numba_kernels import warmup
            warmup()

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
                origin=0,
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
        click_radius_world = 10.0 / self.camera.zoom
        best_dist_sq = click_radius_world * click_radius_world

        query_radius = max(5.0, 15.0 / self.camera.zoom)
        nearby = self.spatial_hash.query_nearby(wx, wy, query_radius)
        ed = self.entity_data
        for eid in nearby:
            idx = ed.eid_to_idx.get(eid)
            if idx is None or not ed.alive[idx]:
                continue
            dx = float(ed.x[idx]) - wx
            dy = float(ed.y[idx]) - wy
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

    def _save_lifetime_stats(self) -> None:
        c = self._cumulative
        tick = max(self.tick, 1)
        ed = self.entity_data
        n = ed.count
        alive = ed.alive[:n]

        data = {
            "tick": tick,
            "all_time": {
                "abiogenesis": c["abiogenesis"],
                "births_asexual": c["births_asexual"],
                "births_sexual": c["births_sexual"],
                "deaths": c["deaths"],
                "total_spawned": c["abiogenesis"] + c["births_asexual"] + c["births_sexual"],
            },
            "per_tick_avg": {
                "abiogenesis": round(c["abiogenesis"] / tick, 3),
                "births_asexual": round(c["births_asexual"] / tick, 3),
                "births_sexual": round(c["births_sexual"] / tick, 3),
                "deaths": round(c["deaths"] / tick, 3),
            },
            "alive_now": {
                "total": int(np.sum(alive)),
                "asexual_repro": int(np.sum(alive & (ed.repro_type[:n] == 0))),
                "sexual_repro": int(np.sum(alive & (ed.repro_type[:n] == 1))),
                "hermaphrodite_repro": int(np.sum(alive & (ed.repro_type[:n] == 2))),
                "origin_abio": int(np.sum(alive & (ed.origin[:n] == 0))),
                "origin_born_asex": int(np.sum(alive & (ed.origin[:n] == 1))),
                "origin_born_sex": int(np.sum(alive & (ed.origin[:n] == 2))),
            },
        }
        with open("stats.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

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

                self._cumulative["abiogenesis"] += self.abiogenesis_system.spawns_tick
                self._cumulative["births_asexual"] += self.reproduction_system.births_asexual_tick
                self._cumulative["births_sexual"] += self.reproduction_system.births_sexual_tick
                self._cumulative["deaths"] += self.death_system.deaths_tick

                if self.tick % 100 == 0:
                    self._save_lifetime_stats()

            self.screen.fill((20, 20, 30))
            self.renderer.render_world(self.world, self.camera)
            self.renderer.render_night_overlay(self.world)

            for lx, ly in self.abiogenesis_system.lightning_events:
                sx, sy = self.camera.world_to_screen(float(lx), float(ly))
                radius = int(10 * self.camera.zoom)
                if radius > 0:
                    lightning_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(lightning_surf, (255, 255, 200, 120), (radius, radius), radius)
                    self.screen.blit(lightning_surf, (int(sx) - radius, int(sy) - radius))

            self.renderer.render_entities_from_soa(self.entity_data, self.camera)
            self.renderer.render_selected(self.em, self.camera, self.ui.selected_entity, self.entity_data)

            self.ui.render_stats(self.em, fps, self.tick, self.sim_speed, self.world)
            self.ui.render_selected_info_soa(self.entity_data, self.ui.selected_entity, self.em)
            self.ui.render_help()
            self.minimap.render(self.world, self.camera, self.em, self.tick, self.entity_data)

            pygame.display.flip()

        pygame.quit()
        self.logger.info("EmuLife stopped")


def main() -> None:
    import sys
    use_mp = "--mp" in sys.argv
    if use_mp:
        sim = SimulationMP()
    else:
        sim = Simulation()
    sim.run()


class _SoadProxy:
    def __init__(self, data: dict) -> None:
        n = data["count"]
        for key, val in data.items():
            setattr(self, key, val)
        self.count = n


class SimulationMP:
    def __init__(self) -> None:
        self.config = Config()
        self.logger = get_logger()

        from core.shared_buffers import SharedEntityBuffer

        self.shared_buf = SharedEntityBuffer(create=True)
        self.frame_counter = multiprocessing.Value("i", 0)
        parent_conn, child_conn = multiprocessing.Pipe()
        self.control_pipe = parent_conn

        self.sim_process = multiprocessing.Process(
            target=_run_sim_worker,
            args=(self.config, self.shared_buf.shm_name, self.frame_counter, child_conn),
            daemon=True,
        )

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.screen.width, self.config.screen.height)
        )
        pygame.display.set_caption("EmuLife — Цифровая Генетическая Экосистема [MP]")
        self.clock = pygame.time.Clock()

        self.world = World(self.config)
        self.camera = Camera(
            x=self.config.world.width / 2,
            y=self.config.world.height / 2,
            zoom=1.0,
            screen_width=self.config.screen.width,
            screen_height=self.config.screen.height,
        )

        self.renderer = Renderer(self.screen, self.config)
        self.ui = UI(self.screen, self.config)
        self.minimap = Minimap(self.screen, self.config)

        self.running = True
        self.paused = False
        self.sim_speed = self.config.simulation.simulation_speed
        self.last_frame = 0
        self.selected_entity: int | None = None
        self._last_proxy: _SoadProxy | None = None

        n_cores = set_affinity()

        self.sim_process.start()

        try:
            set_affinity_for_process(self.sim_process.pid, n_cores)
        except Exception:
            pass

        self.logger.info(f"EmuLife MP initialized ({n_cores}/{os.cpu_count()} cores)")

    def _read_frame(self) -> _SoadProxy | None:
        if self.frame_counter.value == self.last_frame:
            return None
        self.last_frame = self.frame_counter.value
        data = self.shared_buf.read_into_arrays()
        return _SoadProxy(data)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.control_pipe.send({"cmd": "pause", "value": self.paused})
                elif event.key == pygame.K_h:
                    self.ui.show_stats = not self.ui.show_stats
                elif event.key == pygame.K_m:
                    self.minimap.visible = not self.minimap.visible
                elif event.key == pygame.K_f:
                    if self.selected_entity is not None:
                        if self.camera.follow_entity == self.selected_entity:
                            self.camera.follow_entity = None
                        else:
                            self.camera.follow_entity = self.selected_entity
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.sim_speed = min(10.0, self.sim_speed + 0.5)
                    self.control_pipe.send({"cmd": "speed", "value": self.sim_speed})
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.sim_speed = max(0.5, self.sim_speed - 0.5)
                    self.control_pipe.send({"cmd": "speed", "value": self.sim_speed})

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
        proxy = self._last_proxy
        if proxy is None:
            new_proxy = self._read_frame()
            if new_proxy is not None:
                proxy = new_proxy
                self._last_proxy = proxy
            else:
                return

        n = proxy.count
        if n == 0:
            self.selected_entity = None
            return

        dx = proxy.x - wx
        dy = proxy.y - wy
        dist_sq = dx * dx + dy * dy
        dist_sq[~proxy.alive] = 1e18
        best_idx = int(np.argmin(dist_sq))

        click_radius_world = 10.0 / self.camera.zoom
        if dist_sq[best_idx] < click_radius_world * click_radius_world:
            self.selected_entity = int(proxy.eids[best_idx])
        else:
            self.selected_entity = None
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

    def _render_stats_from_soa(self, proxy: _SoadProxy, fps: float, tick: int) -> None:
        if not self.ui.show_stats:
            return

        n = proxy.count
        if HAS_NUMBA:
            herb, omni, pred, plant = compute_stats_kernel(proxy.diet_type, proxy.alive, n)
        else:
            alive = proxy.alive
            herb = int(np.sum(alive & (proxy.diet_type == 0)))
            omni = int(np.sum(alive & (proxy.diet_type == 1)))
            pred = int(np.sum(alive & (proxy.diet_type == 2)))
            plant = int(np.sum(alive & (proxy.diet_type == 3)))
        total = herb + omni + pred + plant

        alive = proxy.alive
        asex = int(np.sum(alive & (proxy.repro_type == 0)))
        sex = int(np.sum(alive & (proxy.repro_type == 1)))
        herma = int(np.sum(alive & (proxy.repro_type == 2)))
        abio = int(np.sum(alive & (proxy.origin == 0)))
        birth_asex = int(np.sum(alive & (proxy.origin == 1)))
        birth_sex = int(np.sum(alive & (proxy.origin == 2)))

        lines = [
            f"FPS: {fps:.0f}  Tick: {tick}  Speed: {self.sim_speed:.1f}x [MP]",
            f"Total: {total}  Herb: {herb}  Omni: {omni}  Pred: {pred}  Plant: {plant}",
            f"Asex: {asex}  Sex: {sex}  Herm: {herma} [MP]",
            f"Abio: {abio}  Born(a): {birth_asex}  Born(s): {birth_sex}",
            f"Max pop: {self.config.simulation.max_population}",
        ]

        y = 5
        for line in lines:
            surf = self.ui.font_small.render(line, True, (255, 255, 255))
            bg = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            self.screen.blit(bg, (4, y))
            self.screen.blit(surf, (7, y + 1))
            y += 18

    def _render_selected_from_soa(self, proxy: _SoadProxy) -> None:
        if self.selected_entity is None:
            return

        eid_matches = np.where(proxy.eids[:proxy.count] == self.selected_entity)[0]
        if len(eid_matches) == 0:
            self.selected_entity = None
            return

        idx = int(eid_matches[0])
        if not proxy.alive[idx]:
            self.selected_entity = None
            return

        from utils.species_namer import get_species_name_from_soa_data, get_diet_name, get_repro_name, get_habitat_name, get_origin_name

        species = get_species_name_from_soa_data(
            diet_type=int(proxy.diet_type[idx]),
            repro_type=int(proxy.repro_type[idx]),
            habitat=int(proxy.habitat[idx]),
            size_gene=float(proxy.size_gene[idx]),
            speed_gene=float(proxy.speed_gene[idx]),
            aggression=float(proxy.aggression[idx]),
        )

        diet_name = get_diet_name(int(proxy.diet_type[idx]))
        repro_name = get_repro_name(int(proxy.repro_type[idx]))
        habitat_name = get_habitat_name(int(proxy.habitat[idx]))

        lines = [
            f"{species} #{self.selected_entity}",
            f"Energy: {proxy.energy[idx]:.1f}/{proxy.max_energy[idx]:.1f}",
            f"Health: {proxy.health[idx]:.1f}/{proxy.max_health[idx]:.1f}",
            f"Age: {proxy.age[idx]}/{proxy.max_age[idx]}",
            f"Origin: {get_origin_name(int(proxy.origin[idx]))}",
            f"Diet: {diet_name}",
            f"Repro: {repro_name}",
            f"Habitat: {habitat_name}",
            f"Aggression: {proxy.aggression[idx]:.2f}",
            f"Vision: {proxy.vision[idx]:.1f}",
            f"Metabolism: {proxy.metabolism[idx]:.2f}",
            f"Pos: ({proxy.x[idx]:.0f}, {proxy.y[idx]:.0f})",
        ]

        if int(proxy.diet_type[idx]) == 3:
            lines.append(f"Photosynth: {proxy.photosynth[idx]:.2f}")
            lines.append(f"Trap power: {proxy.aggression[idx]:.2f}")

        lines.append(
            f"Parent: #{int(proxy.parent_eid[idx])}" if proxy.parent_eid[idx] >= 0 else "Parent: none"
        )

        panel_w = 300
        panel_h = len(lines) * 18 + 10
        panel_x = self.screen.get_width() - panel_w - 10
        panel_y = 5

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        self.screen.blit(bg, (panel_x, panel_y))

        for i, line in enumerate(lines):
            surf = self.ui.font_small.render(line, True, (255, 255, 200))
            self.screen.blit(surf, (panel_x + 5, panel_y + 5 + i * 18))

    def run(self) -> None:
        proxy = None

        while self.running:
            real_dt = self.clock.tick(self.config.screen.fps_cap) / 1000.0
            fps = self.clock.get_fps()

            self._handle_events()
            self._handle_input(real_dt)

            new_proxy = self._read_frame()
            if new_proxy is not None:
                proxy = new_proxy
                self._last_proxy = proxy
                if self.camera.follow_entity is not None:
                    follow_eid = self.camera.follow_entity
                    eid_matches = np.where(proxy.eids[:proxy.count] == follow_eid)[0]
                    if len(eid_matches) > 0:
                        idx = int(eid_matches[0])
                        if proxy.alive[idx]:
                            self.camera.x = float(proxy.x[idx])
                            self.camera.y = float(proxy.y[idx])
                        else:
                            self.camera.follow_entity = None
                    else:
                        self.camera.follow_entity = None

            self.screen.fill((20, 20, 30))
            self.renderer.render_world(self.world, self.camera)

            if proxy is not None:
                self.renderer.render_entities_from_soa(proxy, self.camera)

                if self.selected_entity is not None:
                    eid_matches = np.where(proxy.eids[:proxy.count] == self.selected_entity)[0]
                    if len(eid_matches) > 0:
                        idx = int(eid_matches[0])
                        if proxy.alive[idx]:
                            sx, sy = self.camera.world_to_screen(float(proxy.x[idx]), float(proxy.y[idx]))
                            pygame.draw.circle(self.screen, (255, 255, 0), (int(sx), int(sy)), 15, 2)
                        else:
                            self.selected_entity = None
                    else:
                        self.selected_entity = None

                tick = proxy.tick
                self._render_stats_from_soa(proxy, fps, tick)
                self._render_selected_from_soa(proxy)
                self.minimap.render(self.world, self.camera, None, tick, proxy)
            else:
                self.ui.render_stats(EntityManager(), fps, 0, self.sim_speed)

            self.ui.render_help()
            pygame.display.flip()

        self.control_pipe.send(None)
        self.sim_process.join(timeout=5)
        if self.sim_process.is_alive():
            self.sim_process.terminate()
        self.shared_buf.unlink()
        pygame.quit()
        self.logger.info("EmuLife MP stopped")


if __name__ == "__main__":
    main()
