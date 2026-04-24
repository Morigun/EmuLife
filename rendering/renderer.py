from __future__ import annotations

import math
import numpy as np
import pygame

from core.ecs import EntityManager
from core.entity_data import EntityData
from core.camera import Camera
from core.world import World, TileType, TILE_COLORS_NP
from components.position import Position
from components.appearance import Appearance
from config import Config


class Renderer:
    def __init__(self, screen: pygame.Surface, config: Config) -> None:
        self.screen = screen
        self.config = config

    def render_world(self, world: World, camera: Camera) -> None:
        left, top, right, bottom = camera.visible_bounds()
        min_x = max(0, int(left) - 1)
        max_x = min(world.width - 1, int(right) + 1)
        min_y = max(0, int(top) - 1)
        max_y = min(world.height - 1, int(bottom) + 1)

        screen_w = max_x - min_x + 1
        screen_h = max_y - min_y + 1
        if screen_w <= 0 or screen_h <= 0:
            return

        max_pixels = 800 * 600
        total_pixels = screen_w * screen_h

        if total_pixels > max_pixels:
            scale = int(np.ceil(np.sqrt(total_pixels / max_pixels)))
            ds_w = (screen_w + scale - 1) // scale
            ds_h = (screen_h + scale - 1) // scale

            indices_y = np.linspace(min_y, max_y, ds_h, dtype=int)
            indices_x = np.linspace(min_x, max_x, ds_w, dtype=int)

            region_types = world.tile_types[np.ix_(indices_y, indices_x)]
            rgb = TILE_COLORS_NP[region_types].copy()

            non_water = region_types != TileType.WATER
            max_f = world.max_foods[np.ix_(indices_y, indices_x)]
            has_food = non_water & (max_f > 0)

            if np.any(has_food):
                food_ratio = np.zeros((ds_h, ds_w), dtype=np.float32)
                valid = max_f > 0
                food_ratio[valid] = world.food_values[np.ix_(indices_y, indices_x)][valid] / max_f[valid]
                brightness = (0.5 + food_ratio * 0.5)[:, :, np.newaxis]
                rgb_f = rgb.astype(np.float32)
                rgb_f[has_food] = rgb_f[has_food] * brightness[has_food]
                rgb = rgb_f.astype(np.uint8)

            biomass_region = world.biomass[np.ix_(indices_y, indices_x)]
            high_biomass = biomass_region > 10.0
            if np.any(high_biomass):
                biomass_intensity = np.clip(biomass_region / 100.0, 0, 1)
                brown = np.zeros_like(rgb)
                brown[:, :, 0] = 60
                brown[:, :, 1] = 30
                brown[:, :, 2] = 10
                mask_3d = high_biomass[:, :, np.newaxis]
                rgb_f = rgb.astype(np.float32)
                rgb_f = rgb_f * (1 - biomass_intensity[:, :, np.newaxis] * 0.5) + brown * biomass_intensity[:, :, np.newaxis] * 0.5
                rgb = rgb_f.astype(np.uint8)

            rgb_t = np.ascontiguousarray(rgb.transpose(1, 0, 2))
            surf = pygame.Surface((rgb_t.shape[0], rgb_t.shape[1]))
            pygame.surfarray.blit_array(surf, rgb_t)

            sx, sy = camera.world_to_screen(float(min_x), float(min_y))
            target_w = int(screen_w * camera.zoom)
            target_h = int(screen_h * camera.zoom)
            if target_w >= 1 and target_h >= 1:
                scaled = pygame.transform.scale(surf, (target_w, target_h))
                self.screen.blit(scaled, (int(sx), int(sy)))
        else:
            region_types = world.tile_types[min_y:max_y + 1, min_x:max_x + 1]
            rgb = TILE_COLORS_NP[region_types].copy()

            non_water = region_types != TileType.WATER
            max_f = world.max_foods[min_y:max_y + 1, min_x:max_x + 1]
            has_food = non_water & (max_f > 0)

            if np.any(has_food):
                food_ratio = np.zeros_like(max_f)
                valid = max_f > 0
                food_ratio[valid] = world.food_values[min_y:max_y + 1, min_x:max_x + 1][valid] / max_f[valid]
                brightness = (0.5 + food_ratio * 0.5)[:, :, np.newaxis]
                rgb_f = rgb.astype(np.float32)
                rgb_f[has_food] = rgb_f[has_food] * brightness[has_food]
                rgb = rgb_f.astype(np.uint8)

            biomass_region = world.biomass[min_y:max_y + 1, min_x:max_x + 1]
            high_biomass = biomass_region > 10.0
            if np.any(high_biomass):
                biomass_intensity = np.clip(biomass_region / 100.0, 0, 1)
                brown = np.zeros_like(rgb)
                brown[:, :, 0] = 60
                brown[:, :, 1] = 30
                brown[:, :, 2] = 10
                rgb_f = rgb.astype(np.float32)
                rgb_f = rgb_f * (1 - biomass_intensity[:, :, np.newaxis] * 0.5) + brown * biomass_intensity[:, :, np.newaxis] * 0.5
                rgb = rgb_f.astype(np.uint8)

            rgb_t = np.ascontiguousarray(rgb.transpose(1, 0, 2))
            surf = pygame.Surface((rgb_t.shape[0], rgb_t.shape[1]))
            pygame.surfarray.blit_array(surf, rgb_t)

            sx, sy = camera.world_to_screen(float(min_x), float(min_y))
            scaled_w = int(screen_w * camera.zoom)
            scaled_h = int(screen_h * camera.zoom)
            if scaled_w >= 1 and scaled_h >= 1:
                if camera.zoom != 1.0:
                    scaled = pygame.transform.scale(surf, (scaled_w, scaled_h))
                    self.screen.blit(scaled, (int(sx), int(sy)))
                else:
                    self.screen.blit(surf, (int(sx), int(sy)))

    def render_entities(self, entity_manager: EntityManager, camera: Camera) -> None:
        entities_to_draw = []
        for eid in entity_manager.get_entities_with(Position, Appearance):
            pos = entity_manager.get_component(eid, Position)
            app = entity_manager.get_component(eid, Appearance)
            if pos is None or app is None:
                continue
            if not camera.is_visible(pos.x, pos.y, margin=20):
                continue
            entities_to_draw.append((eid, pos, app))

        for eid, pos, app in entities_to_draw:
            sx, sy = camera.world_to_screen(pos.x, pos.y)
            radius = max(1, int(app.size * camera.zoom * 0.5))

            color = (
                max(0, min(255, app.r)),
                max(0, min(255, app.g)),
                max(0, min(255, app.b)),
            )

            if app.shape == "triangle":
                r = radius
                points = [
                    (int(sx), int(sy - r)),
                    (int(sx - r), int(sy + r)),
                    (int(sx + r), int(sy + r)),
                ]
                pygame.draw.polygon(self.screen, color, points)
            elif app.shape == "pentagon":
                r = radius
                pts = []
                for a in range(5):
                    angle = a * 2 * math.pi / 5 - math.pi / 2
                    pts.append((int(sx + r * math.cos(angle)),
                                int(sy + r * math.sin(angle))))
                pygame.draw.polygon(self.screen, color, pts)
            elif app.shape == "diamond":
                r = radius
                points = [
                    (int(sx), int(sy - r)),
                    (int(sx + r), int(sy)),
                    (int(sx), int(sy + r)),
                    (int(sx - r), int(sy)),
                ]
                pygame.draw.polygon(self.screen, color, points)
            elif app.shape == "hexagon":
                r = radius
                pts = []
                for a in range(6):
                    angle = a * math.pi / 3
                    pts.append((int(sx + r * math.cos(angle)),
                                int(sy + r * math.sin(angle))))
                pygame.draw.polygon(self.screen, color, pts)
            elif app.shape == "square":
                r = radius
                rect = pygame.Rect(int(sx - r), int(sy - r), r * 2, r * 2)
                pygame.draw.rect(self.screen, color, rect)
            else:
                pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius)

    def render_entities_from_soa(self, entity_data: EntityData, camera: Camera) -> None:
        ed = entity_data
        n = ed.count
        if n == 0:
            return

        sx_all = (ed.x[:n] - camera.x) * camera.zoom + camera.screen_width / 2
        sy_all = (ed.y[:n] - camera.y) * camera.zoom + camera.screen_height / 2

        visible = (
            (sx_all > -20)
            & (sx_all < camera.screen_width + 20)
            & (sy_all > -20)
            & (sy_all < camera.screen_height + 20)
            & ed.alive[:n]
        )

        for idx in np.where(visible)[0]:
            idx = int(idx)
            screen_x = float(sx_all[idx])
            screen_y = float(sy_all[idx])
            radius = max(1, int(ed.size_gene[idx] * camera.zoom * 0.5))

            color = (int(ed.r[idx]), int(ed.g[idx]), int(ed.b[idx]))

            diet_int = int(ed.diet_type[idx])
            repro_type_int = int(ed.repro_type[idx])
            habitat_int = int(ed.habitat[idx])

            if diet_int == 3:
                r = radius
                pts = []
                for a in range(6):
                    angle = a * math.pi / 3
                    pts.append((int(screen_x + r * math.cos(angle)),
                                int(screen_y + r * math.sin(angle))))
                pygame.draw.polygon(self.screen, color, pts)
            elif diet_int == 2:
                r = radius
                points = [
                    (int(screen_x), int(screen_y - r)),
                    (int(screen_x - r), int(screen_y + r)),
                    (int(screen_x + r), int(screen_y + r)),
                ]
                pygame.draw.polygon(self.screen, color, points)
            elif repro_type_int == 0:
                r = radius
                points = [
                    (int(screen_x), int(screen_y - r)),
                    (int(screen_x + r), int(screen_y)),
                    (int(screen_x), int(screen_y + r)),
                    (int(screen_x - r), int(screen_y)),
                ]
                pygame.draw.polygon(self.screen, color, points)
            elif repro_type_int == 2:
                r = radius
                pts = []
                for a in range(5):
                    angle = a * 2 * math.pi / 5 - math.pi / 2
                    pts.append((int(screen_x + r * math.cos(angle)),
                                int(screen_y + r * math.sin(angle))))
                pygame.draw.polygon(self.screen, color, pts)
            elif habitat_int == 0:
                r = radius
                rect = pygame.Rect(int(screen_x - r), int(screen_y - r), r * 2, r * 2)
                pygame.draw.rect(self.screen, color, rect)
            else:
                pygame.draw.circle(self.screen, color, (int(screen_x), int(screen_y)), radius)

    def render_selected(self, entity_manager: EntityManager, camera: Camera, selected_id: int | None, entity_data: EntityData = None) -> None:
        if selected_id is None:
            return

        if entity_data is not None:
            idx = entity_data.eid_to_idx.get(selected_id)
            if idx is None or not entity_data.alive[idx]:
                return
            sx, sy = camera.world_to_screen(float(entity_data.x[idx]), float(entity_data.y[idx]))
            diet_int = int(entity_data.diet_type[idx])
            if diet_int == 3:
                trap_power = float(entity_data.aggression[idx])
                trap_radius = self.config.carnivorous_plant.trap_base_radius + trap_power * (self.config.carnivorous_plant.trap_max_radius - self.config.carnivorous_plant.trap_base_radius)
                screen_radius = int(trap_radius * camera.zoom)
                if screen_radius > 0:
                    trap_surf = pygame.Surface((screen_radius * 2, screen_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(trap_surf, (255, 100, 100, 40), (screen_radius, screen_radius), screen_radius)
                    self.screen.blit(trap_surf, (int(sx) - screen_radius, int(sy) - screen_radius))
        else:
            pos = entity_manager.get_component(selected_id, Position)
            if pos is None:
                return
            sx, sy = camera.world_to_screen(pos.x, pos.y)

        radius = 15
        pygame.draw.circle(self.screen, (255, 255, 0), (int(sx), int(sy)), radius, 2)

    def render_night_overlay(self, world: World) -> None:
        p = world.day_progress
        if p < 0.55:
            return
        if p < 0.70:
            darkness = int((p - 0.55) / 0.15 * 80)
        else:
            darkness = 80
        if darkness <= 0:
            return
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((10, 10, 40, min(darkness, 80)))
        self.screen.blit(overlay, (0, 0))
