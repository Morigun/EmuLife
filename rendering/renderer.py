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

_COS_TABLE_5 = []
_SIN_TABLE_5 = []
for _a in range(5):
    _angle = _a * 2 * math.pi / 5 - math.pi / 2
    _COS_TABLE_5.append(math.cos(_angle))
    _SIN_TABLE_5.append(math.sin(_angle))

_COS_TABLE_6 = []
_SIN_TABLE_6 = []
for _a in range(6):
    _angle = _a * math.pi / 3
    _COS_TABLE_6.append(math.cos(_angle))
    _SIN_TABLE_6.append(math.sin(_angle))

_PI_OVER_3 = math.pi / 3

_SHAPE_MASKS: dict[tuple[str, int], np.ndarray] = {}


def _build_shape_masks():
    for size in range(1, 11):
        pts = []
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if abs(dx) + abs(dy) <= size:
                    pts.append((dx, dy))
        _SHAPE_MASKS[('diamond', size)] = np.array(pts, dtype=np.int32)

        pts = []
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if dx * dx + dy * dy <= size * size:
                    pts.append((dx, dy))
        _SHAPE_MASKS[('circle', size)] = np.array(pts, dtype=np.int32)

        pts = []
        for dy in range(-size, size + 1):
            half_w = int((dy + size) / (2 * size) * size) if size > 0 else 0
            for dx in range(-half_w, half_w + 1):
                pts.append((dx, dy))
        _SHAPE_MASKS[('triangle', size)] = np.array(pts, dtype=np.int32)

        pts = []
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= size:
                    pts.append((dx, dy))
        _SHAPE_MASKS[('pentagon', size)] = np.array(pts, dtype=np.int32)

        pts = []
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if abs(dx) + abs(dy) * 0.577 <= size:
                    pts.append((dx, dy))
        _SHAPE_MASKS[('hexagon', size)] = np.array(pts, dtype=np.int32)

        pts = []
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                pts.append((dx, dy))
        _SHAPE_MASKS[('square', size)] = np.array(pts, dtype=np.int32)


_build_shape_masks()

_SHAPE_NAMES = ['diamond', 'square', 'triangle', 'pentagon', 'circle', 'hexagon']


class Renderer:
    def __init__(self, screen: pygame.Surface, config: Config) -> None:
        self.screen = screen
        self.config = config
        self._night_overlay: pygame.Surface | None = None
        self._night_darkness: int = -1

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
            (sx_all > -20) & (sx_all < camera.screen_width + 20)
            & (sy_all > -20) & (sy_all < camera.screen_height + 20)
            & ed.alive[:n]
        )
        visible_indices = np.where(visible)[0]
        if len(visible_indices) == 0:
            return

        screen_arr = pygame.surfarray.pixels3d(self.screen)
        sw, sh = screen_arr.shape[0], screen_arr.shape[1]
        zoom = camera.zoom

        ix = sx_all[visible_indices].astype(np.int32)
        iy = sy_all[visible_indices].astype(np.int32)
        mask = (ix >= 0) & (ix < sw) & (iy >= 0) & (iy < sh)
        ix = ix[mask]
        iy = iy[mask]
        idxs = visible_indices[mask]
        r_arr = ed.r[idxs].astype(np.uint8)
        g_arr = ed.g[idxs].astype(np.uint8)
        b_arr = ed.b[idxs].astype(np.uint8)

        diet = ed.diet_type[idxs]
        repro = ed.repro_type[idxs]
        habitat = ed.habitat[idxs]

        shape_ids = np.full(len(idxs), 4, dtype=np.int8)
        shape_ids[diet == 2] = 2
        shape_ids[diet == 3] = 5
        shape_ids[(diet != 2) & (diet != 3) & (repro == 0)] = 0
        shape_ids[(diet != 2) & (diet != 3) & (repro == 2)] = 3
        shape_ids[(diet != 2) & (diet != 3) & (repro == 1) & (habitat == 0)] = 1

        raw_sizes = (ed.size_gene[idxs] * zoom * 0.5).astype(np.int32)
        quant_sizes = np.clip(raw_sizes, 1, 10)

        group_keys = shape_ids.astype(np.int32) * 11 + quant_sizes
        unique_keys = np.unique(group_keys)

        for key in unique_keys:
            g_mask = group_keys == key
            shape_id = int(key // 11)
            size = int(key % 11)
            shape_name = _SHAPE_NAMES[shape_id]

            smask = _SHAPE_MASKS.get((shape_name, size))
            if smask is None:
                smask = _SHAPE_MASKS[('circle', size)]

            g_ix = ix[g_mask]
            g_iy = iy[g_mask]
            g_r = r_arr[g_mask]
            g_g = g_arr[g_mask]
            g_b = b_arr[g_mask]

            n_ents = len(g_ix)
            n_pixels = len(smask)

            offsets_dx = smask[:, 0]
            offsets_dy = smask[:, 1]

            all_px = np.repeat(g_ix, n_pixels) + np.tile(offsets_dx, n_ents)
            all_py = np.repeat(g_iy, n_pixels) + np.tile(offsets_dy, n_ents)
            all_r = np.repeat(g_r, n_pixels)
            all_g = np.repeat(g_g, n_pixels)
            all_b = np.repeat(g_b, n_pixels)

            valid = (all_px >= 0) & (all_px < sw) & (all_py >= 0) & (all_py < sh)
            all_px = all_px[valid]
            all_py = all_py[valid]
            all_r = all_r[valid]
            all_g = all_g[valid]
            all_b = all_b[valid]

            screen_arr[all_px, all_py, 0] = all_r
            screen_arr[all_px, all_py, 1] = all_g
            screen_arr[all_px, all_py, 2] = all_b

        del screen_arr

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
            self._night_darkness = -1
            return
        if p < 0.70:
            darkness = int((p - 0.55) / 0.15 * 80)
        else:
            darkness = 80
        if darkness <= 0:
            self._night_darkness = -1
            return
        if darkness != self._night_darkness:
            self._night_darkness = darkness
            size = self.screen.get_size()
            self._night_overlay = pygame.Surface(size, pygame.SRCALPHA)
            self._night_overlay.fill((10, 10, 40, min(darkness, 80)))
        self.screen.blit(self._night_overlay, (0, 0))
