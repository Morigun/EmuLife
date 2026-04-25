from __future__ import annotations

import numpy as np
import pygame

from core.ecs import EntityManager
from core.camera import Camera
from core.world import World, TileType, TILE_COLORS, TILE_COLORS_NP
from core.entity_data import EntityData
from components.position import Position
from components.diet import Diet, DietType
from config import Config


class Minimap:
    def __init__(self, screen: pygame.Surface, config: Config) -> None:
        self.screen = screen
        self.config = config
        self.size = 150
        self.margin = 10
        self.surface = pygame.Surface((self.size, self.size))
        self.visible = True
        self._world_surface: pygame.Surface | None = None
        self._entity_frame = 0
        self._entity_interval = 10
        self._entity_surface: pygame.Surface | None = None
        self._night_surface: pygame.Surface | None = None
        self._night_alpha: int = -1

    def _build_world_surface(self, world: World) -> pygame.Surface:
        indices = np.linspace(0, world.height - 1, self.size).astype(int)
        columns = np.linspace(0, world.width - 1, self.size).astype(int)

        sampled = world.tile_types[np.ix_(indices, columns)]
        rgb = TILE_COLORS_NP[sampled]

        rgb_transposed = np.ascontiguousarray(rgb.transpose(1, 0, 2))
        surf = pygame.Surface((self.size, self.size))
        pygame.surfarray.blit_array(surf, rgb_transposed)
        return surf

    def render(self, world: World, camera: Camera, entity_manager: EntityManager, tick: int = 0, entity_data: EntityData = None) -> None:
        if not self.visible:
            return

        if self._world_surface is None:
            self._world_surface = self._build_world_surface(world)

        self.surface.blit(self._world_surface, (0, 0))

        if world.is_night or world.day_progress > 0.55:
            if world.day_progress < 0.70:
                alpha = int((world.day_progress - 0.55) / 0.15 * 60)
            else:
                alpha = 60
            if alpha != self._night_alpha:
                self._night_alpha = alpha
                self._night_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
                self._night_surface.fill((10, 10, 40, min(alpha, 60)))
            self.surface.blit(self._night_surface, (0, 0))

        self._entity_frame += 1
        if self._entity_surface is None or self._entity_frame >= self._entity_interval:
            self._entity_frame = 0
            self._entity_surface = pygame.Surface((self.size, self.size))
            self._entity_surface.fill((1, 0, 1))
            self._entity_surface.set_colorkey((1, 0, 1))

            scale_x = self.size / world.width
            scale_y = self.size / world.height

            if entity_data is not None:
                self._draw_entities_soa(entity_data, scale_x, scale_y)
            else:
                self._draw_entities_ecs(entity_manager, scale_x, scale_y)

        self.surface.blit(self._entity_surface, (0, 0))

        scale_x = self.size / world.width
        scale_y = self.size / world.height
        left, top, right, bottom = camera.visible_bounds()
        vx = int(left * scale_x)
        vy = int(top * scale_y)
        vw = int((right - left) * scale_x)
        vh = int((bottom - top) * scale_y)
        pygame.draw.rect(self.surface, (255, 255, 255), (vx, vy, vw, vh), 1)

        dest_x = self.screen.get_width() - self.size - self.margin
        dest_y = self.screen.get_height() - self.size - self.margin
        self.screen.blit(self.surface, (dest_x, dest_y))
        pygame.draw.rect(
            self.screen, (100, 100, 100),
            (dest_x - 1, dest_y - 1, self.size + 2, self.size + 2), 1
        )

    def _draw_entities_soa(self, ed: EntityData, scale_x: float, scale_y: float) -> None:
        n = ed.count
        if n == 0:
            return
        alive = ed.alive[:n]
        mx = (ed.x[:n] * scale_x).astype(np.int32)
        my = (ed.y[:n] * scale_y).astype(np.int32)
        in_bounds = (mx >= 0) & (mx < self.size) & (my >= 0) & (my < self.size)
        visible = alive & in_bounds

        arr = pygame.surfarray.pixels3d(self._entity_surface)
        for mask, color in [
            (ed.diet_type[:n] == 2, (255, 80, 80)),
            (ed.diet_type[:n] == 1, (255, 255, 80)),
            (ed.diet_type[:n] == 0, (80, 255, 80)),
            (ed.diet_type[:n] == 3, (0, 200, 100)),
        ]:
            idxs = np.where(visible & mask)[0]
            if len(idxs) == 0:
                continue
            px = mx[idxs]
            py = my[idxs]
            v = (px >= 0) & (px < self.size) & (py >= 0) & (py < self.size)
            arr[px[v], py[v], 0] = color[0]
            arr[px[v], py[v], 1] = color[1]
            arr[px[v], py[v], 2] = color[2]
        del arr

    def _draw_entities_ecs(self, entity_manager: EntityManager, scale_x: float, scale_y: float) -> None:
        for eid in entity_manager.get_entities_with(Position, Diet):
            pos = entity_manager.get_component(eid, Position)
            diet = entity_manager.get_component(eid, Diet)
            if pos is None or diet is None:
                continue
            mx = int(pos.x * scale_x)
            my = int(pos.y * scale_y)
            if diet.diet_type == DietType.PREDATOR:
                color = (255, 80, 80)
            elif diet.diet_type == DietType.OMNIVORE:
                color = (255, 255, 80)
            elif diet.diet_type == DietType.CARNIVOROUS_PLANT:
                color = (0, 200, 100)
            else:
                color = (80, 255, 80)
            if 0 <= mx < self.size and 0 <= my < self.size:
                self._entity_surface.set_at((mx, my), color)
