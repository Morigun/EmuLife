from __future__ import annotations

import math
from dataclasses import dataclass

from config import Config


@dataclass
class Camera:
    x: float = 500.0
    y: float = 500.0
    zoom: float = 1.0
    screen_width: int = 1280
    screen_height: int = 720
    pan_speed: float = 300.0
    zoom_min: float = 0.1
    zoom_max: float = 10.0
    follow_entity: int | None = None

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        sx = (wx - self.x) * self.zoom + self.screen_width / 2
        sy = (wy - self.y) * self.zoom + self.screen_height / 2
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        wx = (sx - self.screen_width / 2) / self.zoom + self.x
        wy = (sy - self.screen_height / 2) / self.zoom + self.y
        return wx, wy

    def pan(self, dx: float, dy: float, dt: float) -> None:
        speed = self.pan_speed / self.zoom
        self.x += dx * speed * dt
        self.y += dy * speed * dt

    def zoom_at(self, factor: float, sx: float, sy: float) -> None:
        wx, wy = self.screen_to_world(sx, sy)
        self.zoom = max(self.zoom_min, min(self.zoom_max, self.zoom * factor))
        new_wx, new_wy = self.screen_to_world(sx, sy)
        self.x -= (new_wx - wx)
        self.y -= (new_wy - wy)

    def is_visible(self, wx: float, wy: float, margin: float = 50.0) -> bool:
        sx, sy = self.world_to_screen(wx, wy)
        return (-margin <= sx <= self.screen_width + margin and
                -margin <= sy <= self.screen_height + margin)

    def visible_bounds(self) -> tuple[float, float, float, float]:
        left, top = self.screen_to_world(0, 0)
        right, bottom = self.screen_to_world(self.screen_width, self.screen_height)
        return left, top, right, bottom

    def update(self, entity_manager=None, dt: float = 0.0, entity_data=None) -> None:
        if self.follow_entity is not None:
            if entity_data is not None:
                idx = entity_data.eid_to_idx.get(self.follow_entity)
                if idx is not None and entity_data.alive[idx]:
                    self.x = float(entity_data.x[idx])
                    self.y = float(entity_data.y[idx])
                else:
                    self.follow_entity = None
            elif entity_manager is not None:
                from components.position import Position
                pos = entity_manager.get_component(self.follow_entity, Position)
                if pos:
                    self.x = pos.x
                    self.y = pos.y
                else:
                    self.follow_entity = None
