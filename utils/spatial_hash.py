from __future__ import annotations

from typing import Set, List
import math


class SpatialHash:
    def __init__(self, cell_size: int = 50) -> None:
        self.cell_size = cell_size
        self._cells: dict[tuple[int, int], set[int]] = {}
        self._entity_cells: dict[int, set[tuple[int, int]]] = {}

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _keys_for_radius(self, x: float, y: float, radius: float) -> list[tuple[int, int]]:
        min_cx = int((x - radius) // self.cell_size)
        max_cx = int((x + radius) // self.cell_size)
        min_cy = int((y - radius) // self.cell_size)
        max_cy = int((y + radius) // self.cell_size)
        keys = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                keys.append((cx, cy))
        return keys

    def insert(self, entity_id: int, x: float, y: float) -> None:
        key = self._key(x, y)
        if key not in self._cells:
            self._cells[key] = set()
        self._cells[key].add(entity_id)
        if entity_id not in self._entity_cells:
            self._entity_cells[entity_id] = set()
        self._entity_cells[entity_id].add(key)

    def remove(self, entity_id: int) -> None:
        cells = self._entity_cells.pop(entity_id, set())
        for key in cells:
            bucket = self._cells.get(key)
            if bucket is not None:
                bucket.discard(entity_id)
                if not bucket:
                    del self._cells[key]

    def update(self, entity_id: int, old_x: float, old_y: float, new_x: float, new_y: float) -> None:
        old_key = self._key(old_x, old_y)
        new_key = self._key(new_x, new_y)
        if old_key == new_key:
            return
        self.remove(entity_id)
        self.insert(entity_id, new_x, new_y)

    def query_nearby(self, x: float, y: float, radius: float) -> list[int]:
        keys = self._keys_for_radius(x, y, radius)
        result: set[int] = set()
        for key in keys:
            bucket = self._cells.get(key)
            if bucket is not None:
                result |= bucket
        return list(result)

    def query_nearby_into(self, x: float, y: float, radius: float, result_set: set[int]) -> set[int]:
        result_set.clear()
        keys = self._keys_for_radius(x, y, radius)
        for key in keys:
            bucket = self._cells.get(key)
            if bucket is not None:
                result_set |= bucket
        return result_set

    def query_nearby_excluding(
        self, x: float, y: float, radius: float, exclude_id: int
    ) -> list[int]:
        nearby = self.query_nearby(x, y, radius)
        return [eid for eid in nearby if eid != exclude_id]

    def query_nearby_excluding_into(
        self, x: float, y: float, radius: float, exclude_id: int, result_set: set[int]
    ) -> set[int]:
        self.query_nearby_into(x, y, radius, result_set)
        result_set.discard(exclude_id)
        return result_set

    def clear(self) -> None:
        self._cells.clear()
        self._entity_cells.clear()
