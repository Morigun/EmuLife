from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator


MAX_PER_CELL = 32


class SpatialHash:
    def __init__(self, cell_size: int = 50, world_w: int = 1000, world_h: int = 1000) -> None:
        self.cell_size = cell_size
        self._world_w = world_w
        self._world_h = world_h
        self._cols = (world_w + cell_size - 1) // cell_size
        self._rows = (world_h + cell_size - 1) // cell_size
        self._total_cells = self._rows * self._cols

        self._cell_counts = np.zeros(self._total_cells, dtype=np.int32)
        self._cell_eids = np.full((self._total_cells, MAX_PER_CELL), -1, dtype=np.int32)

        self._entity_cells: dict[int, int] = {}

        self._query_buf = np.zeros(1024, dtype=np.int32)
        self._count_buf = np.zeros(1, dtype=np.int32)

    def clear(self) -> None:
        self._cell_counts.fill(0)
        self._cell_eids.fill(-1)
        self._entity_cells.clear()

    def insert(self, entity_id: int, x: float, y: float) -> None:
        cx = int(x // self.cell_size)
        cy = int(y // self.cell_size)
        if cx < 0 or cx >= self._cols or cy < 0 or cy >= self._rows:
            return
        cell_idx = cy * self._cols + cx
        count = self._cell_counts[cell_idx]
        if count < MAX_PER_CELL:
            self._cell_eids[cell_idx, count] = entity_id
            self._cell_counts[cell_idx] = count + 1
        self._entity_cells[entity_id] = cell_idx

    def remove(self, entity_id: int) -> None:
        cell_idx = self._entity_cells.pop(entity_id, -1)
        if cell_idx < 0:
            return
        count = self._cell_counts[cell_idx]
        eids = self._cell_eids[cell_idx]
        for i in range(count):
            if eids[i] == entity_id:
                eids[i] = eids[count - 1]
                eids[count - 1] = -1
                self._cell_counts[cell_idx] = count - 1
                break

    def update(self, entity_id: int, old_x: float, old_y: float, new_x: float, new_y: float) -> None:
        old_cx = int(old_x // self.cell_size)
        old_cy = int(old_y // self.cell_size)
        new_cx = int(new_x // self.cell_size)
        new_cy = int(new_y // self.cell_size)
        if old_cx == new_cx and old_cy == new_cy:
            return
        self.remove(entity_id)
        self.insert(entity_id, new_x, new_y)

    def query_nearby(self, x: float, y: float, radius: float) -> list[int]:
        results = []
        min_cx = max(0, int((x - radius) // self.cell_size))
        max_cx = min(self._cols - 1, int((x + radius) // self.cell_size))
        min_cy = max(0, int((y - radius) // self.cell_size))
        max_cy = min(self._rows - 1, int((y + radius) // self.cell_size))
        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                cell_idx = cy * self._cols + cx
                count = self._cell_counts[cell_idx]
                eids = self._cell_eids[cell_idx]
                for i in range(count):
                    eid = int(eids[i])
                    if eid >= 0:
                        results.append(eid)
        return results

    def query_nearby_into(self, x: float, y: float, radius: float, result_set: set[int]) -> set[int]:
        result_set.clear()
        min_cx = max(0, int((x - radius) // self.cell_size))
        max_cx = min(self._cols - 1, int((x + radius) // self.cell_size))
        min_cy = max(0, int((y - radius) // self.cell_size))
        max_cy = min(self._rows - 1, int((y + radius) // self.cell_size))
        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                cell_idx = cy * self._cols + cx
                count = self._cell_counts[cell_idx]
                eids = self._cell_eids[cell_idx]
                for i in range(count):
                    eid = int(eids[i])
                    if eid >= 0:
                        result_set.add(eid)
        return result_set

    def query_nearby_excluding(
        self, x: float, y: float, radius: float, exclude_id: int
    ) -> list[int]:
        results = []
        min_cx = max(0, int((x - radius) // self.cell_size))
        max_cx = min(self._cols - 1, int((x + radius) // self.cell_size))
        min_cy = max(0, int((y - radius) // self.cell_size))
        max_cy = min(self._rows - 1, int((y + radius) // self.cell_size))
        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                cell_idx = cy * self._cols + cx
                count = self._cell_counts[cell_idx]
                eids = self._cell_eids[cell_idx]
                for i in range(count):
                    eid = int(eids[i])
                    if eid >= 0 and eid != exclude_id:
                        results.append(eid)
        return results

    def query_nearby_excluding_into(
        self, x: float, y: float, radius: float, exclude_id: int, result_set: set[int]
    ) -> set[int]:
        result_set.clear()
        min_cx = max(0, int((x - radius) // self.cell_size))
        max_cx = min(self._cols - 1, int((x + radius) // self.cell_size))
        min_cy = max(0, int((y - radius) // self.cell_size))
        max_cy = min(self._rows - 1, int((y + radius) // self.cell_size))
        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                cell_idx = cy * self._cols + cx
                count = self._cell_counts[cell_idx]
                eids = self._cell_eids[cell_idx]
                for i in range(count):
                    eid = int(eids[i])
                    if eid >= 0 and eid != exclude_id:
                        result_set.add(eid)
        return result_set

    def query_nearby_eids_array(self, x: float, y: float, radius: float, exclude_id: int = -1):
        min_cx = max(0, int((x - radius) // self.cell_size))
        max_cx = min(self._cols - 1, int((x + radius) // self.cell_size))
        min_cy = max(0, int((y - radius) // self.cell_size))
        max_cy = min(self._rows - 1, int((y + radius) // self.cell_size))

        if min_cx > max_cx or min_cy > max_cy:
            return np.empty(0, dtype=np.int32)

        buf = self._query_buf
        count = 0
        for cy in range(min_cy, max_cy + 1):
            for cx in range(min_cx, max_cx + 1):
                cell_idx = cy * self._cols + cx
                n = self._cell_counts[cell_idx]
                eids = self._cell_eids[cell_idx]
                for i in range(n):
                    eid = eids[i]
                    if eid >= 0 and eid != exclude_id:
                        if count < len(buf):
                            buf[count] = eid
                        else:
                            new_buf = np.empty(count * 2, dtype=np.int32)
                            new_buf[:count] = buf[:count]
                            new_buf[count] = eid
                            buf = new_buf
                            self._query_buf = buf
                        count += 1
        return buf[:count].copy()
