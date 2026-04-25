from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
import opensimplex

from config import Config, TileConfig


class TileType(IntEnum):
    WATER = 0
    LAND = 1
    FOREST = 2
    DESERT = 3


TILE_COLORS = {
    TileType.WATER: (30, 100, 200),
    TileType.LAND: (80, 180, 60),
    TileType.FOREST: (20, 120, 30),
    TileType.DESERT: (210, 190, 120),
}

TILE_COLORS_NP = np.array([
    [30, 100, 200],
    [80, 180, 60],
    [20, 120, 30],
    [210, 190, 120],
], dtype=np.uint8)


class World:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.width = config.world.width
        self.height = config.world.height
        self.tile_types: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.food_values: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)
        self.max_foods: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)
        self.regen_rates: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)
        self.walkable_mask: np.ndarray = np.zeros((self.height, self.width), dtype=np.bool_)
        self.biomass: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)
        self.day_tick: int = 0
        self.day_progress: float = 0.0
        self.is_night: bool = False
        self.food_regen_mult: float = 1.0
        self.vision_mult: float = 1.0
        self.metabolism_mult: float = 1.0
        self._generate()

    def _generate(self) -> None:
        import random
        tc = self.config.tile
        seed1 = random.randint(0, 2**31)
        seed2 = seed1 + 12345

        opensimplex.seed(seed1)
        scale_elev = 0.005
        scale_moist = 0.008

        elev_x = np.arange(self.width) * scale_elev
        elev_y = np.arange(self.height) * scale_elev
        moist_x = np.arange(self.width) * scale_moist
        moist_y = np.arange(self.height) * scale_moist

        elev_grid = opensimplex.noise2array(elev_x, elev_y)
        elev_grid = (elev_grid + 1.0) / 2.0

        opensimplex.seed(seed2)
        moist_grid = opensimplex.noise2array(moist_x, moist_y)
        moist_grid = (moist_grid + 1.0) / 2.0

        tile_types = np.zeros((self.height, self.width), dtype=np.int8)
        max_foods = np.zeros((self.height, self.width), dtype=np.float32)
        regen_rates = np.zeros((self.height, self.width), dtype=np.float32)

        water_mask = elev_grid.T < 0.35
        tile_types[water_mask] = TileType.WATER
        max_foods[water_mask] = tc.water_max_food
        regen_rates[water_mask] = tc.water_regen

        mid_elev = (elev_grid.T >= 0.35) & (elev_grid.T < 0.65)
        high_elev = elev_grid.T >= 0.65

        mid_forest = mid_elev & (moist_grid.T > 0.55)
        tile_types[mid_forest] = TileType.FOREST
        max_foods[mid_forest] = tc.forest_max_food
        regen_rates[mid_forest] = tc.forest_regen

        mid_desert = mid_elev & (moist_grid.T < 0.3)
        tile_types[mid_desert] = TileType.DESERT
        max_foods[mid_desert] = tc.desert_max_food
        regen_rates[mid_desert] = tc.desert_regen

        mid_land = mid_elev & ~mid_forest & ~mid_desert
        tile_types[mid_land] = TileType.LAND
        max_foods[mid_land] = tc.land_max_food
        regen_rates[mid_land] = tc.land_regen

        high_forest = high_elev & (moist_grid.T > 0.5)
        tile_types[high_forest] = TileType.FOREST
        max_foods[high_forest] = tc.forest_max_food
        regen_rates[high_forest] = tc.forest_regen

        high_land = high_elev & ~high_forest
        tile_types[high_land] = TileType.LAND
        max_foods[high_land] = tc.land_max_food
        regen_rates[high_land] = tc.land_regen

        self.tile_types = tile_types
        self.max_foods = max_foods
        self.regen_rates = regen_rates
        self.food_values = max_foods * 0.5
        self.walkable_mask = tile_types != TileType.WATER
        self._active_mask = self.regen_rates > 0

    def get_tile_type(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.tile_types[y, x])
        return TileType.WATER

    def get_food(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.food_values[y, x])
        return 0.0

    def get_max_food(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.max_foods[y, x])
        return 0.0

    def get_regen_rate(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.regen_rates[y, x])
        return 0.0

    def is_walkable(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return bool(self.walkable_mask[iy, ix])
        return False

    def regenerate_all(self, mult: float = 1.0) -> None:
        active = self._active_mask
        np.add(self.food_values, self.regen_rates * mult, out=self.food_values, where=active)
        np.minimum(self.food_values, self.max_foods, out=self.food_values)
