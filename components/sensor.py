from __future__ import annotations

from dataclasses import dataclass, field
from core.ecs import Component
from components.diet import DietType


@dataclass(slots=True)
class NearbyEntity:
    eid: int
    dist: float
    diet_type: DietType | None = None


@dataclass
class Sensor(Component):
    radius: float = 50.0
    nearby_entities: list[NearbyEntity] = field(default_factory=list)
    nearest_food_pos: tuple[float, float] | None = None
    food_cache_tick: int = 0
    food_cache_interval: int = 5
