from core.ecs import System
from core.world import World
from config import Config


class DayNightSystem(System):
    def __init__(self, config: Config) -> None:
        self.config = config

    def update(self, world: object, dt: float) -> None:
        w: World = world
        dnc = self.config.day_night

        w.day_tick += 1
        if w.day_tick >= dnc.cycle_ticks:
            w.day_tick = 0

        w.day_progress = w.day_tick / dnc.cycle_ticks
        p = w.day_progress

        if p < dnc.day_start:
            t = (p - dnc.dawn_start) / (dnc.day_start - dnc.dawn_start) if dnc.day_start > dnc.dawn_start else 0.0
            t = max(0.0, min(1.0, t))
            w.food_regen_mult = _lerp(dnc.night_food_regen_mult, dnc.day_food_regen_mult, t)
            w.vision_mult = _lerp(dnc.night_vision_mult, dnc.day_vision_mult, t)
            w.metabolism_mult = dnc.day_metabolism_mult
            w.is_night = False
        elif p < dnc.dusk_start:
            w.food_regen_mult = dnc.day_food_regen_mult
            w.vision_mult = dnc.day_vision_mult
            w.metabolism_mult = dnc.day_metabolism_mult
            w.is_night = False
        elif p < dnc.night_start:
            t = (p - dnc.dusk_start) / (dnc.night_start - dnc.dusk_start) if dnc.night_start > dnc.dusk_start else 0.0
            t = max(0.0, min(1.0, t))
            w.food_regen_mult = _lerp(dnc.day_food_regen_mult, dnc.night_food_regen_mult, t)
            w.vision_mult = _lerp(dnc.day_vision_mult, dnc.night_vision_mult, t)
            w.metabolism_mult = dnc.night_metabolism_mult
            w.is_night = False
        else:
            w.food_regen_mult = dnc.night_food_regen_mult
            w.vision_mult = dnc.night_vision_mult
            w.metabolism_mult = dnc.night_metabolism_mult
            w.is_night = True


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
