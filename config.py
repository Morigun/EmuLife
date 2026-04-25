from dataclasses import dataclass


@dataclass
class WorldConfig:
    width: int = 1000
    height: int = 1000
    tile_size: int = 1


@dataclass
class SimulationConfig:
    tick_rate: int = 60
    simulation_speed: float = 1.0
    initial_population: int = 200
    max_population: int = 10000


@dataclass
class GenomeConfig:
    genome_length: int = 16
    mutation_rate: float = 0.20
    mutation_strength: float = 0.25


@dataclass
class EnergyConfig:
    base_energy: float = 100.0
    energy_from_food: float = 15.0
    reproduction_energy_cost: float = 0.3
    child_energy_fraction: float = 0.45
    predation_efficiency: float = 1.5
    asexual_reproduction_energy_cost: float = 0.5
    asexual_child_energy_fraction: float = 0.3
    predator_child_energy_fraction: float = 0.85
    blood_meal_fraction: float = 0.20


@dataclass
class ReproductionConfig:
    asexual_cooldown: int = 250
    sexual_cooldown: int = 100
    max_births_per_tick: int = 5
    density_radius: float = 40.0
    density_max_neighbors: int = 8


@dataclass
class ScreenConfig:
    width: int = 1280
    height: int = 720
    fps_cap: int = 60


@dataclass
class TileConfig:
    water_regen: float = 0.10
    land_regen: float = 0.08
    forest_regen: float = 0.18
    desert_regen: float = 0.01
    water_max_food: float = 40.0
    land_max_food: float = 50.0
    forest_max_food: float = 80.0
    desert_max_food: float = 20.0


@dataclass
class AbiogenesisConfig:
    biomass_decay_rate: float = 0.005
    biomass_threshold: float = 10.0
    biomass_spawn_chance: float = 0.00003
    lightning_chance: float = 0.0005
    lightning_biomass_boost: float = 50.0
    lightning_radius: int = 5


@dataclass
class SpatialHashConfig:
    cell_size: int = 50


@dataclass
class DayNightConfig:
    cycle_ticks: int = 1200
    dawn_start: float = 0.0
    day_start: float = 0.15
    dusk_start: float = 0.55
    night_start: float = 0.70
    day_food_regen_mult: float = 1.0
    day_vision_mult: float = 1.0
    day_metabolism_mult: float = 1.0
    night_food_regen_mult: float = 0.3
    night_vision_mult: float = 0.5
    night_metabolism_mult: float = 0.8


@dataclass
class CarnivorousPlantConfig:
    trap_base_radius: float = 8.0
    trap_max_radius: float = 20.0
    trap_damage_mult: float = 50.0
    trap_cooldown_ticks: int = 20
    photosynth_base_rate: float = 0.8
    photosynth_night_mult: float = 0.3
    spawn_spread: float = 3.0
    max_trap_targets: int = 3
    metabolism_mult: float = 0.4
    night_dormancy_mult: float = 0.3
    energy_cap_mult: float = 1.5


@dataclass
class Config:
    world: WorldConfig = WorldConfig()
    simulation: SimulationConfig = SimulationConfig()
    genome: GenomeConfig = GenomeConfig()
    energy: EnergyConfig = EnergyConfig()
    screen: ScreenConfig = ScreenConfig()
    tile: TileConfig = TileConfig()
    spatial_hash: SpatialHashConfig = SpatialHashConfig()
    abiogenesis: AbiogenesisConfig = AbiogenesisConfig()
    reproduction: ReproductionConfig = ReproductionConfig()
    day_night: DayNightConfig = DayNightConfig()
    carnivorous_plant: CarnivorousPlantConfig = CarnivorousPlantConfig()
