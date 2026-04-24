import numpy as np

from core.ecs import System, EntityManager
from core.entity_data import EntityData
from core.world import World
from utils.numba_kernels import energy_update_kernel, HAS_NUMBA
from config import Config


class EnergySystem(System):
    def __init__(self, entity_manager: EntityManager, config: Config, entity_data: EntityData = None) -> None:
        self.em = entity_manager
        self.config = config
        self.entity_data = entity_data

    def update(self, world: object, dt: float) -> None:
        if self.entity_data is None:
            self._update_ecs(world, dt)
            return
        self._update_soa(world, dt)

    def _update_soa(self, world: object, dt: float) -> None:
        w: World = world
        ed = self.entity_data
        ec = self.config.energy
        n = ed.count

        if HAS_NUMBA:
            energy_update_kernel(
                ed.x, ed.y, ed.dx, ed.dy, ed.energy, ed.max_energy,
                ed.metabolism, ed.size_gene, ed.diet_type, ed.habitat,
                ed.efficiency, ed.alive,
                w.biomass,
                w.food_values, w.tile_types,
                n, w.width, w.height, dt, ec.energy_from_food,
                ed.metabolism_mod, ed.efficiency_mod,
                ed.photosynth, w.food_regen_mult,
            )
            return

        s = slice(0, n)

        alive = ed.alive[s]
        raw_size = (ed.size_gene[s] - 3.0) / 7.0
        cost = ed.metabolism[s] * (0.5 + raw_size) * dt
        speed = np.sqrt(ed.dx[s] ** 2 + ed.dy[s] ** 2)
        cost += speed * raw_size * 0.5 * dt
        pred_mask = ed.diet_type[s] == 2
        cost = np.where(pred_mask, cost * 0.5, cost)

        plant_mask = ed.diet_type[s] == 3
        cpc = self.config.carnivorous_plant
        plant_day_mask = plant_mask & (w.food_regen_mult >= 0.5)
        plant_night_mask = plant_mask & (w.food_regen_mult < 0.5)
        cost = np.where(plant_day_mask, cost * cpc.metabolism_mult, cost)
        cost = np.where(plant_night_mask, cost * cpc.metabolism_mult * cpc.night_dormancy_mult, cost)

        cost *= ed.metabolism_mod[s]

        ed.energy[s] -= np.where(alive, cost, 0)

        ix = np.clip(ed.x[s].astype(np.int32), 0, w.width - 1)
        iy = np.clip(ed.y[s].astype(np.int32), 0, w.height - 1)

        bio = w.biomass[iy, ix]
        scav = np.where(pred_mask & alive & (bio > 2.0), np.minimum(bio * 0.05, 3.0) * dt, 0.0).astype(np.float32)
        ed.energy[s] += scav
        scav_pos = scav > 0
        if np.any(scav_pos):
            np.add.at(w.biomass, (iy[scav_pos], ix[scav_pos]), -scav[scav_pos])
            np.clip(w.biomass, 0, None, out=w.biomass)

        from core.world import TileType
        is_water = w.tile_types[iy, ix] == TileType.WATER

        plant_alive = alive & plant_mask
        if np.any(plant_alive):
            tile_t = w.tile_types[iy, ix]
            biome_bonus = np.where(tile_t == 2, 1.3, np.where(tile_t == 3, 0.3, 1.0)).astype(np.float32)
            photo_income = ed.photosynth[s] * w.food_regen_mult * biome_bonus * self.config.carnivorous_plant.photosynth_base_rate * dt
            ed.energy[s] += np.where(plant_alive, photo_income, 0).astype(np.float32)

        non_pred = ed.diet_type[s] != 2
        non_plant = ed.diet_type[s] != 3
        food = w.food_values[iy, ix]
        habitat = ed.habitat[s]

        can_eat = alive & non_pred & non_plant & (food > 0) & (
            ((habitat == 0) & is_water)
            | ((habitat == 1) & ~is_water)
            | (habitat == 2)
        )

        flat_indices = iy * w.width + ix
        counts = np.zeros(w.width * w.height, dtype=np.float32)
        np.add.at(counts, flat_indices, 1.0)
        per_entity_count = counts[flat_indices]

        available = np.maximum(food, 0)
        share = available / np.maximum(per_entity_count, 1.0)
        eat_amount = np.minimum(ec.energy_from_food * dt, share)

        herb_bonus = np.where(ed.diet_type[s] == 0, 1.3, 1.0).astype(np.float32)
        ed.energy[s] += np.where(can_eat, eat_amount * ed.efficiency[s] * ed.efficiency_mod[s] * herb_bonus, 0).astype(np.float32)

        depletion = np.where(can_eat, eat_amount, 0)
        np.add.at(w.food_values, (iy, ix), -depletion)
        np.clip(w.food_values, 0, None, out=w.food_values)

        pred_mask = ed.diet_type[s] == 2
        plant_cap_mask = ed.diet_type[s] == 3
        cap = np.where(pred_mask, ed.max_energy[s] * 1.5, np.where(plant_cap_mask, ed.max_energy[s] * self.config.carnivorous_plant.energy_cap_mult, ed.max_energy[s]))
        ed.energy[s] = np.minimum(ed.energy[s], cap)

        from components.conditions import Condition, Conditions
        low_energy_mask = alive & (ed.energy[s] < ed.max_energy[s] * 0.1)
        for idx_in_mask in np.where(low_energy_mask)[0]:
            idx_int = int(idx_in_mask)
            eid = ed.idx_to_eid.get(idx_int)
            if eid is None:
                continue
            conds = self.em.get_component(eid, Conditions)
            if conds is None:
                conds = Conditions()
                self.em.add_component(eid, conds)
            has_exhaustion = any(e.name == "exhaustion" for e in conds.effects)
            if not has_exhaustion and len(conds.effects) < 3:
                conds.effects.append(
                    Condition("exhaustion", 30, speed_mult=0.6, metabolism_mult=1.0)
                )

    def _update_ecs(self, world: object, dt: float) -> None:
        w: World = world
        ec = self.config.energy

        from components.position import Position
        from components.velocity import Velocity
        from components.energy import Energy
        from components.metabolism import Metabolism
        from components.diet import Diet, DietType
        from components.genome_comp import GenomeComp
        import math

        for eid in list(self.em.get_entities_with(Position, Energy, Metabolism)):
            pos = self.em.get_component(eid, Position)
            energy = self.em.get_component(eid, Energy)
            metab = self.em.get_component(eid, Metabolism)
            if pos is None or energy is None or metab is None:
                continue

            genome_comp = self.em.get_component(eid, GenomeComp)
            size = genome_comp.genome.size if genome_comp else 0.5

            base_cost = metab.rate * (0.5 + size) * dt
            vel = self.em.get_component(eid, Velocity)
            if vel is not None:
                speed = math.sqrt(vel.dx ** 2 + vel.dy ** 2)
                base_cost += speed * size * 0.5 * dt

            diet = self.em.get_component(eid, Diet)
            if diet is not None and diet.diet_type == DietType.PREDATOR:
                base_cost *= 0.5
            elif diet is not None and diet.diet_type == DietType.CARNIVOROUS_PLANT:
                base_cost *= self.config.carnivorous_plant.metabolism_mult
                if w.food_regen_mult < 0.5:
                    base_cost *= self.config.carnivorous_plant.night_dormancy_mult

            from components.conditions import Conditions
            conds = self.em.get_component(eid, Conditions)
            if conds is not None:
                met_mod = 1.0
                for eff in conds.effects:
                    met_mod *= eff.metabolism_mult
                base_cost *= met_mod

            energy.current -= base_cost

            if diet is not None and diet.diet_type == DietType.PREDATOR:
                ix = max(0, min(int(pos.x), w.width - 1))
                iy = max(0, min(int(pos.y), w.height - 1))
                bio_val = w.biomass[iy, ix]
                if bio_val > 2.0:
                    scav = min(bio_val * 0.05, 3.0) * dt
                    energy.current += scav
                    w.biomass[iy, ix] -= scav

            if diet is not None and diet.diet_type == DietType.CARNIVOROUS_PLANT:
                ix = max(0, min(int(pos.x), w.width - 1))
                iy = max(0, min(int(pos.y), w.height - 1))
                tile_t = int(w.tile_types[iy, ix])
                biome_bonus = 1.3 if tile_t == 2 else (0.3 if tile_t == 3 else 1.0)
                genome_comp = self.em.get_component(eid, GenomeComp)
                photo_eff = genome_comp.genome.photosynth if genome_comp and genome_comp.genome else 0.5
                photo_income = photo_eff * w.food_regen_mult * biome_bonus * self.config.carnivorous_plant.photosynth_base_rate * dt
                energy.current += photo_income

            if diet is None or (diet.diet_type != DietType.PREDATOR and diet.diet_type != DietType.CARNIVOROUS_PLANT):
                ix, iy = int(pos.x), int(pos.y)
                fv = w.get_food(ix, iy)
                if fv > 0:
                    eat_amount = min(ec.energy_from_food * dt, fv)
                    eff_mult = diet.efficiency if diet else 1.0
                    if diet and diet.diet_type == DietType.HERBIVORE:
                        eff_mult *= 1.3
                    energy.current += eat_amount * eff_mult
                    w.food_values[iy, ix] -= eat_amount

            if diet and diet.diet_type == DietType.CARNIVOROUS_PLANT:
                cap = energy.max_value * self.config.carnivorous_plant.energy_cap_mult
            elif diet and diet.diet_type == DietType.PREDATOR:
                cap = energy.max_value * 1.5
            else:
                cap = energy.max_value
            energy.current = min(energy.current, cap)
