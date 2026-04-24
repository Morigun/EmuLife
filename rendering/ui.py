from __future__ import annotations

import numpy as np
import pygame

from core.ecs import EntityManager
from core.entity_data import EntityData
from components.diet import DietType, Diet
from components.energy import Energy
from components.age import Age
from components.genome_comp import GenomeComp
from components.health import Health
from config import Config
from utils.species_namer import (
    get_species_name,
    get_diet_name,
    get_repro_name,
    get_habitat_name,
    get_origin_name,
)


class UI:
    def __init__(self, screen: pygame.Surface, config: Config) -> None:
        self.screen = screen
        self.config = config
        self.em: EntityManager | None = None
        self.entity_data: EntityData | None = None
        self.cumulative: dict | None = None
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_medium = pygame.font.SysFont("consolas", 16)
        self.show_stats = True
        self.selected_entity: int | None = None
        self._stats_cache_tick = 0
        self._stats_cache_interval = 30
        self._cached_stats = (0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0)

    def render_stats(
        self,
        entity_manager: EntityManager,
        fps: float,
        tick: int,
        sim_speed: float,
        world=None,
    ) -> None:
        if not self.show_stats:
            return

        if tick - self._stats_cache_tick >= self._stats_cache_interval:
            ed = self.entity_data
            if ed is not None:
                n = ed.count
                alive = ed.alive[:n]
                herb = int(np.sum(alive & (ed.diet_type[:n] == 0)))
                omni = int(np.sum(alive & (ed.diet_type[:n] == 1)))
                pred = int(np.sum(alive & (ed.diet_type[:n] == 2)))
                plant = int(np.sum(alive & (ed.diet_type[:n] == 3)))
                total = herb + omni + pred + plant
                total_energy = float(np.sum(ed.energy[:n][alive]))
                asex = int(np.sum(alive & (ed.repro_type[:n] == 0)))
                sex = int(np.sum(alive & (ed.repro_type[:n] == 1)))
                herma = int(np.sum(alive & (ed.repro_type[:n] == 2)))
                abio = int(np.sum(alive & (ed.origin[:n] == 0)))
                birth_asex = int(np.sum(alive & (ed.origin[:n] == 1)))
                birth_sex = int(np.sum(alive & (ed.origin[:n] == 2)))
            else:
                herb_count = 0
                omni_count = 0
                pred_count = 0
                plant_count = 0
                total_energy = 0.0
                for eid in entity_manager.get_entities_with(Diet, Energy):
                    diet = entity_manager.get_component(eid, Diet)
                    energy = entity_manager.get_component(eid, Energy)
                    if diet is None or energy is None:
                        continue
                    if diet.diet_type == DietType.HERBIVORE:
                        herb_count += 1
                    elif diet.diet_type == DietType.OMNIVORE:
                        omni_count += 1
                    elif diet.diet_type == DietType.PREDATOR:
                        pred_count += 1
                    elif diet.diet_type == DietType.CARNIVOROUS_PLANT:
                        plant_count += 1
                    total_energy += energy.current
                herb, omni, pred, plant = herb_count, omni_count, pred_count, plant_count
                total = herb + omni + pred + plant
                asex = sex = herma = abio = birth_asex = birth_sex = 0

            self._cached_stats = (herb, omni, pred, plant, total, total_energy, asex, sex, herma, abio, birth_asex, birth_sex)
            self._stats_cache_tick = tick

        herb, omni, pred, plant, total, total_energy, asex, sex, herma, abio, birth_asex, birth_sex = self._cached_stats

        lines = [
            f"FPS: {fps:.0f}  Tick: {tick}  Speed: {sim_speed:.1f}x",
            f"Total: {total}  Herb: {herb}  Omni: {omni}  Pred: {pred}  Plant: {plant}",
            f"Asex: {asex}  Sex: {sex}  Herm: {herma}",
            f"Abio: {abio}  Born(a): {birth_asex}  Born(s): {birth_sex}",
            f"Max pop: {self.config.simulation.max_population}",
        ]

        if world is not None:
            phase = _get_day_phase(world)
            lines.append(f"Day phase: {phase} ({world.day_progress:.0%})")

        c = self.cumulative
        if c is not None:
            t = max(tick, 1)
            lines.append(f"All: abio={c['abiogenesis']} b(a)={c['births_asexual']} b(s)={c['births_sexual']} d={c['deaths']}")
            lines.append(f"Avg: abio={c['abiogenesis']/t:.2f} b(a)={c['births_asexual']/t:.2f} b(s)={c['births_sexual']/t:.2f}")

        y = 5
        for line in lines:
            surf = self.font_small.render(line, True, (255, 255, 255))
            bg = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            self.screen.blit(bg, (4, y))
            self.screen.blit(surf, (7, y + 1))
            y += 18

    def render_selected_info(self, entity_manager: EntityManager) -> None:
        if self.selected_entity is None:
            return

        eid = self.selected_entity
        energy = entity_manager.get_component(eid, Energy)
        health = entity_manager.get_component(eid, Health)
        age = entity_manager.get_component(eid, Age)
        diet = entity_manager.get_component(eid, Diet)
        genome_comp = entity_manager.get_component(eid, GenomeComp)

        if energy is None and health is None:
            self.selected_entity = None
            return

        lines = [f"Entity #{eid}"]
        if energy:
            lines.append(f"Energy: {energy.current:.1f}/{energy.max_value:.1f}")
        if health:
            lines.append(f"Health: {health.current:.1f}/{health.max_value:.1f}")
        if age:
            lines.append(f"Age: {age.current}/{age.max_age}")
        if diet:
            lines.append(f"Diet: {diet.diet_type.value}")
        if genome_comp and genome_comp.genome:
            g = genome_comp.genome
            lines.append(f"Size: {g.size:.2f} Speed: {g.speed:.2f}")
            lines.append(f"Vision: {g.vision:.2f} Aggro: {g.aggression:.2f}")

        from components.conditions import Conditions
        conds = entity_manager.get_component(eid, Conditions)
        if conds and conds.effects:
            for eff in conds.effects:
                lines.append(f"  {eff.name}: {eff.duration}t spd\u00d7{eff.speed_mult:.1f} met\u00d7{eff.metabolism_mult:.1f}")

        self._render_panel(lines)

    def render_selected_info_soa(self, entity_data: EntityData, selected_eid: int | None, entity_manager: EntityManager = None) -> None:
        if selected_eid is None:
            return

        idx = entity_data.eid_to_idx.get(selected_eid)
        if idx is None:
            self.selected_entity = None
            return

        if not entity_data.alive[idx]:
            self.selected_entity = None
            return

        from utils.species_namer import get_species_name_from_soa_data

        species = get_species_name_from_soa_data(
            diet_type=int(entity_data.diet_type[idx]),
            repro_type=int(entity_data.repro_type[idx]),
            habitat=int(entity_data.habitat[idx]),
            size_gene=float(entity_data.size_gene[idx]),
            speed_gene=float(entity_data.speed_gene[idx]),
            aggression=float(entity_data.aggression[idx]),
        )

        diet_name = get_diet_name(int(entity_data.diet_type[idx]))
        repro_name = get_repro_name(int(entity_data.repro_type[idx]))
        habitat_name = get_habitat_name(int(entity_data.habitat[idx]))

        lines = [
            f"{species} #{selected_eid}",
            f"Energy: {entity_data.energy[idx]:.1f}/{entity_data.max_energy[idx]:.1f}",
            f"Health: {entity_data.health[idx]:.1f}/{entity_data.max_health[idx]:.1f}",
            f"Age: {entity_data.age[idx]}/{entity_data.max_age[idx]}",
            f"Origin: {get_origin_name(int(entity_data.origin[idx]))}",
            f"Diet: {diet_name}",
            f"Repro: {repro_name}",
            f"Habitat: {habitat_name}",
            f"Aggression: {entity_data.aggression[idx]:.2f}",
            f"Vision: {entity_data.vision[idx]:.1f}",
            f"Metabolism: {entity_data.metabolism[idx]:.2f}",
            f"Pos: ({entity_data.x[idx]:.0f}, {entity_data.y[idx]:.0f})",
            "",
            "--- Genome ---",
            f"Size: {entity_data.size_gene[idx]:.1f} ({(entity_data.size_gene[idx]-3)/7:.2f})",
            f"Speed: {entity_data.speed_gene[idx]:.1f}",
            f"Vision: {entity_data.vision[idx]:.0f}",
            f"Metabolism: {entity_data.metabolism[idx]:.2f}",
            f"Aggression: {entity_data.aggression[idx]:.2f}",
            f"Efficiency: {entity_data.efficiency[idx]:.2f}",
            f"Repro thresh: {entity_data.repro_threshold[idx]:.2f}",
            f"Cooldown: {entity_data.repro_cooldown[idx]}/{entity_data.repro_max_cooldown[idx]}",
        ]

        if int(entity_data.diet_type[idx]) == 3:
            lines.append(f"Photosynth: {entity_data.photosynth[idx]:.2f}")
            lines.append(f"Trap power: {entity_data.aggression[idx]:.2f}")

        lines.append(
            f"Parent: #{int(entity_data.parent_eid[idx])}" if entity_data.parent_eid[idx] >= 0 else "Parent: none"
        )

        from components.conditions import Conditions
        em = entity_manager or self.em
        if em is not None:
            conds = em.get_component(selected_eid, Conditions)
            if conds and conds.effects:
                for eff in conds.effects:
                    lines.append(f"  {eff.name}: {eff.duration}t spd\u00d7{eff.speed_mult:.1f} met\u00d7{eff.metabolism_mult:.1f}")

        self._render_panel(lines)

    def _render_panel(self, lines: list[str]) -> None:
        panel_w = 300
        panel_h = len(lines) * 18 + 10
        panel_x = self.screen.get_width() - panel_w - 10
        panel_y = 5

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        self.screen.blit(bg, (panel_x, panel_y))

        for i, line in enumerate(lines):
            surf = self.font_small.render(line, True, (255, 255, 200))
            self.screen.blit(surf, (panel_x + 5, panel_y + 5 + i * 18))

    def render_help(self) -> None:
        lines = [
            "WASD: Pan | Scroll: Zoom | F: Follow | Click: Select",
            "+/-: Speed | Space: Pause | H: Toggle stats | Esc: Quit",
        ]
        y = self.screen.get_height() - 38
        for line in lines:
            surf = self.font_small.render(line, True, (200, 200, 200))
            bg = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            self.screen.blit(bg, (4, y))
            self.screen.blit(surf, (7, y + 1))
            y += 18


def _get_day_phase(world) -> str:
    p = world.day_progress
    if p < 0.15:
        return "Dawn"
    elif p < 0.55:
        return "Day"
    elif p < 0.70:
        return "Dusk"
    else:
        return "Night"
