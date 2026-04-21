---
## Goal

Создать проект **EmuLife** — симуляцию цифровой генетической экосистемы с самостоятельной экосистемой, эволюцией и естественным отбором. Мир ~1000x1000 клеток, до ~10000 организмов. ECS-архитектура, Pygame для 2D визуализации с камерой (pan/zoom). Игрок — наблюдатель. Проект прошёл через 3 волны оптимизации производительности и 2 итерации балансировки.

## Instructions

- План проекта хранится в `C:\Users\gungr\.local\share\kilo\plans\1776710119278-eager-meadow.md` — содержит все предыдущие оптимизации и текущий план балансировки
- Проект использует ECS (Entity-Component-System) архитектуру с гибридным SoA (Structure of Arrays) слоем для горячих данных
- Все системы имеют dual-path: `_update_soa()` (numpy batch) и `_update_ecs()` (оригинальный Python loop) — SoA-путь используется когда `entity_data is not None`
- При правках нужно учитывать оба пути (SoA и ECS) в каждой системе
- Config-твики уже применены: `reproduction_energy_cost=0.3`, `child_energy_fraction=0.35`, `land_regen=0.08`, `forest_regen=0.18`, `max_age baseline=800`

## Discoveries

### Оптимизация производительности (завершено):

**Волна 1** (1 FPS → 30 FPS при 200 org): numpy-массивы вместо Tile-объектов в world.py, surfarray-рендеринг вместо pygame.draw.rect, spatial hash в InteractionSystem вместо O(n²), инкрементальный spatial hash, numpy для food search, ECS query cache.

**Волна 2** (1-2 FPS при 1000 org): Deferred cache invalidation в ECS (ленивая инвалидация), newborn entities list вместо O(n) скана, hungry-only food search с кешом, `query_nearby_into()` с re-used set, NearbyEntity dataclass с кешированным DietType.

**Волна 3** (1-2 FPS при 2000 org): SoA (Structure of Arrays) слой `core/entity_data.py` — параллельные numpy-массивы для всех горячих данных. Numpy-batch для EnergySystem, MovementSystem, AgingSystem, DeathSystem. Staggered processing (1/N entities per tick) для SensorSystem, BehaviorSystem. Batch spatial hash update через numpy mask.

### Балансировка (завершено):

**Итерация 1** (~2000 тиков → вымирание): **Критический баг** — `ed.size_gene` хранил пиксельный размер (3-10) вместо значения гена (0-1). EnergySystem SoA-путь расходовал в 7-13x больше энергии. **Исправлено**: `raw_size = (ed.size_gene[s] - 3.0) / 7.0`. Также InteractionSystem stagger был убран (хищники атакуют каждый тик).

**Итерация 2** (~2500 тиков → вымирание): Найдены и **исправлены** 6 проблем:
1. **(КРИТИЧЕСКАЯ)** При action=="eat" и `nearest_food_pos=None`, организм замерзает → добавлен wander fallback (`behavior.py:66-68`)
2. Хищники убыточны → `predation_efficiency=1.0` (`config.py:32`), `damage_mult=15.0` для хищников (`interaction.py:70`)
3. Еда из ничего при кластеризации → справедливое распределение 1/N на тайл (`energy_system.py:43-56`)
4. Death spiral → emergency spawn при pop < 30 (`main.py:113-135`) + динамический радиус поиска партнёра при pop < 100 (`reproduction.py:242-244`)
5. Устаревшее поведение → `STAGGER_BATCHES=2` (`sensor_system.py:15`, `behavior.py:16`)
6. Уход с кормовых тайлов → проверка `food > 5` под ногами (`behavior.py:136-140`)

## Accomplished

**Завершено:**
- Полный каркас проекта (ECS, компоненты, системы, рендеринг, камера, мир)
- Все 3 волны оптимизации производительности (SoA + stagger + numpy batch)
- Исправление критического бага с size_gene в EnergySystem
- Config-твики для устойчивости (child_energy, repro_cost, regen, max_age)
- Убран stagger из InteractionSystem

**Следующий шаг — тестирование:**
- Запустить симуляцию и проверить стабильность популяции (цель: >10000 тиков без вымирания)
- При необходимости — третья итерация балансировки

## Relevant files / directories

```
E:\Projects\Python\EmuLife\
├── main.py                          # Точка входа, игровой цикл, spatial hash update
├── config.py                        # Все настраиваемые параметры (текущие: repro_cost=0.3, child=0.35, regen↑)
├── requirements.txt                 # pygame, numpy, opensimplex
├── core/
│   ├── ecs.py                       # EntityManager с deferred cache, Component, System
│   ├── world.py                     # Numpy-массивы: tile_types, food_values, max_foods, regen_rates, walkable_mask
│   ├── entity_data.py               # SoA хранилище: 25+ numpy-массивов, eid↔idx mapping
│   ├── genome.py                    # 16-gene genome, crossover, mutate
│   └── camera.py                    # Pan/zoom, world↔screen, follow entity
├── components/                      # 11 dataclass-компонентов (Position, Velocity, Energy, Health, Age, GenomeComp, Appearance, Sensor, Metabolism, Diet, Reproduction)
├── systems/
│   ├── sensor_system.py             # Staggered (BATCHES=4), numpy food search, hungry-only, cache
│   ├── behavior.py                  # Staggered, решающее дерево (flee/hunt/eat/reproduce/wander) — **СОДЕРЖИТ БАГ FREEZE**
│   ├── movement.py                  # Numpy-batch SoA, moved_mask для spatial hash
│   ├── interaction.py               # НЕ staggered, spatial hash, damage/predation — **НУЖНА БАЛАНСИРОВКА**
│   ├── energy_system.py             # Numpy-batch SoA, eat/drain/cost — **НУЖЕН FIX food distribution**
│   ├── aging.py                     # Numpy-batch SoA, max_age=800+ baseline
│   ├── reproduction.py              # SoA cooldown batch + per-entity, _find_partner radius=30 — **НУЖЕН DYNAMIC RADIUS**
│   ├── death.py                     # Numpy-batch find dead, remove from ECS+SoA+spatial hash
│   └── world_update.py              # numpy regenerate_all
├── rendering/
│   ├── renderer.py                  # surfarray world render, SoA entity render
│   ├── ui.py                        # Stats cache (30 tick interval), selected entity info
│   └── minimap.py                   # Throttled (10 frames), numpy surface
└── utils/
    ├── spatial_hash.py              # query_nearby_into/query_nearby_excluding_into с re-used set
    └── logger.py
```

План со всей историей: `C:\Users\gungr\.local\share\kilo\plans\1776710119278-eager-meadow.md`
---