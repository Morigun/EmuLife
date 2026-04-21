---
## Goal

Создать проект **EmuLife** — симуляцию цифровой генетической экосистемы с самостоятельной экосистемой, эволюцией и естественным отбором. Мир ~1000x1000 клеток, до ~10000 организмов. ECS-архитектура, Pygame для 2D визуализации с камерой (pan/zoom). Игрок — наблюдатель. Проект прошёл через 3 волны оптимизации производительности, 3 итерации балансировки и 1 волну расширения механик.

## Instructions

- Полная документация механик: `md/mechanics.md`
- Проект использует ECS (Entity-Component-System) архитектуру с гибридным SoA (Structure of Arrays) слоем для горячих данных
- Все системы имеют dual-path: `_update_soa()` (numpy batch) и `_update_ecs()` (оригинальный Python loop) — SoA-путь используется когда `entity_data is not None`
- При правках нужно учитывать оба пути (SoA и ECS) в каждой системе
- 12 ECS-компонентов, 10 систем, 16-генный геном (используются гены 0–11)
- Git-репозиторий инициализирован, начальный коммит `9dc243b`

## Discoveries

### Оптимизация производительности (завершено):

**Волна 1** (1 FPS → 30 FPS при 200 org): numpy-массивы вместо Tile-объектов в world.py, surfarray-рендеринг вместо pygame.draw.rect, spatial hash в InteractionSystem вместо O(n²), инкрементальный spatial hash, numpy для food search, ECS query cache.

**Волна 2** (1-2 FPS при 1000 org): Deferred cache invalidation в ECS (ленивая инвалидация), newborn entities list вместо O(n) скана, hungry-only food search с кешом, `query_nearby_into()` с re-used set, NearbyEntity dataclass с кешированным DietType.

**Волна 3** (1-2 FPS при 2000 org): SoA (Structure of Arrays) слой `core/entity_data.py` — параллельные numpy-массивы для всех горячих данных. Numpy-batch для EnergySystem, MovementSystem, AgingSystem, DeathSystem. Staggered processing (1/N entities per tick) для SensorSystem, BehaviorSystem. Batch spatial hash update через numpy mask.

### Балансировка (завершено):

**Итерация 1** (~2000 тиков → вымирание): **Критический баг** — `ed.size_gene` хранил пиксельный размер (3-10) вместо значения гена (0-1). EnergySystem SoA-путь расходовал в 7-13x больше энергии. **Исправлено**: `raw_size = (ed.size_gene[s] - 3.0) / 7.0`. Также InteractionSystem stagger был убран (хищники атакуют каждый тик).

**Итерация 2** (~2500 тиков → вымирание): Найдены и **исправлены** 6 проблем: freeze при отсутствии еды (wander fallback), рентабельность хищников (damage_mult=15), справедливое распределение еды 1/N, emergency spawn при pop<30, STAGGER_BATCHES=2, проверка food>5 под ногами.

**Итерация 3** ("бессмертные 30" + экспоненциальный рост): После добавления асексуального размножения популяция начала расти экспоненциально (80 асексуальных × каждые 60 тиков = x2 за секунду). **Исправлено** 8 проблем:
1. Кулдаун асексуального: 60→250 тиков (`ReproductionConfig.asexual_cooldown`)
2. Кулдаун сексуального: 60→150 тиков (`ReproductionConfig.sexual_cooldown`)
3. Стоимость асексуального: 30%→50% (`EnergyConfig.asexual_reproduction_energy_cost`)
4. Child energy асексуального: 0.35→0.2 (`EnergyConfig.asexual_child_energy_fraction`)
5. Плотностной контроль: >8 соседей в радиусе 40 = блокировка (`ReproductionConfig.density_*`)
6. Глобальный лимит рождений: 5/тик (`ReproductionConfig.max_births_per_tick`)
7. Behavior: reproduce только при energy>80% max, асексуальные НЕ выбирают "reproduce" (`behavior.py:106-114`)
8. Абиогенез: lightning 0.002→0.0005, biomass_spawn 0.0001→0.00003

### Расширение механик (завершено):

**Асексуальное размножение** (геном ген 10): <0.4 = asexual (почкование), ≥0.4 = sexual. Асексуальные — клоны+мутации, без партнёра. Форма: "diamond".

**Среда обитания / водные виды** (геном ген 11): <0.33 = aquatic (только вода), <0.66 = terrestrial (только суша), ≥0.66 = amphibious (везде). Вода имеет еду-планктон (water_max_food=40, water_regen=0.10).

**Абиогенез**: Biomass-массив накапливается при смертях. Молнии (редкие) + biomass spawning создают новых организмов. Emergency spawn убран полностью.

## Accomplished

**Завершено:**
- Полный каркас проекта (ECS, 12 компонентов, 10 систем, рендеринг, камера, мир)
- Все 3 волны оптимизации производительности (SoA + stagger + numpy batch)
- 3 итерации балансировки (size_gene bug, freeze/предаторы, экспоненциальный рост)
- Асексуальное размножение (почкование) + сексуальное с кроссовером
- 3 среды обитания: aquatic/terrestrial/amphibious с водной едой
- Абиогенез: biomass decay + lightning spawning + biomass spawning
- Плотностной контроль размножения (соседи + глобальный лимит)
- Полную документацию механик `md/mechanics.md`
- Git-репозиторий с начальным коммитом

## Relevant files / directories

```
E:\Projects\Python\EmuLife\
├── main.py                          # Точка входа, игровой цикл, spatial hash update, lightning render
├── config.py                        # Все параметры: EnergyConfig, ReproductionConfig, AbiogenesisConfig, TileConfig
├── requirements.txt                 # pygame, numpy, opensimplex
├── run.bat                          # Создание venv + установка + запуск
├── md/
│   ├── current_state.md             # Этот файл
│   └── mechanics.md                 # Полная документация всех механик
├── core/
│   ├── ecs.py                       # EntityManager с deferred cache, Component, System
│   ├── world.py                     # Numpy-массивы: tile_types, food_values, max_foods, regen_rates, walkable_mask, biomass
│   ├── entity_data.py               # SoA: 27 numpy-массивов (x,y,dx,dy,energy,health,age,genome data,habitat,repro_type...)
│   ├── genome.py                    # 16-gene genome (0-11 used), crossover, mutate
│   └── camera.py                    # Pan/zoom, world↔screen, follow entity
├── components/                      # 12 dataclass-компонентов
│   ├── position.py, velocity.py, energy.py, health.py, age.py
│   ├── genome_comp.py, appearance.py, sensor.py, metabolism.py
│   ├── diet.py                      # DietType: HERBIVORE, OMNIVORE, PREDATOR
│   ├── reproduction.py              # threshold, cooldown, max_cooldown, repro_type
│   └── habitat.py                   # habitat_type: "aquatic"|"terrestrial"|"amphibious"
├── systems/
│   ├── sensor_system.py             # Staggered (BATCHES=2), numpy food search, habitat-aware, hungry-only
│   ├── behavior.py                  # Staggered, дерево (flee/hunt/eat/reproduce/wander), asexual skip reproduce
│   ├── movement.py                  # Numpy-batch, habitat-aware walkable mask, bounce, moved_mask
│   ├── interaction.py               # Spatial hash, predator/omnivore damage, predation energy gain
│   ├── energy_system.py             # Numpy-batch, habitat-aware can_eat, 1/N food share
│   ├── aging.py                     # Numpy-batch, max_age=800+size*200, health=0 on death
│   ├── reproduction.py              # Asexual/sexual, density check, birth cap, dynamic partner radius
│   ├── death.py                     # Numpy-batch, biomass deposit on death, ECS+SoA+spatial cleanup
│   ├── abiogenesis.py               # Biomass spawning + lightning strikes, lightning_events for render
│   └── world_update.py              # Food regen (every 2 ticks), biomass decay
├── rendering/
│   ├── renderer.py                  # Surfarray world render, biomass overlay, SoA entity render (diamond/square/triangle/circle)
│   ├── ui.py                        # Stats cache, selected entity info
│   └── minimap.py                   # Throttled minimap
└── utils/
    ├── spatial_hash.py              # query_nearby_into/excluding_into с re-used set
    └── logger.py
```
---
