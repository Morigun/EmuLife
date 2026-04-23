---
## Goal

Создать проект **EmuLife** — симуляцию цифровой генетической экосистемы с самостоятельной экосистемой, эволюцией и естественным отбором. Мир ~1000x1000 клеток, до ~10000 организмов. ECS-архитектура, Pygame для 2D визуализации с камерой (pan/zoom). Игрок — наблюдатель. Проект прошёл через 4 волны оптимизации производительности, 5 итераций балансировки и 1 волну расширения механик. Есть multiprocessing-режим с shared memory.

## Instructions

- Полная документация механик: `md/mechanics.md`
- Проект использует ECS (Entity-Component-System) архитектуру с гибридным SoA (Structure of Arrays) слоем для горячих данных
- Все системы имеют dual-path: `_update_soa()` (numpy/numba batch) и `_update_ecs()` (Python loop) — SoA-путь используется когда `entity_data is not None`
- При правках нужно учитывать оба пути (SoA и ECS) в каждой системе, а также numba-кернелы в `utils/numba_kernels.py`
- 12 ECS-компонентов, 10 систем, 16-генный геном (используются гены 0–11)
- Два режима запуска: обычный (`python main.py`) и multiprocessing (`python main.py --mp`)
- Git-репозиторий, текущий коммит `1c7d8b2` на ветке `master`

## Discoveries

### Оптимизация производительности (завершено):

**Волна 1** (1 FPS → 30 FPS при 200 org): numpy-массивы вместо Tile-объектов в world.py, surfarray-рендеринг вместо pygame.draw.rect, spatial hash в InteractionSystem вместо O(n²), инкрементальный spatial hash, numpy для food search, ECS query cache.

**Волна 2** (1-2 FPS при 1000 org): Deferred cache invalidation в ECS (ленивая инвалидация), newborn entities list вместо O(n) скана, hungry-only food search с кешом, `query_nearby_into()` с re-used set, NearbyEntity dataclass с кешированным DietType.

**Волна 3** (1-2 FPS при 2000 org): SoA (Structure of Arrays) слой `core/entity_data.py` — параллельные numpy-массивы для всех горячих данных. Numpy-batch для EnergySystem, MovementSystem, AgingSystem, DeathSystem. Staggered processing (1/N entities per tick) для SensorSystem, BehaviorSystem. Batch spatial hash update через numpy mask.

**Волна 4** (zаконмичено, коммит `1c7d8b2`): Numba JIT-кернелы (`utils/numba_kernels.py`) для energy_update, movement_update, aging_update, find_nearest_food, compute_stats. Multiprocessing архитектура: `SimWorker` (отдельный процесс с симуляцией) + `SharedEntityBuffer` (shared memory для SoA-массивов) + `SimulationMP` (рендеринг в главном процессе). CPU limiting через affinity.

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

**Итерация 4** (хищники вымирают): Найдены 3 причины: (1) хищники не охотились при energy>30%, тратя тики на wander; (2) fallback "eat" для хищников направлял к растениям, которые они не могут есть; (3) стартовая энергия 20-35% давала 10-18 тиков жизни при расходе ~2 ед/тик. **Исправлено**: убран low_energy gate для хищников (охотятся сразу), "eat" заменён на "wander" для Predators, child_energy_fraction повышен до 0.45/0.30.

**Итерация 5** (хищники вымирают — энергетика охоты): После поведенческого фикса хищники всё ещё вымирали — нетто −77 энергии за убийство. **Исправлено** 5 параметров: (1) `damage_mult` 15→25 — убийство за ~13 тиков вместо 21; (2) `predation_efficiency` 1.0→1.5 — +50% энергии за убийство; (3) радиус атаки dist_sq 9→25 — 5 тайлов вместо 3; (4) метаболизм хищников ×0.6 скидка — на 40% меньше расход; (5) скорость ×1.3 при hunt — догоняет жертву. Результат: нетто +24 энергии за убийство вместо −77.

**Итерация 6** (хищники вымирают — 6 фиксов в 4 файлах): После итерации 5 хищники всё ещё вымирали из-за множества мелких проблем. **Исправлено** 6 проблем:
1. Low-energy wander для хищников убран — хищники с энергией <30% продолжают охотиться вместо бесцельного блуждания (behavior.py SoA+ECS)
2. Hunt speed ×1.5 (было ×1.3) — чистая скорость сближения 1.5/тик вместо 0.9, погоня за 37 тиков вместо 61
3. Лимит целей hunt [:8] (было [:3]) — хищник видит больше потенциальных жертв в сенсоре (behavior.py SoA+ECS)
4. Порог reproduce 0.7 для хищников (было 0.8) — размножаются при 70% энергии вместо 80% (behavior.py SoA+ECS)
5. Радиус поиска партнёра ×2 для хищников — 120 (было 30) при нормальной популяции, базовый радиус 60 (было 30) (reproduction.py)
6. Energy cap 1.5× max + kill bonus +30 — хищники могут «переедать» до 150% max_energy, плоский бонус +30 за убийство (interaction.py SoA+ECS)

**Итерация 7** (хищники вымирают — скрытые баги v4): После итерации 6 хищники продолжали вымирать. Глубокая трассировка выявила **3 новых бага**. **Исправлено** 3 проблемы:
1. CRITICAL: EnergySystem обрезал 1.5× energy buffer — InteractionSystem давал хищникам energy до 1.5× max, но EnergySystem в тот же тик срезал до 1.0× max, теряя ~46% награды. Energy cap 1.5× max для diet_type==2 добавлен в numba_kernels.py, energy_system.py numpy fallback и ECS fallback
2. HIGH: Hunt/Flee (0,0) замораживал хищника — при мёртвых целях в sensor list методы `_hunt_soa`/`_flee_soa`/`_hunt`/`_flee` возвращали (0,0), хищник останавливался. Добавлен random wander fallback при (0,0) в behavior.py SoA+ECS для hunt и flee
3. MEDIUM: Несоответствие порога размножения — hardcoded `max_energy * 0.7/0.8` в behavior.py не совпадал с фактическим порогом из генома (`repro_threshold * max_energy`). Заменён на чтение `ed.repro_threshold[idx]`/`repro.threshold` в обоих путях

**Итерация 8** (хищники вымирают — структурный фикс v5): После 7 итераций параметрических твиков хищники всё ещё вымирали. Причина — **структурные проблемы дизайна**, не параметры. **Исправлено** 4 проблемы:
1. `predator_child_energy_fraction=0.55` — детёныши хищников получают 55% max_energy (было 30% asexual), что даёт 45 тиков до голода вместо 22 (достаточно для цикла охоты ~33 тика). config.py, reproduction.py, entity_data.py
2. Biomass scavenging для хищников — пассивный доход от biomass на тайле: min(bio×0.05, 3.0)×dt. Хищник рядом с местом смерти получает ~60% метаболизма. numba_kernels.py, energy_system.py (numpy+ECS)
3. Scent hunting — при wander хищники ищут ближайшую не-хищную сущность в радиусе 150 (вместо random walk). Расширенный spatial hash query. behavior.py (SoA `_scent_hunt` + ECS `_scent_hunt_ecs`)
4. Emergency spawn рядом с добычей — emergency predator спавнится рядом со случайным non-predator (±30 единиц), а не в случайном месте мира. abiogenesis.py

**Итерация 9** (хищники вымирают — энергетическая состоятельность v7): После 8 итераций хищники всё ещё вымирали. Глубокий энергетический анализ выявил точную причину: детёныш хищника (63.25 энергии) не может завершить первый цикл охоты (66.15 энергии) при средней дистанции до добычи 43.4 единицы — дефицит −2.90. **Исправлено** 4 проблемы + 1 баг:
1. `predator_child_energy_fraction=0.70` (было 0.55) — детёныши получают 80.5 энергии вместо 63.25, что покрывает полный цикл охоты с запасом 17.5. config.py
2. Hunt speed ×2.0 (было ×1.5) — чистая скорость сближения 3.0 ед/тик вместо 1.5, chase phase в 2 раза быстрее. behavior.py (SoA + ECS)
3. Predator metabolism ×0.5 (было ×0.6) — снижение метаболизма хищников на 50% вместо 40%. numba_kernels.py + energy_system.py (numpy + ECS, где ранее скидка отсутствовала вовсе)
4. «Кровавая трапеза» (`blood_meal_fraction=0.20`) — хищник получает 20% от урона как энергию каждый тик атаки, kill phase net cost снижен с 24.0 до 4.0. config.py + interaction.py (SoA + ECS)
5. Баг: `_update_ecs()` в behavior.py не получал `world` — NameError при `action == "eat"` в ECS-режиме. Добавлен параметр в сигнатуру и вызов.

### Расширение механик (завершено):

**Асексуальное размножение** (геном ген 10): <0.4 = asexual (почкование), ≥0.4 = sexual. Асексуальные — клоны+мутации, без партнёра. Форма: "diamond".

**Среда обитания / водные виды** (геном ген 11): <0.33 = aquatic (только вода), <0.66 = terrestrial (только суша), ≥0.66 = amphibious (везде). Вода имеет еду-планктон (water_max_food=40, water_regen=0.10).

**Абиогенез**: Biomass-массив накапливается при смертях. Молнии (редкие) + biomass spawning создают новых организмов. Emergency spawn убран полностью.

### Multiprocessing + Numba (завершено, коммит `1c7d8b2`):

- `SharedEntityBuffer` (`core/shared_buffers.py`) — shared memory буфер для SoA-массивов, позволяет главному процессу читать данные для рендеринга
- `SimWorker` (`core/sim_worker.py`) — отдельный процесс, запускает симуляцию и пишет SoA-данные в shared memory каждый тик
- `SimulationMP` (в `main.py`) — главный процесс с Pygame рендерингом, читает shared memory
- Numba JIT-кернелы (`utils/numba_kernels.py`) для горячих путей: energy_update, movement_update, aging_update, find_nearest_food, compute_stats
- CPU limiting через affinity (`utils/cpu_limit.py`) — ограничивает numba threads и процесс affinity
- Eid-based selection в multiprocessing — выбор сущности по eid, данные читаются из SoA каждый тик
- Научные имена видов (`utils/species_namer.py`) — 18 родов × 27 видов = 486 биноменов на основе генома

## Accomplished

**Завершено (коммит `1c7d8b2`):**
- Полный каркас проекта (ECS, 12 компонентов, 10 систем, рендеринг, камера, мир)
- Все 4 волны оптимизации производительности (SoA + stagger + numpy batch + numba JIT)
- 9 итераций балансировки (size_gene bug, freeze/предаторы, экспоненциальный рост, behavior fix, энергетика охоты, predator viability v3, скрытые баги v4, структурный фикс v5, энергетическая состоятельность v7)
- Multiprocessing режим (SimWorker + SharedEntityBuffer + SimulationMP)
- Numba JIT-кернелы для горячих путей
- CPU core limiting через affinity
- Адаптивный радиус клика + eid-based selection для multiprocessing
- Чтение статистики выбранной сущности из SoA каждый тик
- Научные имена видов (18 родов × 27 видов = 486 биноменов)
- Асексуальное размножение (почкование) + сексуальное с кроссовером
- 3 среды обитания: aquatic/terrestrial/amphibious с водной едой
- Абиогенез: biomass decay + lightning spawning + biomass spawning
- Плотностной контроль размножения (соседи + глобальный лимит)
- Полная документация механик `md/mechanics.md`

## Relevant files / directories

```
E:\Projects\Python\EmuLife\
├── main.py                          # Simulation + SimulationMP, entity selection, game loop
├── config.py                        # Все параметры: EnergyConfig, ReproductionConfig, AbiogenesisConfig, TileConfig
├── requirements.txt                 # pygame, numpy, opensimplex, numba
├── run.bat                          # Создание venv + установка + запуск
├── md/
│   ├── current_state.md             # Этот файл
│   └── mechanics.md                 # Полная документация всех механик
├── core/
│   ├── ecs.py                       # EntityManager с deferred cache, Component, System
│   ├── world.py                     # Numpy-массивы: tile_types, food_values, max_foods, regen_rates, walkable_mask, biomass
│   ├── entity_data.py               # SoA: 28 numpy-массивов (x,y,dx,dy,energy,health,age,genome data,habitat,repro_type,eids...)
│   ├── genome.py                    # 16-gene genome (0-11 used), crossover, mutate
│   ├── camera.py                    # Pan/zoom, world↔screen, follow entity
│   ├── shared_buffers.py            # SharedEntityBuffer — shared memory для multiprocessing
│   └── sim_worker.py                # SimWorker — subprocess для симуляции
├── components/                      # 12 dataclass-компонентов
│   ├── position.py, velocity.py, energy.py, health.py, age.py
│   ├── genome_comp.py, appearance.py, sensor.py, metabolism.py
│   ├── diet.py                      # DietType: HERBIVORE, OMNIVORE, PREDATOR
│   ├── reproduction.py              # threshold, cooldown, max_cooldown, repro_type
│   └── habitat.py                   # habitat_type: "aquatic"|"terrestrial"|"amphibious"
├── systems/
│   ├── sensor_system.py             # Staggered (BATCHES=2), numpy food search, habitat-aware, hungry-only
    │   ├── behavior.py                  # Staggered, дерево (flee/hunt/eat/reproduce/wander), hunt speed ×2.0, scent hunting r=150, predator repro 0.7
│   ├── movement.py                  # Numpy-batch/numba, habitat-aware walkable mask, bounce, moved_mask
    │   ├── interaction.py               # Spatial hash, predator damage_mult=25, attack radius=5, energy cap 1.5× max, kill bonus +30, blood_meal 20%
│   ├── energy_system.py             # Numba-batch/numba, predator metabolism ×0.5, habitat-aware can_eat, 1/N food share, biomass scavenging
│   ├── aging.py                     # Numba kernel, max_age=800+size*200, health=0 on death
│   ├── reproduction.py              # Asexual/sexual, density check, birth cap, predator partner radius ×2
│   ├── death.py                     # Numpy-batch, biomass deposit on death, ECS+SoA+spatial cleanup
│   ├── abiogenesis.py               # Biomass spawning + lightning strikes, emergency predator spawn near prey, lightning_events for render
│   └── world_update.py              # Food regen (every 2 ticks), biomass decay
├── rendering/
│   ├── renderer.py                  # Surfarray world render, biomass overlay, SoA entity render (diamond/square/triangle/circle)
│   ├── ui.py                        # Stats cache, selected entity info (SoA-based), species name display
│   └── minimap.py                   # Throttled minimap
└── utils/
    ├── numba_kernels.py             # Numba JIT: energy_update, movement_update, aging_update, find_nearest_food, compute_stats
    ├── species_namer.py             # Научные имена видов (18 родов × 27 видов)
    ├── cpu_limit.py                 # CPU affinity + numba thread control
    ├── spatial_hash.py              # query_nearby_into/excluding_into с re-used set
    └── logger.py
```
---
