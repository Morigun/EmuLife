# Механики симуляции EmuLife

## 1. Обзор

EmuLife — симуляция цифровой генетической экосистемы. Игрок выступает в роли наблюдателя (камера, пауза, ускорение). Мир представляет собой сетку **1000×1000** тайлов, каждый тайл имеет тип биома и запас еды. Организмы обладают геномом из 16 генов и управляются AI на основе дерева решений.

**Параметры симуляции** (`SimulationConfig`):

| Параметр | Значение | Ключ конфига |
|----------|----------|-------------|
| Tick rate | 60 тиков/сек | `simulation.tick_rate` |
| Начальная популяция | 200 | `simulation.initial_population` |
| Максимум популяции | 10000 | `simulation.max_population` |
| Размер экрана | 1280×720 | `screen.width/height` |
| FPS cap | 60 | `screen.fps_cap` |

---

## 2. Мир (World)

### 2.1 Генерация

Мир генерируется через **opensimplex noise** с двумя шумовыми сетками:
- **Elevation** (масштаб `0.005`) — высота
- **Moisture** (масштаб `0.008`) — влажность

Обе нормализуются из `[-1, 1]` в `[0, 1]`. Правила биомов:

| Условие | Биом | `TileType` |
|---------|------|-----------|
| elevation < 0.35 | Water | 0 |
| 0.35 ≤ elevation < 0.65 AND moisture > 0.55 | Forest | 2 |
| 0.35 ≤ elevation < 0.65 AND moisture < 0.3 | Desert | 3 |
| 0.35 ≤ elevation < 0.65 AND остальное | Land | 1 |
| elevation ≥ 0.65 AND moisture > 0.5 | Forest | 2 |
| elevation ≥ 0.65 AND остальное | Land | 1 |

`walkable_mask = tile_types != TileType.WATER` (вода непроходима для `is_walkable`).

### 2.2 Еда и биомы

Начальная еда на тайле = `max_food * 0.5`. Параметры каждого биома (`TileConfig`):

| Биом | `max_food` | `regen_rate` |
|------|-----------|-------------|
| Water | 40.0 | 0.10 |
| Land | 50.0 | 0.08 |
| Forest | 80.0 | 0.18 |
| Desert | 20.0 | 0.01 |

Регенерация (`WorldUpdateSystem`) каждые **2 тика**: `food = min(food + regen_rate, max_food)`.

### 2.3 Biomass

`world.biomass` — массив `float32[height, width]`. Накапливается при смерти организмов. Убывает каждый тик: `biomass *= (1.0 - biomass_decay_rate)` = `×0.995`.

Используется для абиогенеза (см. раздел 12).

---

## 3. Геном

### 3.1 Гены

16 генов (индексы 0–15), все в диапазоне `[0, 1]`. Гены 12–15 зарезервированы и не используются.

| Индекс | Свойство (`Genome`) | Влияние | Производное значение |
|--------|--------------------|---------|--------------------|
| 0 | `size` | Физический размер | `size_val = 3.0 + gene × 7.0` → [3, 10] |
| 1 | `speed` | Скорость движения | `max_speed = 1.0 + gene × 4.0` → [1, 5] |
| 2 | `vision` | Радиус обзора | `vision = 20.0 + gene × 80.0` → [20, 100] |
| 3 | `metabolism_gene` | Скорость метаболизма | `metab_rate = 0.5 + gene × 2.0` → [0.5, 2.5] |
| 4 | `diet_type_value` | Тип питания | <0.33: Herbivore, <0.66: Omnivore, ≥0.66: Predator |
| 5 | `repro_threshold_gene` | Порог размножения | `0.5 + gene × 0.4` → [0.5, 0.9] |
| 6 | `aggression` | Агрессия | Прямое значение [0, 1] |
| 7 | `r_color` | Красный канал | `int(gene × 255)` |
| 8 | `g_color` | Зелёный канал | `int(gene × 255)` |
| 9 | `b_color` | Синий канал | `int(gene × 255)` |
| 10 | `reproduction_type` | Тип размножения | <0.4: asexual, ≥0.4: sexual |
| 11 | `habitat` | Среда обитания | <0.33: aquatic, <0.66: terrestrial, ≥0.66: amphibious |
| 12–15 | — | Зарезервированы | — |

### 3.2 Мутация

`GenomeConfig`: `mutation_rate = 0.1`, `mutation_strength = 0.1`.

Для каждого гена: с вероятностью 0.1 прибавляется `uniform(-0.1, +0.1)`, результат клэмпится в `[0, 1]`.

### 3.3 Кроссовер

Одноточечный: случайная точка разрыва `breakpoint ∈ [0, length]`. Потомок получает гены `parent1[0:break] + parent2[break:]`.

---

## 4. Организм — компоненты и параметры

### 4.1 ECS-компоненты (12 штук)

| Компонент | Поля | Источник в SoA |
|-----------|------|---------------|
| `Position` | `x`, `y` | `ed.x[idx]`, `ed.y[idx]` |
| `Velocity` | `dx`, `dy` | `ed.dx[idx]`, `ed.dy[idx]` |
| `Energy` | `current`, `max_value` | `ed.energy[idx]`, `ed.max_energy[idx]` |
| `Health` | `current`, `max_value` | `ed.health[idx]`, `ed.max_health[idx]` |
| `Age` | `current`, `max_age` | `ed.age[idx]`, `ed.max_age[idx]` |
| `GenomeComp` | `genome: Genome` | (ECS-only, данные в SoA) |
| `Appearance` | `r`, `g`, `b`, `size`, `shape` | `ed.r/g/b[idx]`, `ed.size_gene[idx]` |
| `Sensor` | `radius`, `nearby_entities`, `nearest_food_pos`, `food_cache_tick`, `food_cache_interval` | `ed.vision[idx]` |
| `Metabolism` | `rate` | `ed.metabolism[idx]` |
| `Diet` | `diet_type: DietType`, `efficiency` | `ed.diet_type[idx]`, `ed.efficiency[idx]` |
| `Reproduction` | `threshold`, `cooldown`, `max_cooldown`, `repro_type` | `ed.repro_threshold/cooldown/max_cooldown/repro_type[idx]` |
| `Habitat` | `habitat_type: str` | `ed.habitat[idx]` |

### 4.2 Производные характеристики

Вычисляются при создании организма (`entity_data.add()`):

| Характеристика | Формула | Диапазон |
|---------------|---------|----------|
| `size_val` | `3.0 + gene[0] × 7.0` | [3, 10] |
| `max_speed` | `1.0 + gene[1] × 4.0` | [1, 5] |
| `vision` | `20.0 + gene[2] × 80.0` | [20, 100] |
| `metabolism` | `0.5 + gene[3] × 2.0` | [0.5, 2.5] |
| `max_energy` | `50.0 + size_val × 10.0` | [80, 150] |
| `max_health` | `30.0 + size_val × 10.0` | [60, 130] |
| `max_age` | `int(800 + size_val × 200)` | [1400, 2800] тиков |
| `repro_threshold` | `0.5 + gene[5] × 0.4` | [0.5, 0.9] |
| `efficiency` | `0.8 + aggression × 0.2` | [0.8, 1.0] |

Форма организма (`Appearance.shape`):
- Asexual → `"diamond"`
- Predator → `"triangle"`
- Остальные → `"circle"`

---

## 5. Система питания (Diet)

### 5.1 Типы питания

| Тип | `diet_type` (int) | Ген `gene[4]` | Поведение |
|-----|-------------------|--------------|-----------|
| Herbivore | 0 | < 0.33 | Ест только растительную еду |
| Omnivore | 1 | 0.33–0.66 | Ест растительную еду + охотится на Herbivore |
| Predator | 2 | ≥ 0.66 | Не ест растения, охотится на всех не-хищников |

### 5.2 Механика питания (EnergySystem)

Обработка в numpy-batch (`_update_soa`):

**Маска `can_eat`**: организм может есть, если:
1. `alive` = true
2. `diet_type != 2` (не Predator)
3. `food > 0` на его тайле
4. Habitat совместим:
   - aquatic (0) → только вода (`is_water`)
   - terrestrial (1) → только суша (`~is_water`)
   - amphibious (2) → везде

**Распределение еды**: еда на тайле делится поровну между всеми организмами на нём:
```
share = food[ y, x ] / max(entity_count_on_tile, 1)
eat_amount = min(energy_from_food * dt, share)
```
`energy_from_food = 15.0` (`EnergyConfig`).

**Получение энергии**: `energy += eat_amount × efficiency`

**Уменьшение еды**: `food[y,x] -= eat_amount` (clamped ≥ 0).

---

## 6. Среда обитания (Habitat)

### 6.1 Типы среды

| Тип | `habitat` (int) | Ген `gene[11]` | Проходимые тайлы |
|-----|-----------------|----------------|-----------------|
| aquatic | 0 | < 0.33 | Только вода |
| terrestrial | 1 | 0.33–0.66 | Только суша (не вода) |
| amphibious | 2 | ≥ 0.66 | Все тайлы |

### 6.2 Правила движения (MovementSystem)

В SoA-пути (`_update_soa`):

```python
walkable = (
    (terrestrial & ~is_water)
    | (aquatic & is_water)
    | amphibious
) & alive
```

- Если `walkable` → позиция обновляется: `pos += velocity × dt` (clamped в границы мира).
- Если `~walkable` → **отскок**: `velocity *= -0.5`, позиция не меняется.

---

## 7. Размножение

### 7.1 Общие условия

Для любого размножения организм должен:
1. Быть живым
2. Иметь `repro_cooldown == 0` (кулдаун уменьшается на 1 каждый тик)
3. Иметь `energy ≥ repro_threshold × max_energy`
4. Иметь ≤ 8 соседей в радиусе 40 (плотностной контроль)
5. Глобальный лимит: ≤ 5 рождений/тик
6. `entity_count < max_population` (10000)

### 7.2 Асексуальное размножение

Условие: `gene[10] < 0.4` → `repro_type = "asexual"` (SoA: `repro_type = 0`).

| Параметр | Значение |
|----------|----------|
| Кулдаун | 250 тиков |
| Стоимость энергии | 50% энергии родителя (`asexual_reproduction_energy_cost = 0.5`) |
| Энергия потомка | `max_energy × 0.2` (`asexual_child_energy_fraction`) |
| Геном потомка | Клон родителя + мутация |
| Партнёр | Не требуется |
| Спавн | ±5 единиц от родителя |

Асексуальные организмы **не могут выбрать действие "reproduce"** в BehaviorSystem — размножение происходит автоматически в ReproductionSystem при выполнении условий.

### 7.3 Сексуальное размножение

Условие: `gene[10] ≥ 0.4` → `repro_type = "sexual"` (SoA: `repro_type = 1`).

| Параметр | Значение |
|----------|----------|
| Кулдаун | 150 тиков |
| Стоимость энергии | 30% от обоих родителей (`reproduction_energy_cost = 0.3`) |
| Энергия потомка | `max_energy × 0.35` (`child_energy_fraction`) |
| Геном потомка | Кроссовер(parent1, parent2) + мутация |
| Партнёр | Требуется (та же diet_type, не в кулдауне) |
| Спавн | ±5 единиц от родителя |

### 7.4 Поиск партнёра (`_find_partner`)

Динамический радиус поиска:
```
search_radius = min(200.0, 30.0 + max(0.0, 100.0 - entity_count) × 2.0)
```
При 100+ организмов: радиус = 30. При 0 организмов: радиус = 200.

Критерии партнёра:
- Та же `diet_type`
- Не в кулдауне (`repro.cooldown <= 0`)
- Первый подходящий из spatial hash query

---

## 8. Поведение (AI)

### 8.1 Дерево решений (`_decide_soa`)

Приоритеты действий (в порядке проверки):

```
1. PREDATOR + видит не-хищника + low_energy → "hunt"
2. HERBIVORE/OMNIVORE + видит PREDATOR → "flee"
3. low_energy → "eat"
4. sexual + cooldown==0 + energy > 80% max_energy → "reproduce"
5. Иначе → "wander"
```

Где `low_energy = energy < max_energy × 0.3`.

**Асексуальные организмы** (`repro_type == 0`) не имеют шага 4 — действие "reproduce" доступно только сексуальным.

### 8.2 Реализация действий

| Действие | Логика |
|----------|--------|
| `flee` | Сумма векторов от хищников (до 5 ближайших) → направление прочь |
| `hunt` | Направление к ближайшему не-хищнику |
| `eat` | К ближайшей еде (`sensor.nearest_food_pos`), если на тайле `food > 5` — стоять. Если еды нет — random wander |
| `reproduce` | К ближайшему партнёру той же diet_type с `cooldown == 0`. Если нет — random wander |
| `wander` | `random.uniform(-1, 1)` по обеим осям |

Вектор нормализуется и умножается на `max_speed`: `velocity = normalize(target) × max_speed`.

### 8.3 Stagger

`STAGGER_BATCHES = 2` — обрабатывается **половина** организмов за тик (чередование по `idx % 2`).

---

## 9. Взаимодействие (хищничество)

### 9.1 Условия атаки

Атакующий перебирает ближайшие сущности из spatial hash (радиус 5). Проверка дальности: `dist_sq > 9` (т.е. расстояние > 3 тайлов) — пропускается.

| Атакующий | Цель | `damage_mult` |
|-----------|------|--------------|
| Predator (2) → | любой не-Predator | 15.0 |
| Omnivore (1) → | только Herbivore (0) | 5.0 |

Herbivore **никогда не атакуют**.

### 9.2 Расчёт урона

```
damage = raw_size × aggression × damage_mult × dt
```
Где `raw_size = (size_gene - 3.0) / 7.0` ∈ [0, 1].

### 9.3 Энергия при убийстве

Если `health[target] ≤ 0`:
- Predator: `gain = victim_energy × predation_efficiency` (1.0 = 100%)
- Omnivore: `gain = victim_energy × predation_efficiency × 0.5` (50%)

Энергия атакующего: `energy = min(max_energy, energy + gain)`.

---

## 10. Энергетика

### 10.1 Расход энергии

```
cost = metabolism × (0.5 + raw_size) × dt + speed × raw_size × 0.5 × dt
```
Где:
- `raw_size = (size_gene - 3.0) / 7.0` ∈ [0, 1]
- `metabolism` = `0.5 + gene[3] × 2.0` ∈ [0.5, 2.5]
- `speed` = `sqrt(dx² + dy²)` (текущая скорость)

### 10.2 Приём энергии

См. раздел 5.2. Формула:
```
energy += eat_amount × efficiency
```
Где `eat_amount = min(15.0 × dt, food / entity_count_on_tile)`.

### 10.3 Ограничения

- `energy` ограничена сверху: `min(energy, max_energy)`
- `energy ≤ 0` → смерть

---

## 11. Старение и смерть

### 11.1 Старение (`AgingSystem`)

Каждый тик: `age += 1` (для живых).

При `age >= max_age`: `health = 0` → смерть на следующем тике DeathSystem.

`max_age = int(800 + size_val × 200)`:
- Минимум (size_val=3): 1400 тиков
- Максимум (size_val=10): 2800 тиков

### 11.2 Смерть (`DeathSystem`)

Условия смерти: `health ≤ 0` **или** `energy ≤ 0`.

При смерти:
1. `biomass[y, x] += size_gene × 5.0` (возврат биомассы в мир)
2. Удаление из spatial hash
3. Удаление из SoA (`entity_data.remove()`)
4. Удаление из ECS (`em.remove_entity()`)

---

## 12. Абиогенез

### 12.1 Biomass spawning

Для каждого тайла где `biomass > threshold` (10.0):
```
probability = biomass[y,x] × 0.00003
```
Если `random() < probability` → создаётся организм на этом тайле.

### 12.2 Lightning (молния)

Вероятность за тик: `0.0005` (`lightning_chance`).

При срабатывании:
1. Случайная позиция `(lx, ly)` проверяется на `walkable_mask` (до 100 попыток).
2. `biomass[y0:y1, x0:x1] += 50.0` в радиусе 5 тайлов (`lightning_biomass_boost`).
3. Создаётся организм с `energy_fraction = 0.3`.

### 12.3 Новый организм абиогенеза

- Случайный геном (`Genome.random_instance()`)
- Если тайл не walkable (вода): `gene[11]` принудительно устанавливается < 0.33 (aquatic)
- `energy_fraction = 0.3`
- `parent_energy_sum = 100.0`

### 12.4 Конфиг (`AbiogenesisConfig`)

| Параметр | Значение | Ключ |
|----------|----------|------|
| Biomass decay rate | 0.005 | `biomass_decay_rate` |
| Biomass threshold | 10.0 | `biomass_threshold` |
| Spawn chance multiplier | 0.00003 | `biomass_spawn_chance` |
| Lightning chance | 0.0005 | `lightning_chance` |
| Lightning biomass boost | 50.0 | `lightning_biomass_boost` |
| Lightning radius | 5 | `lightning_radius` |

---

## 13. Порядок систем (game loop)

Системы выполняются в строгом порядке каждый тик (`main.py: systems[]`):

```
1.  SensorSystem         — обновление сенсоров, поиск ближайшей еды
2.  BehaviorSystem       — принятие решения AI, установка velocity
3.  MovementSystem       — перемещение, проверка проходимости
4.  InteractionSystem    — хищничество, нанесение урона
5.  EnergySystem         — расход/приём энергии, питание
6.  AgingSystem          — увеличение возраста, смерть от старости
7.  ReproductionSystem   — создание потомков
8.  DeathSystem          — удаление мёртвых организмов
9.  AbiogenesisSystem    — зарождение жизни из биомассы/молний
10. WorldUpdateSystem    — регенерация еды (каждые 2 тика), decay биомассы
```

**Порядок имеет значение**: например, AgingSystem ставит `health=0` перед DeathSystem, а ReproductionSystem работает до DeathSystem, чтобы мёртвые не размножались.

Spatial hash обновляется **инкрементально** между тиками (`_update_spatial_hash_incremental`): перемещённые организмы обновляют позицию, новорождённые добавляются.

---

## 14. Оптимизация

### 14.1 SoA (Structure of Arrays)

Все данные организмов хранятся в параллельных numpy-массивах в `EntityData`:

`x`, `y`, `dx`, `dy`, `energy`, `max_energy`, `health`, `max_health`, `age`, `max_age`, `metabolism`, `size_gene`, `speed_gene`, `vision`, `diet_type`, `efficiency`, `repro_threshold`, `repro_cooldown`, `repro_max_cooldown`, `aggression`, `alive`, `r`, `g`, `b`, `repro_type`, `habitat`

`MAX_ENTITIES = 15000` — предвыделенные массивы. Индексы освобождённых организмов переиспользуются через `_free` list.

### 14.2 Staggered processing

`STAGGER_BATCHES = 2` (хардкод в `sensor_system.py` и `behavior.py`):
- **SensorSystem**: обрабатывает 1/2 организмов за тик
- **BehaviorSystem**: обрабатывает 1/2 организмов за тик

Индекс серии: `idx % 2 == tick % 2`.

### 14.3 Spatial hash

`cell_size = 50`. Операции:
- `insert/remove/update` — O(1) amortized
- `query_nearby` — перебор ячеек в радиусе
- `query_nearby_into` — заполняет переданный `set` (без аллокации нового)
- `query_nearby_excluding_into` — то же с исключением одного ID

Инкрементальное обновление в `main.py`: только перемещённые организмы вызывают `update()`, новорождённые — `insert()`.

### 14.4 Рендеринг

Numpy surfarray для отрисовки мира (массовая запись пикселей). Сущности рендерятся из SoA напрямую (`render_entities_from_soa`), минуя ECS.
