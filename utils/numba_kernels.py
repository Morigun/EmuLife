from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    prange = range


@njit(cache=True)
def find_nearest_food_kernel(food_values, tile_types, x, y, radius, habitat_int, width, height):
    step = 5
    ix = int(x)
    iy = int(y)
    r = int(radius)

    x0 = max(0, ix - r)
    x1 = min(width, ix + r + 1)
    y0 = max(0, iy - r)
    y1 = min(height, iy + r + 1)

    best_dist_sq = 1e18
    best_fx = -1.0
    best_fy = -1.0
    radius_sq = radius * radius

    for sample_y in range(y0, y1, step):
        dy = float(sample_y - iy)
        dy2 = dy * dy
        for sample_x in range(x0, x1, step):
            if food_values[sample_y, sample_x] <= 1.0:
                continue

            is_water = tile_types[sample_y, sample_x] == 0

            if habitat_int == 0 and not is_water:
                continue
            if habitat_int == 1 and is_water:
                continue

            dx = float(sample_x - ix)
            dist_sq = dx * dx + dy2
            if dist_sq > radius_sq:
                continue
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_fx = float(sample_x)
                best_fy = float(sample_y)

    if best_fx < 0:
        return False, 0.0, 0.0
    return True, best_fx, best_fy


@njit(cache=True)
def energy_update_kernel(
    x, y, dx_arr, dy_arr, energy, max_energy, metabolism, size_gene,
    diet_type, habitat, efficiency, alive,
    biomass,
    food_values, tile_types, n, width, height, dt, energy_from_food
):
    total_tiles = width * height

    for i in range(n):
        if not alive[i]:
            continue
        raw_size = (size_gene[i] - 3.0) / 7.0
        cost = metabolism[i] * (0.5 + raw_size) * dt
        speed = (dx_arr[i] ** 2 + dy_arr[i] ** 2) ** 0.5
        cost += speed * raw_size * 0.5 * dt
        if diet_type[i] == 2:
            cost = cost * 0.5
        energy[i] -= cost

        if diet_type[i] == 2:
            ix = min(max(int(x[i]), 0), width - 1)
            iy = min(max(int(y[i]), 0), height - 1)
            bio = biomass[iy, ix]
            if bio > 2.0:
                scav = min(bio * 0.05, 3.0) * dt
                energy[i] += scav
                biomass[iy, ix] -= scav

    entity_count = np.zeros(total_tiles, dtype=np.float32)
    for i in range(n):
        if not alive[i]:
            continue
        ix = min(max(int(x[i]), 0), width - 1)
        iy = min(max(int(y[i]), 0), height - 1)
        entity_count[iy * width + ix] += 1.0

    tile_depletion = np.zeros(total_tiles, dtype=np.float32)
    for i in range(n):
        if not alive[i]:
            continue

        ix = min(max(int(x[i]), 0), width - 1)
        iy = min(max(int(y[i]), 0), height - 1)

        is_water = tile_types[iy, ix] == 0
        non_pred = diet_type[i] != 2
        food = food_values[iy, ix]

        can_eat = non_pred and food > 0 and (
            (habitat[i] == 0 and is_water)
            or (habitat[i] == 1 and not is_water)
            or (habitat[i] == 2)
        )

        if can_eat:
            count = entity_count[iy * width + ix]
            share = food / count if count > 0 else food
            eat_amount = min(energy_from_food * dt, share)
            energy[i] += eat_amount * efficiency[i]
            tile_depletion[iy * width + ix] += eat_amount

    for i in range(n):
        if alive[i]:
            ix = min(max(int(x[i]), 0), width - 1)
            iy = min(max(int(y[i]), 0), height - 1)
            tidx = iy * width + ix
            d = tile_depletion[tidx]
            if d > 0:
                food_values[iy, ix] -= d
                tile_depletion[tidx] = 0.0
                if food_values[iy, ix] < 0:
                    food_values[iy, ix] = 0.0

    for i in range(n):
        if alive[i]:
            cap = max_energy[i] * 1.5 if diet_type[i] == 2 else max_energy[i]
            if energy[i] > cap:
                energy[i] = cap


@njit(parallel=True, cache=True)
def movement_update_kernel(
    x, y, dx_arr, dy_arr, habitat, alive,
    tile_types, n, width, height, dt
):
    moved = np.zeros(n, dtype=np.bool_)
    old_x_out = np.empty(n, dtype=np.float32)
    old_y_out = np.empty(n, dtype=np.float32)

    for i in prange(n):
        old_x_out[i] = x[i]
        old_y_out[i] = y[i]

        if not alive[i]:
            moved[i] = False
            continue

        new_x = x[i] + dx_arr[i] * dt
        new_y = y[i] + dy_arr[i] * dt

        if new_x < 0.0:
            new_x = 0.0
        elif new_x > width - 1:
            new_x = float(width - 1)
        if new_y < 0.0:
            new_y = 0.0
        elif new_y > height - 1:
            new_y = float(height - 1)

        ix = int(new_x)
        iy = int(new_y)
        is_water = tile_types[iy, ix] == 0

        h = habitat[i]
        walkable = (
            (h == 1 and not is_water)
            or (h == 0 and is_water)
            or (h == 2)
        )

        if walkable:
            x[i] = new_x
            y[i] = new_y
            moved[i] = True
        else:
            dx_arr[i] = -dx_arr[i] * 0.5
            dy_arr[i] = -dy_arr[i] * 0.5
            moved[i] = False

    return moved, old_x_out, old_y_out


@njit(cache=True)
def aging_update_kernel(age, max_age, health, alive, n):
    for i in range(n):
        if not alive[i]:
            continue
        age[i] += 1
        if age[i] >= max_age[i]:
            health[i] = 0


@njit(cache=True)
def compute_stats_kernel(diet_type, alive, n):
    herb = 0
    omni = 0
    pred = 0
    for i in range(n):
        if not alive[i]:
            continue
        d = diet_type[i]
        if d == 0:
            herb += 1
        elif d == 1:
            omni += 1
        elif d == 2:
            pred += 1
    return herb, omni, pred


if HAS_NUMBA:
    def warmup():
        n = 4
        f32 = np.zeros(n, dtype=np.float32)
        i8 = np.zeros(n, dtype=np.int8)
        bool_arr = np.ones(n, dtype=np.bool_)
        energy = np.ones(n, dtype=np.float32) * 100.0
        max_e = np.ones(n, dtype=np.float32) * 200.0
        food = np.ones((10, 10), dtype=np.float32) * 50.0
        tiles = np.ones((10, 10), dtype=np.int8)
        bio_arr = np.zeros((10, 10), dtype=np.float32)
        metab = np.ones(n, dtype=np.float32)
        size_g = np.ones(n, dtype=np.float32) * 5.0
        eff = np.ones(n, dtype=np.float32)
        dx = np.zeros(n, dtype=np.float32)
        dy = np.zeros(n, dtype=np.float32)
        age = np.ones(n, dtype=np.int32) * 10
        max_age = np.ones(n, dtype=np.int32) * 1000
        hp = np.ones(n, dtype=np.float32) * 50.0

        energy_update_kernel(
            f32.copy(), f32.copy(), dx.copy(), dy.copy(),
            energy.copy(), max_e.copy(), metab.copy(), size_g.copy(),
            i8.copy(), i8.copy(), eff.copy(), bool_arr.copy(),
            bio_arr.copy(),
            food.copy(), tiles.copy(), n, 10, 10, 1.0, 15.0,
        )

        movement_update_kernel(
            f32.copy(), f32.copy(), dx.copy(), dy.copy(),
            i8.copy(), bool_arr.copy(), tiles.copy(), n, 10, 10, 1.0,
        )

        find_nearest_food_kernel(
            food.copy(), tiles.copy(), 5.0, 5.0, 50.0, 1, 10, 10,
        )

        aging_update_kernel(age.copy(), max_age.copy(), hp.copy(), bool_arr.copy(), n)

        compute_stats_kernel(i8.copy(), bool_arr.copy(), n)
