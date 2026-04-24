from __future__ import annotations


_GENUS_TABLE = {
    (0, 0, 0): "Phytofissa",
    (0, 0, 1): "Chloroclonus",
    (0, 0, 2): "Viridipares",
    (0, 1, 0): "Algamator",
    (0, 1, 1): "Herbivora",
    (0, 1, 2): "Floramixta",
    (1, 0, 0): "Pantofissa",
    (1, 0, 1): "Omnoclonus",
    (1, 0, 2): "Pancipares",
    (1, 1, 0): "Mixamator",
    (1, 1, 1): "Versatilis",
    (1, 1, 2): "Adaptivus",
    (2, 0, 0): "Carnofissa",
    (2, 0, 1): "Voracclonus",
    (2, 0, 2): "Maculapares",
    (2, 1, 0): "Piscamator",
    (2, 1, 1): "Praedator",
    (2, 1, 2): "Dominatus",
    (0, 2, 0): "Ambiflora",
    (0, 2, 1): "Dualclonus",
    (0, 2, 2): "Panciparex",
    (1, 2, 0): "Bivalis",
    (1, 2, 1): "Diplophagus",
    (1, 2, 2): "Amphimixus",
    (2, 2, 0): "Dualcarnis",
    (2, 2, 1): "Bivorax",
    (2, 2, 2): "Dimorphus",
}

_SPECIES_TABLE = {
    (0, 0, 0): "minimus",
    (0, 0, 1): "lenis",
    (0, 0, 2): "pumilus",
    (0, 1, 0): "gracilis",
    (0, 1, 1): "modicus",
    (0, 1, 2): "velox",
    (0, 2, 0): "tenuis",
    (0, 2, 1): "agilis",
    (0, 2, 2): "ferratus",
    (1, 0, 0): "pacatus",
    (1, 0, 1): "tranquillus",
    (1, 0, 2): "medioximus",
    (1, 1, 0): "medius",
    (1, 1, 1): "vulgaris",
    (1, 1, 2): "ferox",
    (1, 2, 0): "mitis",
    (1, 2, 1): "celer",
    (1, 2, 2): "impiger",
    (2, 0, 0): "gigas",
    (2, 0, 1): "maior",
    (2, 0, 2): "robustus",
    (2, 1, 0): "magnus",
    (2, 1, 1): "validus",
    (2, 1, 2): "fortis",
    (2, 2, 0): "colossus",
    (2, 2, 1): "grandis",
    (2, 2, 2): "titanius",
}

_DIET_NAMES = {0: "Herbivore", 1: "Omnivore", 2: "Predator", 3: "Carn.Plant"}
_REPRO_NAMES = {0: "Asexual", 1: "Sexual", 2: "Hermaphro"}
_HABITAT_NAMES = {0: "Aquatic", 1: "Terrestrial", 2: "Amphibious"}
_ORIGIN_NAMES = {0: "Abiogenesis", 1: "Birth (asexual)", 2: "Birth (sexual)"}


def _quantize(value: float, low: float = 0.33, high: float = 0.66) -> int:
    if value < low:
        return 0
    elif value < high:
        return 1
    return 2


def get_species_name(genes) -> str:
    if hasattr(genes, "genes"):
        g = genes.genes
    else:
        g = list(genes)

    diet = _quantize(g[4])
    repro = 0 if g[10] < 0.3 else (2 if g[10] < 0.7 else 1)
    habitat = _quantize(g[11])

    genus = _GENUS_TABLE.get((diet, repro, habitat), "Incertae")

    size = _quantize(g[0])
    speed = _quantize(g[1])
    aggression = _quantize(g[6])

    species = _SPECIES_TABLE.get((size, speed, aggression), "incertae")

    return f"{genus} {species}"


def get_species_name_from_soa_data(
    diet_type: int,
    repro_type: int,
    habitat: int,
    size_gene: float,
    speed_gene: float,
    aggression: float,
) -> str:
    genus = _GENUS_TABLE.get((int(diet_type), int(repro_type), int(habitat)), "Incertae")

    size = _quantize((size_gene - 3.0) / 7.0)
    speed = _quantize((speed_gene - 1.0) / 4.0)
    aggro = _quantize(aggression)

    species = _SPECIES_TABLE.get((size, speed, aggro), "incertae")

    return f"{genus} {species}"


def get_diet_name(diet_type: int) -> str:
    return _DIET_NAMES.get(int(diet_type), "Unknown")


def get_repro_name(repro_type: int) -> str:
    return _REPRO_NAMES.get(int(repro_type), "Unknown")


def get_habitat_name(habitat: int) -> str:
    return _HABITAT_NAMES.get(int(habitat), "Unknown")


def get_origin_name(origin: int) -> str:
    return _ORIGIN_NAMES.get(int(origin), "Unknown")
